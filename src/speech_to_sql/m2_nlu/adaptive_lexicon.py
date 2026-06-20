from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speech_to_sql.m2_nlu.semantic_normalizer import normalize_schema_terms


FEEDBACK_DIR = Path(__file__).parent / "data" / "feedback"
DEFAULT_TRANSCRIPT_FEEDBACK_FILE = FEEDBACK_DIR / "transcript_feedback.jsonl"
TRANSCRIPT_FEEDBACK_FILE_ENV = "SPEECH_TO_SQL_TRANSCRIPT_FEEDBACK_FILE"
MIN_COUNT_ENV = "SPEECH_TO_SQL_ADAPTIVE_MIN_COUNT"

_CACHE: dict[str, Any] = {
    "path": None,
    "mtime": None,
    "min_count": None,
    "replacements": None,
}


@dataclass(frozen=True)
class AppliedCorrection:
    source: str
    target: str
    count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "count": self.count,
        }


def transcript_feedback_path() -> Path:
    configured = os.getenv(TRANSCRIPT_FEEDBACK_FILE_ENV)
    return Path(configured) if configured else DEFAULT_TRANSCRIPT_FEEDBACK_FILE


def reset_adaptive_cache() -> None:
    _CACHE.update({
        "path": None,
        "mtime": None,
        "min_count": None,
        "replacements": None,
    })


def normalize_noisy_text(text: str) -> dict[str, Any]:
    """
    Apply user-approved transcript corrections before NLU extraction.

    Learned replacements come from explicit feedback. Schema/operator
    replacements are generated from existing config lexicons and only apply
    when similarity plus context is strong enough.
    """
    replacements = learned_replacements()
    updated = text
    corrections: list[dict[str, Any]] = []
    for source, target, count in replacements:
        pattern = _replacement_pattern(source)
        if not pattern.search(updated):
            continue
        updated = pattern.sub(target, updated)
        corrections.append(AppliedCorrection(source=source, target=target, count=count).as_dict())

    schema_result = normalize_schema_terms(updated)
    updated = schema_result["text"]
    corrections.extend(schema_result["corrections"])

    return {
        "text": updated,
        "changed": updated != text,
        "corrections": corrections,
    }


def learned_replacements(min_count: int | None = None) -> list[tuple[str, str, int]]:
    path = transcript_feedback_path()
    threshold = min_count if min_count is not None else _min_count()
    mtime = path.stat().st_mtime if path.exists() else None
    if (
        _CACHE["path"] == path
        and _CACHE["mtime"] == mtime
        and _CACHE["min_count"] == threshold
    ):
        return list(_CACHE["replacements"] or [])

    rows = _load_feedback_rows(path)
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        if row.get("approved") is False:
            continue
        row_corrections = row.get("corrections")
        if not isinstance(row_corrections, list):
            row_corrections = derive_transcript_corrections(
                str(row.get("original_query") or row.get("query") or ""),
                str(row.get("corrected_query") or ""),
            )
        for item in row_corrections:
            source = _normalize_phrase(str(item.get("source") or item.get("from") or ""))
            target = _normalize_phrase(str(item.get("target") or item.get("to") or ""))
            if not _valid_replacement(source, target):
                continue
            counts[(source, target)] = counts.get((source, target), 0) + 1

    replacements = [
        (source, target, count)
        for (source, target), count in counts.items()
        if count >= threshold
    ]
    replacements.sort(key=lambda item: (-len(item[0].split()), -item[2], item[0]))
    _CACHE.update({
        "path": path,
        "mtime": mtime,
        "min_count": threshold,
        "replacements": replacements,
    })
    return list(replacements)


def derive_transcript_corrections(original: str, corrected: str) -> list[dict[str, str]]:
    original_tokens = _tokens(original)
    corrected_tokens = _tokens(corrected)
    if not original_tokens or not corrected_tokens:
        return []

    matcher = difflib.SequenceMatcher(a=original_tokens, b=corrected_tokens, autojunk=False)
    corrections: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag not in {"replace", "delete", "insert"}:
            continue
        source_tokens = original_tokens[i1:i2]
        target_tokens = corrected_tokens[j1:j2]
        if not source_tokens or not target_tokens:
            continue
        source = " ".join(source_tokens)
        target = " ".join(target_tokens)
        if not _valid_replacement(source, target):
            continue
        key = (source, target)
        if key in seen:
            continue
        corrections.append({"source": source, "target": target})
        seen.add(key)
    return corrections


def feedback_summary(limit: int = 6) -> dict[str, Any]:
    path = transcript_feedback_path()
    rows = _load_feedback_rows(path)
    replacements = learned_replacements()
    latest = rows[-limit:] if limit > 0 else []
    return {
        "feedback_count": len(rows),
        "learned_replacements": [
            {"source": source, "target": target, "count": count}
            for source, target, count in replacements[:limit]
        ],
        "latest": [
            {
                "created_at": row.get("created_at"),
                "original_query": row.get("original_query") or row.get("query"),
                "corrected_query": row.get("corrected_query"),
                "corrected_ingredients": row.get("corrected_ingredients", []),
            }
            for row in latest
        ],
    }


def _load_feedback_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _min_count() -> int:
    raw = os.getenv(MIN_COUNT_ENV)
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 1


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z-]{1,}|\d+(?:\.\d+)?", text.lower())


def _normalize_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s-]+", " ", value.lower())).strip()


def _valid_replacement(source: str, target: str) -> bool:
    source = _normalize_phrase(source)
    target = _normalize_phrase(target)
    if not source or not target or source == target:
        return False
    if len(source) < 3 or len(target) < 3:
        return False
    if len(source.split()) > 4 or len(target.split()) > 4:
        return False
    if not re.fullmatch(r"[a-z0-9][a-z0-9\s-]*", source):
        return False
    if not re.fullmatch(r"[a-z0-9][a-z0-9\s-]*", target):
        return False
    return True


def _replacement_pattern(source: str) -> re.Pattern:
    words = [re.escape(part) for part in source.split()]
    return re.compile(r"\b" + r"[\s-]+".join(words) + r"\b", re.IGNORECASE)
