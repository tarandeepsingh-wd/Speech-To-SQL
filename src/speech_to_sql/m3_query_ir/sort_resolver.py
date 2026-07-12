from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz


def resolve_sort_intent(
    text: str,
    config: dict[str, Any],
    *,
    order_ops: list[dict[str, Any]] | None = None,
    default_direction: str = "ASC",
) -> tuple[str | None, str]:
    order_ops = order_ops or []
    if (
        not order_ops
        and not _contains_config_term(text, config["sort_cues"])
        and not _quickest_sort_context(text)
    ):
        return None, default_direction

    sort_field = _best_sort_field(text, config, order_ops)
    if not sort_field and _quickest_sort_context(text):
        sort_field = "total_time"
    if not sort_field:
        return None, default_direction
    if sort_field == "time":
        sort_field = "total_time"

    direction = _sort_direction(text, config, sort_field, default_direction)
    for op in order_ops:
        canonical = str(op.get("canonical", ""))
        if canonical.endswith("DESC"):
            return sort_field, "DESC"
        if canonical.endswith("ASC"):
            return sort_field, "ASC"
    return sort_field, direction


def _sort_direction(
    text: str,
    config: dict[str, Any],
    sort_field: str,
    default_direction: str,
) -> str:
    phrase_cues = config.get("sort_direction_phrases", {})
    if _contains_config_term(text, phrase_cues.get("desc", [])):
        return "DESC"
    if _contains_config_term(text, phrase_cues.get("asc", [])):
        return "ASC"

    cues = config["sort_direction_cues"]
    desc_match = _last_config_term_position(text, cues["desc"])
    asc_match = _last_config_term_position(text, cues["asc"])
    if desc_match is not None and asc_match is not None:
        return "DESC" if desc_match > asc_match else "ASC"
    if desc_match is not None:
        return "DESC"
    if asc_match is not None:
        return "ASC"
    if sort_field == "protein" and _contains_config_term(text, config["protein_desc_cues"]):
        return "DESC"
    if sort_field in {"time", "total_time"} and _quickest_sort_context(text):
        return "ASC"
    return default_direction


def _best_config_key(text: str, terms_by_key: dict[str, list[str]]) -> str | None:
    best_key = None
    best_score: tuple[int, int] | None = None
    for key, terms in terms_by_key.items():
        for term in terms:
            match = re.search(rf"\b{re.escape(term)}\b", text)
            if not match:
                continue
            score = (len(term.split()), len(term))
            if best_score is None or score > best_score:
                best_key = key
                best_score = score
    return best_key


def _best_sort_field(
    text: str,
    config: dict[str, Any],
    order_ops: list[dict[str, Any]],
) -> str | None:
    cue_positions = _sort_cue_positions(text, config, order_ops)
    candidates = []
    for key, terms in config["sort_terms"].items():
        for term in terms:
            for position in _config_term_positions(text, term):
                distance = min((abs(position - cue) for cue in cue_positions), default=10_000)
                candidates.append((
                    distance // 16,
                    -len(term.split()),
                    -len(term),
                    distance,
                    position,
                    key,
                ))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][-1]


def _sort_cue_positions(
    text: str,
    config: dict[str, Any],
    order_ops: list[dict[str, Any]],
) -> list[int]:
    positions: list[int] = []
    for op in order_ops:
        span = op.get("span")
        if isinstance(span, (list, tuple)) and len(span) == 2:
            try:
                positions.append(int(span[0]))
            except (TypeError, ValueError):
                pass
        op_text = str(op.get("text") or "").lower()
        if op_text:
            positions.extend(_config_term_positions(text, op_text))

    for term in config.get("sort_cues", []):
        positions.extend(_config_term_positions(text, term))
    for term in config.get("time_fields", {}).get("quick_terms", []):
        if term in {"quickest", "fastest", "shortest"}:
            positions.extend(_config_term_positions(text, term))
    for terms in config.get("sort_direction_phrases", {}).values():
        for term in terms:
            positions.extend(_config_term_positions(text, term))
    return positions


def _quickest_sort_context(text: str) -> bool:
    return bool(re.search(r"\b(?:quickest|fastest|shortest)\b", text))


def _contains_config_term(text: str, terms: list[str]) -> bool:
    return any(_config_term_positions(text, term) for term in terms)


def _last_config_term_position(text: str, terms: list[str]) -> int | None:
    positions = [
        position
        for term in terms
        for position in _config_term_positions(text, term)
    ]
    if not positions:
        return None
    return max(positions)


def _config_term_positions(text: str, term: str) -> list[int]:
    positions = []
    saw_exact = False
    for match in re.finditer(rf"\b{re.escape(term)}\b", text):
        saw_exact = True
        prefix = text[max(0, match.start() - 3):match.start()]
        if term in {"least", "most"} and prefix == "at ":
            continue
        if term in {"min", "minimum", "max", "maximum"} and _followed_by_numeric_value(text, match.end()):
            continue
        positions.append(match.start())
    if positions:
        return positions
    if saw_exact:
        return []
    return _fuzzy_config_term_positions(text, term)


def _fuzzy_config_term_positions(text: str, term: str) -> list[int]:
    term_tokens = re.findall(r"[a-z0-9]+", term.lower())
    if not term_tokens or all(len(token) < 5 for token in term_tokens):
        return []

    text_tokens = [
        (match.group(0), match.start())
        for match in re.finditer(r"[a-z0-9]+", text.lower())
    ]
    if len(term_tokens) > len(text_tokens):
        return []

    positions: list[int] = []
    for start in range(0, len(text_tokens) - len(term_tokens) + 1):
        window = text_tokens[start:start + len(term_tokens)]
        if all(_fuzzy_token_match(actual, expected) for (actual, _pos), expected in zip(window, term_tokens)):
            positions.append(window[0][1])
    return positions


def _fuzzy_token_match(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    if len(expected) < 5 or len(actual) < 4:
        return False
    threshold = 83 if max(len(actual), len(expected)) >= 6 else 86
    return fuzz.ratio(actual, expected) >= threshold


def _followed_by_numeric_value(text: str, end: int) -> bool:
    return bool(re.match(r"\s+\d+(?:\.\d+)?\b", text[end:]))
