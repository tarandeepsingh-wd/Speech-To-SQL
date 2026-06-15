from __future__ import annotations

import re
from typing import Any


def resolve_sort_intent(
    text: str,
    config: dict[str, Any],
    *,
    order_ops: list[dict[str, Any]] | None = None,
    default_direction: str = "ASC",
) -> tuple[str | None, str]:
    order_ops = order_ops or []
    if not order_ops and not _contains_config_term(text, config["sort_cues"]):
        return None, default_direction

    sort_field = _best_config_key(text, config["sort_terms"])
    if not sort_field:
        return None, default_direction

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


def _contains_config_term(text: str, terms: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _last_config_term_position(text: str, terms: list[str]) -> int | None:
    positions = [
        match.start()
        for term in terms
        for match in re.finditer(rf"\b{re.escape(term)}\b", text)
    ]
    if not positions:
        return None
    return max(positions)
