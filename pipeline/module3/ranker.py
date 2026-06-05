from __future__ import annotations

import re
from typing import Any

from module3.config import load_recipe_db_config


def rank_and_limit(
    recipes: list[dict],
    tag_result: dict,
    *,
    transcript: str | None = None,
    default_limit: int | None = None,
) -> list[dict]:
    transcript = transcript or tag_result.get("transcript", "") or tag_result.get("query", "")
    config = load_recipe_db_config()
    sort_key, descending = _sort_intent(tag_result, transcript.lower(), config)
    ranked = list(recipes)

    if sort_key:
        ranked.sort(
            key=lambda recipe: _sortable_value(recipe, sort_key, config),
            reverse=descending,
        )

    limit = _limit(tag_result, transcript, default_limit)
    if limit:
        ranked = ranked[:limit]
    return ranked


def _sort_intent(
    tag_result: dict,
    text: str,
    config: dict[str, Any],
) -> tuple[str | None, bool]:
    order_ops = [
        op for op in tag_result.get("sql_operators", []) or []
        if str(op.get("canonical", "")).startswith("ORDER BY")
    ]
    sort_cue = _contains_config_term(text, config["sort_cues"])
    if not order_ops and not sort_cue:
        return None, False

    sort_key = None
    for candidate, terms in config["sort_terms"].items():
        if _contains_config_term(text, terms):
            sort_key = candidate
            break

    if not sort_key:
        return None, False

    descending = False
    direction_cues = config["sort_direction_cues"]
    if _contains_config_term(text, direction_cues["desc"]):
        descending = True
    if _contains_config_term(text, direction_cues["asc"]):
        descending = False
    if sort_key == "protein" and _contains_config_term(text, config["protein_desc_cues"]):
        descending = True

    if order_ops:
        canonical = str(order_ops[0].get("canonical", ""))
        if canonical.endswith("DESC"):
            descending = True
        elif canonical.endswith("ASC"):
            descending = False

    return sort_key, descending


def _limit(tag_result: dict, transcript: str, default_limit: int | None) -> int | None:
    for op in tag_result.get("sql_operators", []) or []:
        if op.get("canonical") != "LIMIT":
            continue
        match = re.search(r"\d+", str(op.get("text") or ""))
        if match:
            return int(match.group(0))

    match = re.search(r"\b(?:top|first|limit)\s+(\d+)\b", transcript, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return default_limit


def _sortable_value(recipe: dict, sort_key: str, config: dict[str, Any]) -> float:
    aliases = config["response_field_aliases"]
    for field in aliases.get(sort_key, [sort_key]):
        value = _to_float(recipe.get(field))
        if value is not None:
            return value
    return float("-inf")


def _contains_config_term(text: str, terms: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))
