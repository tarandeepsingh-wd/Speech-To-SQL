from __future__ import annotations

import re
from typing import Any


def resolve_time_filter(
    tag_result: dict[str, Any],
    text: str,
    config: dict[str, Any],
) -> tuple[str, float | None, float | None] | None:
    explicit = tag_result.get("time_filters") or []
    if explicit:
        item = explicit[0]
        return (
            item.get("field") or config["time_fields"]["default"],
            item.get("min"),
            item.get("max"),
        )

    time_config = config["time_fields"]
    terms = time_config["terms"]
    if not any(term in text for term in terms):
        return None

    field_name = time_config["default"]
    for term, candidate in terms.items():
        if re.search(rf"\b{re.escape(term)}\b", text):
            field_name = candidate
            break

    if _contains_config_term(text, time_config["quick_terms"]):
        return field_name, 0, float(time_config["quick_max_minutes"])
    return None


def _contains_config_term(text: str, terms: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)
