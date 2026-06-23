from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

from speech_to_sql.common.config import load_recipe_db_config, load_sql_schema_config
from speech_to_sql.m3_query_ir.schema_linker import link_base_field, link_nutrition_field


@dataclass(frozen=True)
class IntentEnrichment:
    target: str
    added: dict[str, Any]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "added": self.added,
            "reason": self.reason,
        }


def enrich_tag_result(
    tag_result: dict[str, Any],
    *,
    transcript: str | None = None,
    recipe_config: dict[str, Any] | None = None,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if tag_result.get("_intent_enriched"):
        return tag_result

    recipe_config = recipe_config or load_recipe_db_config()
    schema = schema or load_sql_schema_config()
    enriched = copy.deepcopy(tag_result)
    transcript = transcript or enriched.get("transcript", "") or enriched.get("query", "")
    text = _normalize_text(transcript)
    enrichments: list[IntentEnrichment] = []

    _enrich_nutrition_filters(enriched, text, recipe_config, schema, enrichments)
    _enrich_time_filters(enriched, transcript, text, recipe_config, schema, enrichments)

    if transcript and not enriched.get("transcript"):
        enriched["transcript"] = transcript
    enriched["_intent_enriched"] = True
    enriched["_intent_enrichments"] = [item.as_dict() for item in enrichments]
    return enriched


def _enrich_nutrition_filters(
    tag_result: dict[str, Any],
    text: str,
    recipe_config: dict[str, Any],
    schema: dict[str, Any],
    enrichments: list[IntentEnrichment],
) -> None:
    filters = [
        dict(item)
        for item in tag_result.get("nutrition_filters", []) or []
        if isinstance(item, dict)
    ]
    existing = {
        nutrient
        for nutrient in (
            link_nutrition_field(item.get("nutrient"), recipe_config=recipe_config, schema=schema)
            for item in filters
        )
        if nutrient
    }
    consumed_terms = list(tag_result.get("_consumed_nutrition_terms") or [])

    for rule in recipe_config["intent_enrichment"]["nutrition_qualifiers"]:
        matched_terms = _matched_terms(text, rule["terms"])
        if not matched_terms:
            continue
        nutrient = link_nutrition_field(
            rule["nutrient"],
            recipe_config=recipe_config,
            schema=schema,
        )
        if not nutrient:
            continue
        replacement = _generic_presence_filter_index(filters, nutrient, recipe_config, schema)
        if nutrient in existing and replacement is None:
            continue
        added = {
            "nutrient": nutrient,
            "operator": rule["operator"],
            "value": rule["value"],
            "source": "intent_enrichment",
        }
        if replacement is None:
            filters.append(added)
        else:
            filters[replacement] = {**filters[replacement], **added}
        existing.add(nutrient)
        consumed_terms.extend(matched_terms)
        consumed_terms.append(str(rule["nutrient"]))
        enrichments.append(IntentEnrichment(
            target="nutrition_filters",
            added=added,
            reason="intent_enrichment.nutrition_qualifiers",
        ))

    tag_result["nutrition_filters"] = filters
    tag_result["_consumed_nutrition_terms"] = _unique_terms(consumed_terms)


def _generic_presence_filter_index(
    filters: list[dict[str, Any]],
    nutrient: str,
    recipe_config: dict[str, Any],
    schema: dict[str, Any],
) -> int | None:
    for index, item in enumerate(filters):
        resolved = link_nutrition_field(
            item.get("nutrient"),
            recipe_config=recipe_config,
            schema=schema,
        )
        if resolved != nutrient:
            continue
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            continue
        operator = str(item.get("operator") or "").strip()
        if operator == ">" and value == 0:
            return index
    return None


def _enrich_time_filters(
    tag_result: dict[str, Any],
    transcript: str,
    text: str,
    recipe_config: dict[str, Any],
    schema: dict[str, Any],
    enrichments: list[IntentEnrichment],
) -> None:
    if tag_result.get("time_filters"):
        return

    patterns = recipe_config["intent_enrichment"]["time_patterns"]
    time_filter = _time_filter_from_patterns(transcript, patterns, recipe_config)
    if not time_filter:
        return

    field = _time_field_from_text(text, recipe_config, schema)
    added = {"field": field, **time_filter, "source": "intent_enrichment"}
    tag_result["time_filters"] = [added]
    enrichments.append(IntentEnrichment(
        target="time_filters",
        added=added,
        reason="intent_enrichment.time_patterns",
    ))


def _time_filter_from_patterns(
    transcript: str,
    patterns: dict[str, str],
    recipe_config: dict[str, Any],
) -> dict[str, float | None] | None:
    between = re.search(patterns["between"], transcript, re.IGNORECASE)
    if between:
        return {
            "min": _normalize_minutes(float(between.group(1)), transcript, recipe_config),
            "max": _normalize_minutes(float(between.group(2)), transcript, recipe_config),
        }

    upper = re.search(patterns["upper"], transcript, re.IGNORECASE)
    if upper:
        return {
            "min": 0,
            "max": _normalize_minutes(float(upper.group(1)), transcript, recipe_config),
        }

    lower = re.search(patterns["lower"], transcript, re.IGNORECASE)
    if lower:
        return {
            "min": _normalize_minutes(float(lower.group(1)), transcript, recipe_config),
            "max": None,
        }
    return None


def _time_field_from_text(
    text: str,
    recipe_config: dict[str, Any],
    schema: dict[str, Any],
) -> str:
    explicit_patterns = (
        (r"\b(?:cook|cooking|cook_time)\s+time\b|\bcook_time\b", "cook_time"),
        (r"\b(?:prep|prepare|prepared|preparation|prep_time)\s+time\b|\bprep_time\b", "prep_time"),
        (r"\b(?:total|overall|total_time)\s+time\b|\btotal_time\b", "total_time"),
    )
    for pattern, field in explicit_patterns:
        if re.search(pattern, text):
            return link_base_field(field, recipe_config=recipe_config, schema=schema) or field
    return recipe_config["time_fields"]["default"]


def _normalize_minutes(value: float, transcript: str, recipe_config: dict[str, Any]) -> float:
    text = _normalize_text(transcript)
    if _contains_any_term(text, recipe_config["time_fields"]["hour_terms"]):
        return value * 60
    return value


def _contains_any_term(text: str, terms: list[str]) -> bool:
    return bool(_matched_terms(text, terms))


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    return [
        _normalize_text(term)
        for term in terms
        if re.search(rf"\b{re.escape(_normalize_text(term))}\b", text)
    ]


def _unique_terms(terms: list[str]) -> list[str]:
    result = []
    seen = set()
    for term in terms:
        normalized = _normalize_text(term)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())).strip()
