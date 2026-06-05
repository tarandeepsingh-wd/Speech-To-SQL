from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

from module3.config import load_recipe_db_config, load_sql_schema_config
from module3.schema_linker import link_base_field, link_nutrition_field


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
        link_nutrition_field(item.get("nutrient"), recipe_config=recipe_config, schema=schema)
        for item in filters
    }

    for rule in recipe_config["intent_enrichment"]["nutrition_qualifiers"]:
        if not _contains_any_term(text, rule["terms"]):
            continue
        nutrient = link_nutrition_field(
            rule["nutrient"],
            recipe_config=recipe_config,
            schema=schema,
        )
        if not nutrient or nutrient in existing:
            continue
        added = {
            "nutrient": nutrient,
            "operator": rule["operator"],
            "value": rule["value"],
            "source": "intent_enrichment",
        }
        filters.append(added)
        existing.add(nutrient)
        enrichments.append(IntentEnrichment(
            target="nutrition_filters",
            added=added,
            reason="intent_enrichment.nutrition_qualifiers",
        ))

    tag_result["nutrition_filters"] = filters


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
    terms = recipe_config["time_fields"]["terms"]
    for term, field in sorted(terms.items(), key=lambda item: len(item[0]), reverse=True):
        if _contains_any_term(text, [term]):
            return link_base_field(field, recipe_config=recipe_config, schema=schema) or field
    return recipe_config["time_fields"]["default"]


def _normalize_minutes(value: float, transcript: str, recipe_config: dict[str, Any]) -> float:
    text = _normalize_text(transcript)
    if _contains_any_term(text, recipe_config["time_fields"]["hour_terms"]):
        return value * 60
    return value


def _contains_any_term(text: str, terms: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(_normalize_text(term))}\b", text) for term in terms)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())).strip()
