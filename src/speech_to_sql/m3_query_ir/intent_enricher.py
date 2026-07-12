from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from speech_to_sql.common.config import load_recipe_db_config, load_sql_schema_config
from speech_to_sql.m3_query_ir.schema_linker import link_base_field, link_nutrition_field
from speech_to_sql.m3_query_ir.sort_resolver import resolve_sort_intent


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


_TIME_BASE_OP_MAP = {
    "less than or equal": "<=",
    "greater than or equal": ">=",
    "more than or equal": ">=",
    "under": "<",
    "below": "<",
    "less than": "<",
    "fewer than": "<",
    "within": "<=",
    "at most": "<=",
    "max": "<=",
    "maximum": "<=",
    "over": ">",
    "above": ">",
    "more than": ">",
    "greater than": ">",
    "higher than": ">",
    "larger than": ">",
    "at least": ">=",
    "min": ">=",
    "minimum": ">=",
    "around": "~",
    "about": "~",
    "approximately": "~",
}
_TIME_NEGATED_OPERATOR_RE = re.compile(r"^(?:not|no)\s+(.+)$", re.IGNORECASE)
_TIME_NEGATION_MAP = {
    "<": ">=",
    "<=": ">",
    ">": "<=",
    ">=": "<",
}


def _operator_pattern(terms: list[str]) -> str:
    parts = []
    for term in sorted(terms, key=len, reverse=True):
        tokens = re.findall(r"[a-z0-9]+", term.lower())
        if tokens:
            parts.append(r"\s+".join(re.escape(token) for token in tokens))
    return "|".join(parts)


_TIME_OPERATOR_PATTERN = _operator_pattern(list(_TIME_BASE_OP_MAP))
_TIME_COMPARISON_PATTERN = re.compile(
    rf"\b(?P<op>(?:(?:not|no)\s+)?(?:{_TIME_OPERATOR_PATTERN}))\s+"
    rf"(?P<value>\d+(?:\.\d+)?)\s*"
    rf"(?:minutes?|mins?|min|hours?|hrs?|hr)\b",
    re.IGNORECASE,
)


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

    _enrich_dietary_classifications(enriched, recipe_config, enrichments)
    _enrich_nutrition_filters(enriched, text, recipe_config, schema, enrichments)
    _enrich_time_filters(enriched, transcript, text, recipe_config, schema, enrichments)

    if transcript and not enriched.get("transcript"):
        enriched["transcript"] = transcript
    enriched["_intent_enriched"] = True
    enriched["_intent_enrichments"] = [item.as_dict() for item in enrichments]
    return enriched


def _enrich_dietary_classifications(
    tag_result: dict[str, Any],
    recipe_config: dict[str, Any],
    enrichments: list[IntentEnrichment],
) -> None:
    rules = recipe_config.get("intent_enrichment", {}).get("dietary_classifications", [])
    if not rules:
        return

    styles = _unique_preserving(tag_result.get("dietary_styles") or [])
    excluded = _unique_preserving(tag_result.get("excluded_dietary_styles") or [])
    changed = False

    for rule in rules:
        rule_styles = {
            _normalize_text(str(style))
            for style in rule.get("styles", [])
            if str(style).strip()
        }
        if not rule_styles:
            continue

        positive_matches = [
            style for style in styles
            if _normalize_text(str(style)) in rule_styles
        ]
        if positive_matches:
            styles = [
                style for style in styles
                if _normalize_text(str(style)) not in rule_styles
            ]
            excluded.extend(str(item) for item in rule.get("exclude", []) if str(item).strip())
            changed = True
            enrichments.append(IntentEnrichment(
                target="excluded_dietary_styles",
                added={
                    "from": positive_matches,
                    "exclude": list(rule.get("exclude", [])),
                },
                reason="intent_enrichment.dietary_classifications",
            ))

        negated_matches = [
            style for style in excluded
            if _normalize_text(str(style)) in rule_styles
        ]
        if negated_matches:
            excluded = [
                style for style in excluded
                if _normalize_text(str(style)) not in rule_styles
            ]
            styles.extend(str(item) for item in rule.get("negated_include", []) if str(item).strip())
            changed = True
            enrichments.append(IntentEnrichment(
                target="dietary_styles",
                added={
                    "from_negated": negated_matches,
                    "include": list(rule.get("negated_include", [])),
                },
                reason="intent_enrichment.dietary_classifications",
            ))

    if changed:
        tag_result["dietary_styles"] = _unique_preserving(styles)
        tag_result["excluded_dietary_styles"] = _unique_preserving(excluded)


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
    filters = _remove_order_only_presence_filters(
        filters,
        text,
        recipe_config,
        schema,
        enrichments,
    )
    existing = {
        nutrient
        for nutrient in (
            link_nutrition_field(item.get("nutrient"), recipe_config=recipe_config, schema=schema)
            for item in filters
        )
        if nutrient
    }
    consumed_terms = list(tag_result.get("_consumed_nutrition_terms") or [])

    for rule in recipe_config.get("intent_enrichment", {}).get("nutrition_qualifiers", []):
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
        removed_presence_nutrients = _suppress_generic_presence_filters(
            filters,
            rule.get("suppress_presence_nutrients", []),
            recipe_config,
            schema,
            enrichments,
        )
        for removed_nutrient in removed_presence_nutrients:
            existing.discard(removed_nutrient)
        consumed_terms.extend(matched_terms)
        consumed_terms.append(str(rule["nutrient"]))
        enrichments.append(IntentEnrichment(
            target="nutrition_filters",
            added=added,
            reason="intent_enrichment.nutrition_qualifiers",
        ))

    tag_result["nutrition_filters"] = filters
    tag_result["_consumed_nutrition_terms"] = _unique_terms(consumed_terms)


def _suppress_generic_presence_filters(
    filters: list[dict[str, Any]],
    nutrients: list[Any],
    recipe_config: dict[str, Any],
    schema: dict[str, Any],
    enrichments: list[IntentEnrichment],
) -> list[str]:
    removed_nutrients: list[str] = []
    for term in nutrients or []:
        nutrient = link_nutrition_field(term, recipe_config=recipe_config, schema=schema)
        if not nutrient:
            continue
        replacement = _generic_presence_filter_index(filters, nutrient, recipe_config, schema)
        if replacement is None:
            continue
        removed = filters.pop(replacement)
        removed_nutrients.append(nutrient)
        enrichments.append(IntentEnrichment(
            target="nutrition_filters",
            added={"removed_generic_presence_filter": removed},
            reason="intent_enrichment.suppress_presence_nutrients",
        ))
    return removed_nutrients


def _remove_order_only_presence_filters(
    filters: list[dict[str, Any]],
    text: str,
    recipe_config: dict[str, Any],
    schema: dict[str, Any],
    enrichments: list[IntentEnrichment],
) -> list[dict[str, Any]]:
    if not _contains_any_term(text, recipe_config.get("sort_cues", [])):
        return filters

    sort_field, _direction = resolve_sort_intent(text, recipe_config)
    sort_nutrient = link_nutrition_field(sort_field, recipe_config=recipe_config, schema=schema)
    if not sort_nutrient:
        return filters

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for item in filters:
        nutrient = link_nutrition_field(
            item.get("nutrient"),
            recipe_config=recipe_config,
            schema=schema,
        )
        operator = str(item.get("operator") or "").strip()
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            value = None
        if nutrient == sort_nutrient and operator == ">" and value == 0:
            removed.append(dict(item))
            continue
        kept.append(item)

    if removed:
        enrichments.append(IntentEnrichment(
            target="nutrition_filters",
            added={"removed_order_only_presence_filters": removed},
            reason="intent_enrichment.order_only_presence_cleanup",
        ))
    return kept


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
    time_filters = _time_filters_from_patterns(
        transcript,
        patterns,
        recipe_config,
        schema,
    )
    if not time_filters:
        return

    tag_result["time_filters"] = time_filters
    for added in time_filters:
        enrichments.append(IntentEnrichment(
            target="time_filters",
            added=added,
            reason="intent_enrichment.time_patterns",
        ))


def _time_filters_from_patterns(
    transcript: str,
    patterns: dict[str, str],
    recipe_config: dict[str, Any],
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    used_positions: set[int] = set()
    text = _normalize_text(transcript)

    for between in re.finditer(patterns["between"], transcript, re.IGNORECASE):
        used_positions.update(range(between.start(), between.end()))
        found.append({
            "field": _time_field_from_span(transcript, between.span(), recipe_config, schema),
            "min": _normalize_minutes(float(between.group(1)), transcript, recipe_config),
            "max": _normalize_minutes(float(between.group(2)), transcript, recipe_config),
            "source": "intent_enrichment",
        })

    for comparison in _TIME_COMPARISON_PATTERN.finditer(transcript):
        positions = set(range(comparison.start(), comparison.end()))
        if positions & used_positions:
            continue
        value = _normalize_minutes(float(comparison.group("value")), transcript, recipe_config)
        operator = _time_operator_from_text(comparison.group("op"))
        bounds = _time_bounds_from_operator(operator, value)
        if not bounds:
            continue
        used_positions |= positions
        found.append({
            "field": _time_field_from_span(transcript, comparison.span(), recipe_config, schema),
            **bounds,
            "source": "intent_enrichment",
        })

    time_config = recipe_config["time_fields"]
    if not found and _contains_any_term(text, time_config.get("quick_sort_terms", [])):
        return found

    if not found and _contains_any_term(text, time_config.get("quick_prep_terms", [])):
        found.append({
            "field": link_base_field("prep_time", recipe_config=recipe_config, schema=schema) or "prep_time",
            "min": 0,
            "max": float(time_config.get("quick_prep_max_minutes", 20)),
            "source": "intent_enrichment",
        })

    if not found and _contains_any_term(text, time_config["quick_terms"]):
        field = _time_field_from_text(text, recipe_config, schema)
        found.append({
            "field": field,
            "min": 0,
            "max": float(time_config["quick_max_minutes"]),
            "source": "intent_enrichment",
        })
    return _dedupe_time_filters(found)


def _time_operator_from_text(op_text: str | None) -> str:
    normalized = re.sub(r"\s+", " ", (op_text or "").lower()).strip()
    negated = _TIME_NEGATED_OPERATOR_RE.match(normalized)
    if negated:
        operator = _TIME_BASE_OP_MAP.get(negated.group(1).strip())
        return _TIME_NEGATION_MAP.get(operator, operator or "")
    return _TIME_BASE_OP_MAP.get(normalized, "")


def _time_bounds_from_operator(operator: str, value: float) -> dict[str, float | None] | None:
    if operator in {"<", "<="}:
        return {"min": 0, "max": value}
    if operator in {">", ">="}:
        return {"min": value, "max": None}
    if operator == "~":
        return {"min": round(value * 0.9, 1), "max": round(value * 1.1, 1)}
    return None


def _time_field_from_span(
    transcript: str,
    span: tuple[int, int],
    recipe_config: dict[str, Any],
    schema: dict[str, Any],
) -> str:
    text = transcript.lower()
    start, end = span
    if re.search(r"\bto\s+prepare\b", text[start:min(len(text), end + 72)]):
        return link_base_field("prep_time", recipe_config=recipe_config, schema=schema) or "prep_time"
    candidates: list[tuple[int, int, str]] = []
    field_patterns = (
        ("prep_time", r"\b(?:prep|prepare|prepares|prepared|preparing|preparation)\b"),
        ("cook_time", r"\b(?:cook|cooks|cooked|cooking)\b"),
        ("total_time", r"\b(?:total|overall|ready|take|takes|taking)\b"),
    )
    for field, pattern in field_patterns:
        for match in re.finditer(pattern, text):
            if match.end() < max(0, start - 96) or match.start() > end + 40:
                continue
            if field == "prep_time" and not _prep_term_applies_to_span(text, match, span):
                continue
            distance = min(abs(start - match.end()), abs(match.start() - end))
            before_bonus = 0 if match.start() <= start else 20
            candidates.append((distance + before_bonus, -len(match.group(0)), field))
    if candidates:
        candidates.sort()
        field = candidates[0][2]
        return link_base_field(field, recipe_config=recipe_config, schema=schema) or field
    return _time_field_from_text(_normalize_text(transcript), recipe_config, schema)


def _prep_term_applies_to_span(text: str, match: re.Match, span: tuple[int, int]) -> bool:
    term = match.group(0)
    start, end = span
    if term in {"prep", "preparation"}:
        return True
    local = text[max(0, match.start() - 16):min(len(text), end + 96)]
    if re.search(r"\bprepared\s+time\b", local):
        return True
    if re.search(r"\bto\s+prepare\b", local):
        return True
    if re.search(r"\bcook(?:s|ed|ing)?\b", text[end:min(len(text), end + 120)]):
        return True
    return term in {"prepare", "prepares", "preparing"}


def _dedupe_time_filters(filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen = set()
    for item in filters:
        key = (item.get("field"), item.get("min"), item.get("max"))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


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
    normalized_text = _normalize_text(text)
    text_tokens = _tokenize(normalized_text)
    matches: list[str] = []
    for term in terms:
        normalized_term = _normalize_text(term)
        if not normalized_term:
            continue
        if re.search(rf"\b{re.escape(normalized_term)}\b", normalized_text):
            matches.append(normalized_term)
            continue
        if _fuzzy_phrase_in_tokens(text_tokens, _tokenize(normalized_term)):
            matches.append(normalized_term)
    return matches


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _fuzzy_phrase_in_tokens(text_tokens: list[str], term_tokens: list[str]) -> bool:
    if not text_tokens or not term_tokens or len(term_tokens) > len(text_tokens):
        return False
    if len(term_tokens) == 1:
        return False
    if all(len(token) < 5 for token in term_tokens):
        return False
    for start in range(0, len(text_tokens) - len(term_tokens) + 1):
        window = text_tokens[start:start + len(term_tokens)]
        if all(_fuzzy_token_match(actual, expected) for actual, expected in zip(window, term_tokens)):
            return True
    return False


def _fuzzy_token_match(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    if len(expected) < 5 or len(actual) < 4:
        return False
    threshold = 83 if max(len(actual), len(expected)) >= 6 else 86
    return fuzz.ratio(actual, expected) >= threshold


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


def _unique_preserving(items: list[Any]) -> list[str]:
    result: list[str] = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = _normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())).strip()
