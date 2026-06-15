from __future__ import annotations

import copy
import re
from typing import Any


DEFAULT_NUTRITION_SOURCE = "nutrition"
INGREDIENT_NUTRITION_SOURCE = "ingredient_nutrition"
NUTRITION_SOURCES = {DEFAULT_NUTRITION_SOURCE, INGREDIENT_NUTRITION_SOURCE}

_INGREDIENT_SCOPE_TERMS = [
    "ingredient level",
    "ingredient nutrition",
    "per ingredient",
    "at least one ingredient",
    "one ingredient",
    "any ingredient",
    "an ingredient",
    "each ingredient",
    "ingredient that",
    "ingredients that",
    "ingredient has",
    "ingredients have",
    "ingredient contains",
    "ingredients contain",
    "ingredient with",
    "ingredients with",
    "ingredient contributes",
    "ingredients contribute",
    "contributes",
    "contribute",
]

_RECIPE_SCOPE_TERMS = [
    "total recipe",
    "recipe level",
    "full recipe",
    "whole recipe",
    "recipe nutrition",
    "recipe nutrient",
    "per recipe",
    "recipe has",
    "recipes have",
    "recipe contains",
    "recipes contain",
]

_CLAUSE_BOUNDARY = re.compile(
    r"\b(?:but|where|whereas|while|that\s+uses?|and\s+hav(?:e|es))\b|[;,.]"
)


def assign_nutrition_sources(
    tag_result: dict[str, Any],
    *,
    transcript: str | None = None,
    schema: dict[str, Any],
) -> dict[str, Any]:
    if tag_result.get("_nutrition_scoped"):
        return tag_result

    scoped = copy.deepcopy(tag_result)
    text = transcript or scoped.get("transcript", "") or scoped.get("query", "")
    text_lower = text.lower()
    filters = [
        dict(item)
        for item in scoped.get("nutrition_filters", []) or []
        if isinstance(item, dict)
    ]

    used_ranges: list[tuple[int, int]] = []
    resolutions: list[dict[str, Any]] = []
    for item in filters:
        if item.get("nutrient_group"):
            item["table_source"] = DEFAULT_NUTRITION_SOURCE
            resolutions.append(_resolution(item, None, "nutrition group"))
            continue

        explicit_source = _explicit_source(item, schema)
        if explicit_source:
            item["table_source"] = explicit_source
            resolutions.append(_resolution(item, None, "explicit table_source"))
            continue

        position = _locate_filter_position(text_lower, item, schema, used_ranges)
        source, reason = _infer_source(text_lower, position)
        item["table_source"] = source
        resolutions.append(_resolution(item, position, reason))

    scoped["nutrition_filters"] = filters
    scoped["_nutrition_scoped"] = True
    scoped["_nutrition_scope"] = resolutions
    return scoped


def nutrition_filter_source(
    item: dict[str, Any],
    *,
    schema: dict[str, Any] | None = None,
) -> str:
    if item.get("nutrient_group"):
        return DEFAULT_NUTRITION_SOURCE

    for key in ("table_source", "relation", "relation_source"):
        value = str(item.get(key) or "").strip()
        if _valid_source(value, schema):
            return value

    value = str(item.get("source") or "").strip()
    if _valid_source(value, schema):
        return value

    return DEFAULT_NUTRITION_SOURCE


def _explicit_source(item: dict[str, Any], schema: dict[str, Any]) -> str | None:
    for key in ("table_source", "relation", "relation_source", "source"):
        value = str(item.get(key) or "").strip()
        if _valid_source(value, schema):
            return value
    return None


def _valid_source(value: str, schema: dict[str, Any] | None) -> bool:
    if value not in NUTRITION_SOURCES:
        return False
    if schema is None:
        return True
    return value in (schema.get("relations") or {})


def _locate_filter_position(
    text: str,
    item: dict[str, Any],
    schema: dict[str, Any],
    used_ranges: list[tuple[int, int]],
) -> int | None:
    terms = _candidate_terms(str(item.get("nutrient") or ""), schema)
    matches: list[tuple[int, int]] = []
    for term in terms:
        for match in _term_pattern(term).finditer(text):
            matches.append(match.span())

    matches.sort(key=lambda span: (span[0], -(span[1] - span[0])))
    for start, end in matches:
        if any(_overlaps((start, end), used) for used in used_ranges):
            continue
        used_ranges.append((start, end))
        return start
    if matches:
        return matches[0][0]
    return None


def _candidate_terms(nutrient: str, schema: dict[str, Any]) -> list[str]:
    nutrient = nutrient.strip()
    if not nutrient:
        return []

    terms = {nutrient, nutrient.replace("_", " ")}
    aliases = (schema.get("semantic_aliases") or {}).get("nutrition") or {}
    for alias, target in aliases.items():
        if str(target) == nutrient:
            terms.add(str(alias))

    for source in NUTRITION_SOURCES:
        columns = ((schema.get("relations") or {}).get(source) or {}).get("columns") or {}
        for logical_name, column in columns.items():
            if str(logical_name) == nutrient:
                terms.add(str(logical_name).replace("_", " "))
                terms.add(str(column))

    expanded = set()
    for term in terms:
        clean = _normalize_term(term)
        if not clean:
            continue
        expanded.add(clean)
        if clean.endswith("y"):
            expanded.add(clean[:-1] + "ies")
        elif not clean.endswith("s"):
            expanded.add(clean + "s")
    return sorted(expanded, key=len, reverse=True)


def _term_pattern(term: str) -> re.Pattern[str]:
    tokens = re.findall(r"[a-z0-9]+", term.lower())
    if not tokens:
        return re.compile(r"a^")
    return re.compile(r"\b" + r"[\s,()+-]+".join(re.escape(token) for token in tokens) + r"\b")


def _infer_source(text: str, position: int | None) -> tuple[str, str]:
    if position is None:
        return DEFAULT_NUTRITION_SOURCE, "default recipe nutrition"

    clause = _clause_around(text, position)
    ingredient_distance = _nearest_cue_distance(text, clause, position, _INGREDIENT_SCOPE_TERMS)
    recipe_distance = _nearest_cue_distance(text, clause, position, _RECIPE_SCOPE_TERMS)

    if ingredient_distance is not None and (
        recipe_distance is None or ingredient_distance <= recipe_distance
    ):
        return INGREDIENT_NUTRITION_SOURCE, "ingredient-level scope cue"
    if recipe_distance is not None:
        return DEFAULT_NUTRITION_SOURCE, "recipe-level scope cue"
    return DEFAULT_NUTRITION_SOURCE, "default recipe nutrition"


def _clause_around(text: str, position: int) -> tuple[int, int]:
    start = 0
    end = len(text)
    for match in _CLAUSE_BOUNDARY.finditer(text):
        if match.end() <= position:
            start = match.end()
        elif match.start() >= position:
            end = match.start()
            break
    return start, end


def _nearest_cue_distance(
    text: str,
    clause: tuple[int, int],
    position: int,
    terms: list[str],
) -> int | None:
    start, end = clause
    window = text[start:end]
    best: int | None = None
    for term in terms:
        pattern = _term_pattern(term)
        for match in pattern.finditer(window):
            absolute_start = start + match.start()
            absolute_end = start + match.end()
            if absolute_end < position:
                distance = position - absolute_end
            elif absolute_start > position:
                distance = absolute_start - position
            else:
                distance = 0
            if distance <= 96 and (best is None or distance < best):
                best = distance
    return best


def _resolution(item: dict[str, Any], position: int | None, reason: str) -> dict[str, Any]:
    return {
        "nutrient": item.get("nutrient") or item.get("nutrient_group"),
        "table_source": item.get("table_source"),
        "position": position,
        "reason": reason,
    }


def _normalize_term(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]+", " ", value.lower())).strip()


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]
