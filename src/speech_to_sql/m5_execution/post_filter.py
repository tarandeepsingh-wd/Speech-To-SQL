from __future__ import annotations

import re
from typing import Any

from speech_to_sql.common.config import load_recipe_db_config
from speech_to_sql.m3_query_ir.intent_preparer import prepare_intent


def post_filter(recipes: list[dict], tag_result: dict) -> list[dict]:
    config = load_recipe_db_config()
    tag_result = prepare_intent(tag_result, recipe_config=config)
    logic = tag_result.get("ingredient_logic") or {}
    include = _normalize_terms(logic.get("include") or tag_result.get("ingredients") or [])
    exclude = _normalize_terms(logic.get("exclude") or [])
    include_operator = logic.get("include_operator") or "AND"
    dietary_styles = tag_result.get("dietary_styles") or []
    nutrition_filters = tag_result.get("nutrition_filters") or []
    utensils = _normalize_terms(tag_result.get("utensils") or [])

    filtered = []
    for recipe in recipes:
        searchable = _recipe_search_text(recipe, config)

        if include:
            matches = [_contains_term(searchable, term) for term in include]
            if include_operator == "OR":
                if not any(matches):
                    continue
            else:
                if not all(matches):
                    continue

        if exclude and any(_contains_term(searchable, term) for term in exclude):
            continue

        if utensils and not _matches_utensils(recipe, utensils, config):
            continue

        if not _matches_diet(recipe, dietary_styles, config):
            continue

        if not _matches_nutrition(recipe, nutrition_filters, config):
            continue

        filtered.append(recipe)
    return filtered


def _recipe_search_text(recipe: dict, config: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in config["search_text_fields"]:
        value = recipe.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(_flatten_list_text(value, config))
    return _normalize_text(" ".join(parts))


def _flatten_list_text(items: list[Any], config: dict[str, Any]) -> list[str]:
    values = []
    for item in items:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, dict):
            for key in config["nested_ingredient_fields"]:
                value = item.get(key)
                if isinstance(value, str):
                    values.append(value)
    return values


def _matches_utensils(recipe: dict, utensils: list[str], config: dict[str, Any]) -> bool:
    value = ""
    for field in config["utensil_fields"]:
        value = recipe.get(field)
        if value:
            break
    text = _normalize_text(str(value or ""))
    if not text:
        return True
    return all(_contains_term(text, utensil) for utensil in utensils)


def _matches_diet(recipe: dict, styles: list[str], config: dict[str, Any]) -> bool:
    diet_aliases = config["diet_aliases"]
    for style in styles:
        api_style = diet_aliases.get(style, style if style in diet_aliases.values() else None)
        if not api_style:
            continue
        if api_style in recipe and not _truthy_recipe_flag(recipe.get(api_style), config):
            return False
    return True


def _matches_nutrition(
    recipe: dict,
    nutrition_filters: list[dict],
    config: dict[str, Any],
) -> bool:
    for nf in nutrition_filters:
        nutrient = nf.get("nutrient")
        value = _field_float(recipe, nutrient, config)
        if value is None:
            continue
        try:
            target = float(nf["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if not _compare(value, nf.get("operator"), target):
            return False
    return True


def _field_float(recipe: dict, nutrient: str | None, config: dict[str, Any]) -> float | None:
    if not nutrient:
        return None
    aliases = config["response_field_aliases"]
    for field in aliases.get(nutrient, [nutrient]):
        if field in recipe:
            return _to_float(recipe.get(field))
    return None


def _compare(value: float, op: str | None, target: float) -> bool:
    if op == "<":
        return value < target
    if op == "<=":
        return value <= target
    if op == ">":
        return value > target
    if op == ">=":
        return value >= target
    if op == "~":
        return target * 0.9 <= value <= target * 1.1
    if op == "=":
        return value == target
    return True


def _truthy_recipe_flag(value: Any, config: dict[str, Any]) -> bool:
    if isinstance(value, bool):
        return value
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return str(value).strip().lower() in set(config["truthy_values"])


def _normalize_terms(items: list[Any]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        text = _normalize_text(str(item))
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())).strip()


def _contains_term(text: str, term: str) -> bool:
    if not term:
        return True
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))
