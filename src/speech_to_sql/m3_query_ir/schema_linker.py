from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

from speech_to_sql.common.config import load_recipe_db_config, load_sql_schema_config


@dataclass(frozen=True)
class SchemaLink:
    source: str
    original: str
    resolved: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "original": self.original,
            "resolved": self.resolved,
            "reason": self.reason,
        }


def link_tag_result(
    tag_result: dict[str, Any],
    *,
    transcript: str | None = None,
    recipe_config: dict[str, Any] | None = None,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if tag_result.get("_schema_linked"):
        return tag_result

    recipe_config = recipe_config or load_recipe_db_config()
    schema = schema or load_sql_schema_config()
    linked = copy.deepcopy(tag_result)
    links: list[SchemaLink] = []
    unresolved: list[dict[str, str]] = []

    linked["nutrition_filters"] = _link_nutrition_filters(
        linked.get("nutrition_filters") or [],
        recipe_config,
        schema,
        links,
        unresolved,
    )
    linked["dietary_styles"] = _link_dietary_styles(
        linked.get("dietary_styles") or [],
        recipe_config,
        links,
        unresolved,
    )
    linked["time_filters"] = _link_time_filters(
        linked.get("time_filters") or [],
        recipe_config,
        schema,
        links,
        unresolved,
    )
    linked["category_filters"] = _link_categories(
        linked,
        linked.get("categories") or [],
        recipe_config,
        schema,
        links,
        unresolved,
    )
    linked["ingredient_logic"] = _normalize_ingredient_logic(linked)

    if transcript and not linked.get("transcript"):
        linked["transcript"] = transcript

    linked["_schema_linked"] = True
    linked["_schema_links"] = [item.as_dict() for item in links]
    linked["_unresolved_schema_terms"] = unresolved
    return linked


def link_nutrition_field(
    term: Any,
    *,
    recipe_config: dict[str, Any] | None = None,
    schema: dict[str, Any] | None = None,
) -> str | None:
    recipe_config = recipe_config or load_recipe_db_config()
    schema = schema or load_sql_schema_config()
    return _resolve_alias(
        term,
        _nutrition_alias_index(recipe_config, schema),
    )


def link_dietary_style(
    term: Any,
    *,
    recipe_config: dict[str, Any] | None = None,
) -> str | None:
    recipe_config = recipe_config or load_recipe_db_config()
    return _resolve_alias(term, _diet_alias_index(recipe_config))


def link_base_field(
    term: Any,
    *,
    recipe_config: dict[str, Any] | None = None,
    schema: dict[str, Any] | None = None,
) -> str | None:
    recipe_config = recipe_config or load_recipe_db_config()
    schema = schema or load_sql_schema_config()
    return _resolve_alias(term, _base_alias_index(recipe_config, schema))


def _link_nutrition_filters(
    filters: list[dict[str, Any]],
    recipe_config: dict[str, Any],
    schema: dict[str, Any],
    links: list[SchemaLink],
    unresolved: list[dict[str, str]],
) -> list[dict[str, Any]]:
    alias_index = _nutrition_alias_index(recipe_config, schema)
    linked_filters = []
    for item in filters:
        if not isinstance(item, dict):
            continue
        linked = dict(item)
        original = linked.get("nutrient")
        resolved = _resolve_alias(original, alias_index)
        if resolved:
            linked["nutrient"] = resolved
            _record_link(links, "nutrition", original, resolved, "semantic_aliases.nutrition")
        elif original:
            _record_unresolved(unresolved, "nutrition", original)
        linked_filters.append(linked)
    return linked_filters


def _link_dietary_styles(
    styles: list[Any],
    recipe_config: dict[str, Any],
    links: list[SchemaLink],
    unresolved: list[dict[str, str]],
) -> list[str]:
    alias_index = _diet_alias_index(recipe_config)
    linked_styles = []
    seen = set()
    for style in styles:
        resolved = _resolve_alias(style, alias_index)
        if not resolved:
            _record_unresolved(unresolved, "dietary", style)
            continue
        key = resolved.lower()
        if key in seen:
            continue
        seen.add(key)
        linked_styles.append(resolved)
        _record_link(links, "dietary", style, resolved, "recipe_db.diet_aliases")
    return linked_styles


def _link_time_filters(
    filters: list[dict[str, Any]],
    recipe_config: dict[str, Any],
    schema: dict[str, Any],
    links: list[SchemaLink],
    unresolved: list[dict[str, str]],
) -> list[dict[str, Any]]:
    alias_index = _base_alias_index(recipe_config, schema)
    linked_filters = []
    for item in filters:
        if not isinstance(item, dict):
            continue
        linked = dict(item)
        original = linked.get("field")
        resolved = _resolve_alias(original, alias_index)
        if resolved:
            linked["field"] = resolved
            _record_link(links, "base", original, resolved, "semantic_aliases.base")
        elif original:
            _record_unresolved(unresolved, "base", original)
        linked_filters.append(linked)
    return linked_filters


def _link_categories(
    tag_result: dict[str, Any],
    categories: list[dict[str, Any]],
    recipe_config: dict[str, Any],
    schema: dict[str, Any],
    links: list[SchemaLink],
    unresolved: list[dict[str, str]],
) -> list[dict[str, Any]]:
    cuisine_alias_index = _cuisine_alias_index(recipe_config)
    flavor_alias_index = _ingredient_flavor_alias_index(recipe_config)
    base_columns = schema["base_columns"]
    flavor_columns = schema.get("relations", {}).get("ingredient_flavor", {}).get("columns", {})
    filters: list[dict[str, Any]] = []
    for item in categories:
        if not isinstance(item, dict):
            continue
        original = item.get("term") or item.get("value") or item.get("text")
        if _resolve_alias(original, _nutrition_alias_index(recipe_config, schema)):
            continue
        if _term_is_part_of_ingredient(original, tag_result):
            continue

        flavor_filter = _resolve_cuisine_alias(original, flavor_alias_index)
        if flavor_filter:
            field = str(flavor_filter.get("field") or "")
            value = flavor_filter.get("value")
            if field not in flavor_columns or value is None:
                _record_unresolved(unresolved, "ingredient_flavor", original)
                continue
            filters.append({
                "source": "ingredient_flavor",
                "field": field,
                "operator": "=",
                "value": value,
                "term": str(original),
            })
            _record_link(links, "ingredient_flavor", original, field, "recipe_db.ingredient_flavor_aliases")
            continue

        resolved = _resolve_cuisine_alias(original, cuisine_alias_index)
        if not resolved:
            continue
        for field, value in resolved.items():
            if field not in base_columns:
                _record_unresolved(unresolved, "category", field)
                continue
            filters.append({
                "field": field,
                "operator": "=",
                "value": value,
                "term": str(original),
            })
            _record_link(links, "category", original, field, "recipe_db.cuisine_aliases")
    return filters


def _normalize_ingredient_logic(tag_result: dict[str, Any]) -> dict[str, Any]:
    logic = dict(tag_result.get("ingredient_logic") or {})
    consumed = {
        _normalize_key(term)
        for term in tag_result.get("_consumed_nutrition_terms") or []
    }
    consumed.update(
        _normalize_key(item.get("term") or item.get("value") or item.get("text"))
        for item in tag_result.get("category_filters") or []
        if isinstance(item, dict)
    )
    include = logic.get("include") or tag_result.get("ingredients") or []
    exclude = logic.get("exclude") or []
    logic["include"] = [
        item for item in _unique_strings(include)
        if _normalize_key(item) not in consumed
    ]
    logic["exclude"] = [
        item for item in _unique_strings(exclude)
        if _normalize_key(item) not in consumed
    ]
    logic["include_operator"] = str(logic.get("include_operator") or "AND").upper()
    return logic


def _nutrition_alias_index(
    recipe_config: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, str]:
    nutrition_columns = schema["relations"]["nutrition"]["columns"]
    index = _normalized_index(nutrition_columns.keys())
    index.update(_normalized_mapping(schema.get("semantic_aliases", {}).get("nutrition", {})))

    for logical_name, response_fields in recipe_config["response_field_aliases"].items():
        if logical_name not in nutrition_columns:
            continue
        index.update(_normalized_index(response_fields, logical_name))
    return index


def _diet_alias_index(recipe_config: dict[str, Any]) -> dict[str, str]:
    aliases = recipe_config["diet_aliases"]
    index = _normalized_mapping(aliases)
    for canonical in aliases.values():
        index[_normalize_key(canonical)] = canonical
    return index


def _cuisine_alias_index(recipe_config: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        _normalize_key(source): dict(target)
        for source, target in recipe_config.get("cuisine_aliases", {}).items()
        if source and isinstance(target, dict)
    }


def _ingredient_flavor_alias_index(recipe_config: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        _normalize_key(source): dict(target)
        for source, target in recipe_config.get("ingredient_flavor_aliases", {}).items()
        if source and isinstance(target, dict)
    }


def _term_is_part_of_ingredient(term: Any, tag_result: dict[str, Any]) -> bool:
    normalized = _normalize_key(str(term or ""))
    if not normalized:
        return False
    ingredient_values: list[Any] = []
    ingredient_values.extend(tag_result.get("ingredients") or [])
    logic = tag_result.get("ingredient_logic") or {}
    ingredient_values.extend(logic.get("include") or [])
    ingredient_values.extend(logic.get("exclude") or [])
    for ingredient in ingredient_values:
        ingredient_key = _normalize_key(str(ingredient or ""))
        if ingredient_key != normalized and re.search(rf"\b{re.escape(normalized)}\b", ingredient_key):
            return True
    return False


def _base_alias_index(
    recipe_config: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, str]:
    index = _normalized_index(schema["base_columns"].keys())
    index.update(_normalized_mapping(schema.get("semantic_aliases", {}).get("base", {})))
    index.update(_normalized_mapping(recipe_config["time_fields"]["terms"]))

    for logical_name, response_fields in recipe_config["response_field_aliases"].items():
        if logical_name not in schema["base_columns"]:
            continue
        index.update(_normalized_index(response_fields, logical_name))
    return index


def _resolve_alias(term: Any, alias_index: dict[str, str]) -> str | None:
    if term is None:
        return None
    return alias_index.get(_normalize_key(str(term)))


def _resolve_cuisine_alias(
    term: Any,
    alias_index: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    if term is None:
        return None
    return alias_index.get(_normalize_key(str(term)))


def _record_link(
    links: list[SchemaLink],
    source: str,
    original: Any,
    resolved: str,
    reason: str,
) -> None:
    if original is None:
        return
    original_text = str(original)
    if _normalize_key(original_text) == _normalize_key(resolved):
        return
    links.append(SchemaLink(source, original_text, resolved, reason))


def _record_unresolved(
    unresolved: list[dict[str, str]],
    source: str,
    original: Any,
) -> None:
    if original is None:
        return
    original_text = str(original).strip()
    if not original_text:
        return
    item = {"source": source, "original": original_text}
    if item not in unresolved:
        unresolved.append(item)


def _normalized_mapping(mapping: dict[str, str]) -> dict[str, str]:
    return {
        _normalize_key(source): target
        for source, target in mapping.items()
        if source and target
    }


def _normalized_index(items: Any, target: str | None = None) -> dict[str, str]:
    return {
        _normalize_key(item): target or str(item)
        for item in items
        if str(item).strip()
    }


def _normalize_key(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]+", " ", value.lower())).strip()


def _unique_strings(items: list[Any]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result
