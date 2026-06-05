from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from module3.client import ApiResult
from module3.config import load_recipe_db_config


def recipe_key(recipe: dict) -> str | None:
    config = load_recipe_db_config()
    for field in config["identity_fields"]:
        value = recipe.get(field)
        if value not in (None, ""):
            return f"{field}:{value}"
    fallback_fields = config["dedupe_fallback_fields"]
    title = _first_value(recipe, fallback_fields["title"])
    region = _first_value(recipe, fallback_fields["region"])
    if title:
        return f"title:{str(title).lower()}|{str(region or '').lower()}"
    return None


def dedupe_recipes(recipes: Iterable[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    no_key = []
    for recipe in recipes:
        key = recipe_key(recipe)
        if not key:
            no_key.append(recipe)
            continue
        if key in merged:
            merged[key].update(_without_empty_values(recipe))
        else:
            merged[key] = dict(recipe)
    return list(merged.values()) + no_key


def merge_api_results(results: list[ApiResult]) -> list[dict]:
    """
    AND-merge API result sets by recipe id when possible.

    If only one endpoint returns usable keys, it becomes the base set and later
    client-side filtering still applies the full Module 1 intent. If no endpoint
    has stable keys, we return a deduped union.
    """
    successful = [result for result in results if result.ok and result.recipes]
    if not successful:
        return []
    if len(successful) == 1:
        return dedupe_recipes(successful[0].recipes)

    keyed_sets: list[dict[str, dict]] = []
    unkeyed_recipes: list[dict] = []
    for result in successful:
        keyed: dict[str, dict] = {}
        for recipe in result.recipes:
            key = recipe_key(recipe)
            if key:
                if key in keyed:
                    keyed[key].update(_without_empty_values(recipe))
                else:
                    keyed[key] = dict(recipe)
            else:
                unkeyed_recipes.append(recipe)
        if keyed:
            keyed_sets.append(keyed)

    if not keyed_sets:
        return dedupe_recipes(unkeyed_recipes)

    common_keys = set(keyed_sets[0])
    for keyed in keyed_sets[1:]:
        common_keys &= set(keyed)

    if not common_keys:
        primary = successful[0].recipes
        return dedupe_recipes(primary)

    by_key = defaultdict(dict)
    for keyed in keyed_sets:
        for key in common_keys:
            by_key[key].update(_without_empty_values(keyed[key]))
    return list(by_key.values())


def _without_empty_values(recipe: dict) -> dict:
    return {
        key: value
        for key, value in recipe.items()
        if value not in (None, "")
    }


def _first_value(recipe: dict, fields: list[str]):
    for field in fields:
        value = recipe.get(field)
        if value not in (None, ""):
            return value
    return None
