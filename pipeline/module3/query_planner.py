from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from module3.config import (
    endpoint_method,
    endpoint_params,
    endpoint_path,
    endpoint_priority,
    endpoint_reason,
    load_recipe_db_config,
)
from module3.intent_preparer import prepare_intent
from module3.time_resolver import resolve_time_filter


@dataclass(frozen=True)
class ApiQueryPlan:
    label: str
    method: str
    path: str
    params: dict[str, Any] = field(default_factory=dict)
    priority: int = 100
    role: str = "secondary"
    reason: str = ""
    post_filter_safe: bool = True

    def url(self, base_url: str) -> str:
        return f"{base_url.rstrip('/')}/{self.path.lstrip('/')}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "method": self.method,
            "path": self.path,
            "params": dict(self.params),
            "priority": self.priority,
            "role": self.role,
            "reason": self.reason,
            "post_filter_safe": self.post_filter_safe,
        }


def plan_queries(
    tag_result: dict,
    *,
    transcript: str | None = None,
    page: int | None = None,
    fetch_limit: int | None = None,
    include_general_fallback: bool = True,
) -> list[ApiQueryPlan]:
    """
    Convert Module 1 tag_entities output into RecipeDB API calls.

    Endpoint paths, parameter names, aliases, and response/schema facts come
    from module3/config/recipe_db.json.
    """
    config = load_recipe_db_config()
    tag_result = prepare_intent(tag_result, transcript=transcript, recipe_config=config)
    pagination = config["pagination"]
    page = page if page is not None else int(pagination["default_page"])
    fetch_limit = fetch_limit if fetch_limit is not None else int(pagination["default_fetch_limit"])
    transcript = transcript or tag_result.get("transcript", "") or tag_result.get("query", "")
    text = transcript.lower()
    plans: list[ApiQueryPlan] = []

    ingredient_plan = _ingredient_category_title_plan(tag_result, text, page, fetch_limit, config)
    if ingredient_plan:
        plans.append(ingredient_plan)

    region = _extract_region(tag_result, text, config)
    dietary_styles = _dietary_styles(tag_result)
    primary_diet = _first_supported_diet(dietary_styles, config)

    if region and primary_diet:
        plans.append(_make_plan(
            "region_diet",
            label=f"region_diet:{region}:{primary_diet}",
            params={
                "region": region,
                "diet": _api_diet(primary_diet, config),
                "page": page,
                "limit": fetch_limit,
            },
            role="primary" if not ingredient_plan else "secondary",
            config=config,
        ))
    elif region:
        plans.append(_make_plan(
            "cuisine",
            label=f"cuisine:{region}",
            params=_cuisine_param_values(region, tag_result, text, page, fetch_limit, config),
            role="primary" if not ingredient_plan else "secondary",
            config=config,
            path_values={"region": region},
        ))

    if primary_diet and not (region and primary_diet):
        plans.append(_make_plan(
            "diet",
            label=f"diet:{primary_diet}",
            params={"diet": _api_diet(primary_diet, config), "page": page, "limit": fetch_limit},
            role="primary" if not ingredient_plan and not region else "secondary",
            config=config,
        ))

    plans.extend(_nutrition_plans(tag_result, page, fetch_limit, config))
    plans.extend(_time_plans(
        tag_result,
        text,
        page,
        fetch_limit,
        config,
        skip_if_cuisine_has_time=bool(region),
    ))
    plans.extend(_utensil_plans(tag_result, page, fetch_limit, config))

    if not plans and include_general_fallback:
        title = _title_query(tag_result, text)
        if title:
            plans.append(_make_plan(
                "title",
                label=f"title:{title}",
                params={"title": title},
                role="primary",
                config=config,
            ))
        else:
            plans.append(_make_plan(
                "general",
                label="general",
                params={
                    "page": page,
                    "limit": int(pagination["general_fallback_limit"]),
                },
                role="primary",
                config=config,
            ))

    return _normalize_roles(sorted(plans, key=lambda plan: plan.priority))


def _make_plan(
    endpoint_name: str,
    *,
    label: str,
    params: dict[str, Any],
    role: str,
    config: dict[str, Any],
    path_values: dict[str, Any] | None = None,
) -> ApiQueryPlan:
    return ApiQueryPlan(
        label=label,
        method=endpoint_method(endpoint_name, config),
        path=endpoint_path(endpoint_name, config, **(path_values or {})),
        params=endpoint_params(endpoint_name, params, config),
        priority=endpoint_priority(endpoint_name, config),
        role=role,
        reason=endpoint_reason(endpoint_name, config),
    )


def _normalize_roles(plans: list[ApiQueryPlan]) -> list[ApiQueryPlan]:
    if not plans:
        return plans
    normalized = []
    primary_assigned = False
    for plan in plans:
        role = plan.role
        if not primary_assigned:
            role = "primary"
            primary_assigned = True
        elif role == "primary":
            role = "secondary"
        if role == plan.role:
            normalized.append(plan)
        else:
            normalized.append(ApiQueryPlan(
                label=plan.label,
                method=plan.method,
                path=plan.path,
                params=plan.params,
                priority=plan.priority,
                role=role,
                reason=plan.reason,
                post_filter_safe=plan.post_filter_safe,
            ))
    return normalized


def _ingredient_category_title_plan(
    tag_result: dict,
    text: str,
    page: int,
    fetch_limit: int,
    config: dict[str, Any],
) -> ApiQueryPlan | None:
    logic = tag_result.get("ingredient_logic") or {}
    include = _unique_strings(logic.get("include") or tag_result.get("ingredients") or [])
    exclude = _unique_strings(logic.get("exclude") or [])
    include_categories, exclude_categories = _category_filters(tag_result, text, config)
    title = _title_query(tag_result, text)

    if not (include or exclude or include_categories or exclude_categories or title):
        return None

    params: dict[str, Any] = {"page": page, "limit": fetch_limit}
    if include:
        params["include_ingredients"] = _csv(include)
    if exclude:
        params["exclude_ingredients"] = _csv(exclude)
    if include_categories:
        params["include_categories"] = _csv(include_categories)
    if exclude_categories:
        params["exclude_categories"] = _csv(exclude_categories)
    if title:
        params["title"] = title

    return _make_plan(
        "ingredient_category_title",
        label="ingredient_category_title",
        params=params,
        role="primary",
        config=config,
    )


def _nutrition_plans(
    tag_result: dict,
    page: int,
    fetch_limit: int,
    config: dict[str, Any],
) -> list[ApiQueryPlan]:
    plans: list[ApiQueryPlan] = []
    endpoint_names = set(config["endpoints"])
    for nf in tag_result.get("nutrition_filters", []) or []:
        nutrient = nf.get("nutrient")
        if nutrient not in endpoint_names:
            continue
        bounds = _numeric_bounds(nf)
        if not bounds:
            continue
        min_value, max_value = bounds
        plans.append(_make_plan(
            nutrient,
            label=_filter_label(nutrient, nf),
            params={
                "min": min_value,
                "max": max_value,
                "page": page,
                "limit": fetch_limit,
            },
            role="secondary",
            config=config,
        ))
    return plans


def _time_plans(
    tag_result: dict,
    text: str,
    page: int,
    fetch_limit: int,
    config: dict[str, Any],
    *,
    skip_if_cuisine_has_time: bool,
) -> list[ApiQueryPlan]:
    if skip_if_cuisine_has_time:
        return []
    time_filter = resolve_time_filter(tag_result, text, config)
    if not time_filter:
        return []
    field_name, min_value, max_value = time_filter
    return [_make_plan(
        "time_range",
        label=f"time:{field_name}",
        params={
            "field": field_name,
            "min": min_value,
            "max": max_value,
            "page": page,
            "limit": fetch_limit,
        },
        role="secondary",
        config=config,
    )]


def _utensil_plans(
    tag_result: dict,
    page: int,
    fetch_limit: int,
    config: dict[str, Any],
) -> list[ApiQueryPlan]:
    utensils = _unique_strings(tag_result.get("utensils") or [])
    if not utensils:
        return []
    return [_make_plan(
        "utensils",
        label="utensils",
        params={"utensils": _csv(utensils), "page": page, "limit": fetch_limit},
        role="secondary",
        config=config,
    )]


def _cuisine_param_values(
    region: str,
    tag_result: dict,
    text: str,
    page: int,
    fetch_limit: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = {"page": page, "limit": fetch_limit}
    cuisine = _cuisine_alias(region, config)
    if cuisine.get("continent"):
        params["continent"] = cuisine["continent"]
    if cuisine.get("sub_region"):
        params["sub_region"] = cuisine["sub_region"]

    time_filter = resolve_time_filter(tag_result, text, config)
    if time_filter:
        field_name, min_value, max_value = time_filter
        params["field"] = field_name
        params["min"] = min_value
        params["max"] = max_value
    return params


def _numeric_bounds(nutrition_filter: dict) -> tuple[float | None, float | None] | None:
    try:
        value = float(nutrition_filter["value"])
    except (KeyError, TypeError, ValueError):
        return None
    op = nutrition_filter.get("operator")
    if op in {"<", "<="}:
        return 0, value
    if op in {">", ">="}:
        return value, None
    if op == "~":
        return round(value * 0.9, 1), round(value * 1.1, 1)
    if op == "=":
        return value, value
    return None


def _category_filters(
    tag_result: dict,
    text: str,
    config: dict[str, Any],
) -> tuple[list[str], list[str]]:
    cuisine_types = set(config["category_types"]["cuisine"])
    include: list[str] = []
    exclude: list[str] = []
    for item in tag_result.get("categories", []) or []:
        category = item.get("category")
        term = item.get("term")
        if not category:
            continue
        if str(category).lower() in cuisine_types:
            continue
        if term and _is_excluded_near_term(text, str(term)):
            exclude.append(str(category))
        else:
            include.append(str(category))
    return _unique_strings(include), _unique_strings(exclude)


def _is_excluded_near_term(text: str, term: str) -> bool:
    if not term:
        return False
    match = re.search(rf"\b{re.escape(term.lower())}\b", text)
    if not match:
        return False
    window = text[max(0, match.start() - 32):match.start()]
    return bool(re.search(r"\b(?:without|exclude|excluding|avoid|no|free of)\b", window))


def _extract_region(tag_result: dict, text: str, config: dict[str, Any]) -> str | None:
    cuisine_types = set(config["category_types"]["cuisine"])
    for item in tag_result.get("categories", []) or []:
        category = str(item.get("category", "")).lower()
        if category in cuisine_types:
            return str(item.get("term") or item.get("region") or "").strip() or None

    aliases = config["cuisine_aliases"]
    for term in sorted(aliases, key=len, reverse=True):
        if re.search(rf"\b{re.escape(term)}\b", text):
            return aliases[term].get("region") or term.title()
    return None


def _dietary_styles(tag_result: dict) -> list[str]:
    return _unique_strings(tag_result.get("dietary_styles") or [])


def _first_supported_diet(styles: list[str], config: dict[str, Any]) -> str | None:
    aliases = config["diet_aliases"]
    for style in styles:
        if style in aliases or style in aliases.values():
            return style
    return None


def _api_diet(style: str, config: dict[str, Any]) -> str:
    return config["diet_aliases"].get(style, style)


def _cuisine_alias(region: str, config: dict[str, Any]) -> dict[str, Any]:
    return config["cuisine_aliases"].get(region.lower(), {})


def _title_query(tag_result: dict, text: str) -> str | None:
    for key in ("title", "recipe_title"):
        value = tag_result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    quoted = re.search(r"['\"]([^'\"]{3,80})['\"]", text)
    if quoted:
        return quoted.group(1).strip()
    return None


def _filter_label(name: str, nutrition_filter: dict) -> str:
    return f"{name}:{nutrition_filter.get('operator', '')}{nutrition_filter.get('value', '')}"


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


def _csv(items: list[str]) -> str:
    return ",".join(items)


def _contains_config_term(text: str, terms: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)
