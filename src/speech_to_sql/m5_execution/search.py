from __future__ import annotations

from dataclasses import replace
from typing import Any

from speech_to_sql.m5_execution.api_verifier import verify_recipes_against_ir
from speech_to_sql.m5_execution.client import ApiResult, RecipeDBClient
from speech_to_sql.m5_execution.post_filter import post_filter
from speech_to_sql.m5_execution.query_planner import ApiQueryPlan, plan_queries
from speech_to_sql.m5_execution.ranker import rank_and_limit
from speech_to_sql.m5_execution.result_merger import dedupe_recipes, merge_api_results
from speech_to_sql.common.config import load_recipe_db_config
from speech_to_sql.m3_query_ir.intent_preparer import prepare_intent
from speech_to_sql.m3_query_ir.sql_intent_builder import build_query_ir
from speech_to_sql.m4_sql.sql_query_builder import build_sql_query


def search_text(
    query: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    max_api_calls: int = 3,
    fetch_limit: int = 100,
    final_limit: int | None = None,
    execute: bool = True,
) -> dict[str, Any]:
    from speech_to_sql.m2_nlu.entity_tagger import tag_entities

    tag_result = tag_entities(query)
    tag_result["transcript"] = query
    return search_recipes(
        tag_result,
        transcript=query,
        api_key=api_key,
        base_url=base_url,
        max_api_calls=max_api_calls,
        fetch_limit=fetch_limit,
        final_limit=final_limit,
        execute=execute,
    )


def search_recipes(
    tag_result: dict,
    *,
    transcript: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_api_calls: int = 3,
    fetch_limit: int = 100,
    final_limit: int | None = None,
    execute: bool = True,
) -> dict[str, Any]:
    transcript = transcript or tag_result.get("transcript", "") or tag_result.get("query", "")
    tag_result = prepare_intent(tag_result, transcript=transcript)
    plans = plan_queries(tag_result, transcript=transcript, fetch_limit=fetch_limit)
    selected_plans = _select_plans(plans, max_api_calls)
    query_ir = build_query_ir(tag_result, transcript=transcript)
    sql_debug = build_sql_query(tag_result, transcript=transcript, query_ir=query_ir)

    if not execute:
        return {
            "recipes": [],
            "count": 0,
            "plans": [plan.as_dict() for plan in plans],
            "selected_plans": [plan.as_dict() for plan in selected_plans],
            "sql": sql_debug,
            "intent_enrichments": tag_result.get("_intent_enrichments", []),
            "schema_links": tag_result.get("_schema_links", []),
            "executed": False,
        }

    config = load_recipe_db_config()
    max_pages = int(config.get("request", {}).get("max_pages_per_plan", 1) or 1)
    client = RecipeDBClient(api_key=api_key, base_url=base_url)
    api_results = [_execute_plan_pages(client, plan, max_pages) for plan in selected_plans]
    merged = merge_api_results(api_results)
    filtered = post_filter(merged, tag_result)
    ranked = rank_and_limit(
        filtered,
        tag_result,
        transcript=transcript,
        default_limit=final_limit,
    )
    api_verification = verify_recipes_against_ir(ranked, query_ir)

    return {
        "recipes": ranked,
        "count": len(ranked),
        "plans": [plan.as_dict() for plan in plans],
        "selected_plans": [plan.as_dict() for plan in selected_plans],
        "filters_applied": [plan.label for plan in selected_plans],
        "sql": sql_debug,
        "intent_enrichments": tag_result.get("_intent_enrichments", []),
        "schema_links": tag_result.get("_schema_links", []),
        "api_verification": api_verification.as_dict(),
        "api_results": [_result_summary(result) for result in api_results],
        "executed": True,
    }


def _select_plans(plans: list[ApiQueryPlan], max_api_calls: int) -> list[ApiQueryPlan]:
    if max_api_calls <= 0:
        return []
    primary = [plan for plan in plans if plan.role == "primary"]
    secondary = [plan for plan in plans if plan.role != "primary"]
    selected = primary[:1] + secondary
    return selected[:max_api_calls]


def _result_summary(result: ApiResult) -> dict[str, Any]:
    return {
        "label": result.plan.label,
        "ok": result.ok,
        "status_code": result.status_code,
        "recipe_count": len(result.recipes),
        "error": result.error,
        "from_cache": result.from_cache,
    }


def _execute_plan_pages(client: RecipeDBClient, plan: ApiQueryPlan, max_pages: int) -> ApiResult:
    first = client.execute(plan)
    if max_pages <= 1 or not first.ok or not first.recipes or "page" not in plan.params:
        return first

    total_pages = _total_pages(first.data)
    if total_pages <= 1:
        return first

    recipes = list(first.recipes)
    status_code = first.status_code
    from_cache = first.from_cache
    current_page = _int_value(plan.params.get("page")) or 1
    final_page = min(total_pages, current_page + max_pages - 1)

    for page in range(current_page + 1, final_page + 1):
        paged_plan = replace(plan, params={**plan.params, "page": page})
        page_result = client.execute(paged_plan)
        from_cache = from_cache and page_result.from_cache
        if not page_result.ok:
            return ApiResult(
                plan=plan,
                ok=False,
                status_code=page_result.status_code,
                data=first.data,
                recipes=dedupe_recipes(recipes),
                error=page_result.error,
                from_cache=from_cache,
            )
        status_code = page_result.status_code
        recipes.extend(page_result.recipes)

    return ApiResult(
        plan=plan,
        ok=True,
        status_code=status_code,
        data=first.data,
        recipes=dedupe_recipes(recipes),
        from_cache=from_cache,
    )


def _total_pages(data: Any) -> int:
    if not isinstance(data, dict):
        return 1
    candidates = [
        data.get("totalPages"),
        (data.get("pagination") or {}).get("totalPages")
        if isinstance(data.get("pagination"), dict)
        else None,
    ]
    payload = data.get("payload")
    if isinstance(payload, dict):
        pagination = payload.get("pagination")
        if isinstance(pagination, dict):
            candidates.extend([
                pagination.get("totalPages"),
                pagination.get("total_pages"),
            ])
    for candidate in candidates:
        value = _int_value(candidate)
        if value:
            return value
    return 1


def _int_value(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
