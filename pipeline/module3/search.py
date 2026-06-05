from __future__ import annotations

from typing import Any

from module3.api_verifier import verify_recipes_against_ir
from module3.client import ApiResult, RecipeDBClient
from module3.post_filter import post_filter
from module3.query_planner import ApiQueryPlan, plan_queries
from module3.ranker import rank_and_limit
from module3.result_merger import merge_api_results
from module3.intent_preparer import prepare_intent
from module3.sql_intent_builder import build_query_ir
from module3.sql_query_builder import build_sql_query


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
    from module1_asr.entity_tagger import tag_entities

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

    client = RecipeDBClient(api_key=api_key, base_url=base_url)
    api_results = [client.execute(plan) for plan in selected_plans]
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
