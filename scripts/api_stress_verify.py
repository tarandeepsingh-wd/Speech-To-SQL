from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from scripts.stress_verify_queries import generate_cases
from speech_to_sql.m2_nlu.entity_tagger import tag_entities
from speech_to_sql.m3_query_ir.intent_preparer import prepare_intent
from speech_to_sql.m3_query_ir.sql_intent_builder import build_query_ir
from speech_to_sql.m4_sql.sql_query_builder import build_sql_query
from speech_to_sql.m5_execution.api_verifier import verify_recipes_against_ir
from speech_to_sql.m5_execution.client import ApiResult, RecipeDBClient
from speech_to_sql.m5_execution.post_filter import post_filter
from speech_to_sql.m5_execution.query_planner import ApiQueryPlan, plan_queries
from speech_to_sql.m5_execution.ranker import rank_and_limit
from speech_to_sql.m5_execution.result_merger import merge_api_results


def execute_with_retry(
    client: RecipeDBClient,
    plan: ApiQueryPlan,
    *,
    retries: int,
    cooldown_seconds: float,
) -> ApiResult:
    result = client.execute(plan)
    attempts = 0
    while result.status_code == 429 and attempts < retries:
        attempts += 1
        time.sleep(cooldown_seconds)
        result = client.execute(plan)
    return result


def run_case(
    case: Any,
    *,
    client: RecipeDBClient,
    max_api_calls: int,
    fetch_limit: int,
    final_limit: int,
    retries: int,
    cooldown_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "case_id": case.case_id,
        "category": case.category,
        "query": case.query,
        "ok": False,
        "errors": [],
    }
    errors: list[str] = result["errors"]

    try:
        tags = tag_entities(case.query)
        tags["transcript"] = case.query
        prepared = prepare_intent(tags, transcript=case.query)
        plans = plan_queries(prepared, transcript=case.query, fetch_limit=fetch_limit)
        selected = _select_plans(plans, max_api_calls)
        api_results = [
            execute_with_retry(
                client,
                plan,
                retries=retries,
                cooldown_seconds=cooldown_seconds,
            )
            for plan in selected
        ]
        query_ir = build_query_ir(prepared, transcript=case.query)
        sql = build_sql_query(prepared, transcript=case.query, query_ir=query_ir)
        merged = merge_api_results(api_results)
        filtered = post_filter(merged, prepared)
        ranked = rank_and_limit(filtered, prepared, transcript=case.query, default_limit=final_limit)
        verification = verify_recipes_against_ir(ranked, query_ir)

        result.update({
            "intent": {
                "ingredients": prepared.get("ingredient_logic", {}).get("include", []),
                "excluded_ingredients": prepared.get("ingredient_logic", {}).get("exclude", []),
                "dietary_styles": prepared.get("dietary_styles", []),
                "category_filters": prepared.get("category_filters", []),
                "nutrition_filters": prepared.get("nutrition_filters", []),
                "time_filters": prepared.get("time_filters", []),
                "utensils": prepared.get("utensils", []),
                "actions": prepared.get("actions", []),
            },
            "all_plans": [plan.as_dict() for plan in plans],
            "selected_plans": [plan.as_dict() for plan in selected],
            "api_results": [_api_result_summary(item) for item in api_results],
            "sql": {
                "query": sql.get("query"),
                "params": sql.get("params", []),
                "validation": sql.get("validation", {}),
            },
            "merged_count": len(merged),
            "filtered_count": len(filtered),
            "result_count": len(ranked),
            "first_recipe": _recipe_summary(ranked[0]) if ranked else None,
            "api_verification": verification.as_dict(),
        })

        if not selected:
            errors.append("no_api_plan_selected")
        failed_calls = [item for item in api_results if not item.ok]
        if failed_calls:
            errors.append("api_call_failed")
        if any(item.status_code == 429 for item in api_results):
            errors.append("rate_limited")
        validation = sql.get("validation", {})
        if not validation.get("ok"):
            errors.append(f"sql_validation_failed={validation.get('errors')}")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        result["ok"] = not errors
    return result


def _select_plans(plans: list[ApiQueryPlan], max_api_calls: int) -> list[ApiQueryPlan]:
    if max_api_calls <= 0:
        return []
    primary = [plan for plan in plans if plan.role == "primary"]
    secondary = [plan for plan in plans if plan.role != "primary"]
    return (primary[:1] + secondary)[:max_api_calls]


def _api_result_summary(result: ApiResult) -> dict[str, Any]:
    return {
        "label": result.plan.label,
        "method": result.plan.method,
        "path": result.plan.path,
        "params": dict(result.plan.params),
        "ok": result.ok,
        "status_code": result.status_code,
        "recipe_count": len(result.recipes),
        "from_cache": result.from_cache,
        "error": result.error,
    }


def _recipe_summary(recipe: dict[str, Any]) -> dict[str, Any]:
    return {
        "Recipe_id": recipe.get("Recipe_id") or recipe.get("recipe_id"),
        "Recipe_title": recipe.get("Recipe_title") or recipe.get("recipe_title") or recipe.get("title"),
        "Region": recipe.get("Region") or recipe.get("region"),
        "Calories": recipe.get("Calories") or recipe.get("calories"),
        "keys": sorted(recipe.keys())[:25],
    }


def write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live RecipeDB API stress verification for generated NL queries.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-api-calls", type=int, default=2)
    parser.add_argument("--fetch-limit", type=int, default=10)
    parser.add_argument("--final-limit", type=int, default=5)
    parser.add_argument("--min-interval", type=float, default=3.5)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--cooldown-seconds", type=float, default=90)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "docs" / "verification" / "api_stress_100_latest.json")
    parser.add_argument("--show-failures", type=int, default=10)
    args = parser.parse_args()

    os.environ["RDB2_MIN_REQUEST_INTERVAL_SECONDS"] = str(args.min_interval)
    client = RecipeDBClient(ttl_seconds=0)
    cases = generate_cases(args.count)
    started = time.perf_counter()
    results = []
    partial_report = args.report.with_suffix(".partial.json")
    for index, case in enumerate(cases, start=1):
        item = run_case(
            case,
            client=client,
            max_api_calls=args.max_api_calls,
            fetch_limit=args.fetch_limit,
            final_limit=args.final_limit,
            retries=args.retries,
            cooldown_seconds=args.cooldown_seconds,
        )
        results.append(item)
        call_statuses = ",".join(
            str(api_result.get("status_code"))
            for api_result in item.get("api_results", [])
        )
        cache_hits_for_case = sum(
            1
            for api_result in item.get("api_results", [])
            if api_result.get("from_cache")
        )
        print(
            f"[{index}/{len(cases)}] {'PASS' if item['ok'] else 'FAIL'} "
            f"#{item['case_id']} {item['category']} "
            f"calls={len(item.get('api_results', []))} "
            f"status={call_statuses or 'none'} "
            f"cache={cache_hits_for_case} rows={item.get('result_count', 0)} "
            f"query={item['query']}",
            flush=True,
        )
        write_report(
            partial_report,
            {
                "partial": True,
                "completed": index,
                "case_count": len(cases),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "results": results,
            },
        )
    failures = [item for item in results if not item["ok"]]
    api_call_total = sum(len(item.get("api_results", [])) for item in results)
    api_call_success = sum(
        1
        for item in results
        for api_result in item.get("api_results", [])
        if api_result.get("ok")
    )
    status_codes = Counter(
        str(api_result.get("status_code"))
        for item in results
        for api_result in item.get("api_results", [])
    )
    cache_hits = sum(
        1
        for item in results
        for api_result in item.get("api_results", [])
        if api_result.get("from_cache")
    )
    by_category = Counter(item["category"] for item in results)
    failed_by_category = Counter(item["category"] for item in failures)
    result_positive = sum(1 for item in results if item.get("result_count", 0) > 0)
    verified = sum(1 for item in results if (item.get("api_verification") or {}).get("verified"))
    payload = {
        "ok": not failures,
        "case_count": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "settings": {
            "max_api_calls": args.max_api_calls,
            "fetch_limit": args.fetch_limit,
            "final_limit": args.final_limit,
            "min_interval": args.min_interval,
            "retries": args.retries,
            "cooldown_seconds": args.cooldown_seconds,
        },
        "api_call_total": api_call_total,
        "api_call_success": api_call_success,
        "cache_hits": cache_hits,
        "status_codes": dict(sorted(status_codes.items())),
        "result_positive": result_positive,
        "fully_verified": verified,
        "by_category": dict(sorted(by_category.items())),
        "failed_by_category": dict(sorted(failed_by_category.items())),
        "failures": failures,
        "results": results,
    }
    write_report(args.report, payload)

    print(f"API stress cases: {payload['case_count']}")
    print(f"Passed: {payload['passed']}")
    print(f"Failed: {payload['failed']}")
    print(f"API calls: {api_call_success}/{api_call_total} succeeded")
    print(f"Cache hits: {cache_hits}")
    print(f"Status codes: {payload['status_codes']}")
    print(f"Queries with returned final recipes: {result_positive}/{len(results)}")
    print(f"Fully API-verified queries: {verified}/{len(results)}")
    print(f"Elapsed: {payload['elapsed_seconds']}s")
    print(f"Report: {args.report}")
    print("By category:")
    for category, total in sorted(by_category.items()):
        failed = failed_by_category.get(category, 0)
        print(f"  {category}: {total - failed}/{total} passed")
    if failures:
        print("Failures:")
        for failure in failures[: args.show_failures]:
            print(f"  #{failure['case_id']} [{failure['category']}] {failure['query']}")
            for error in failure["errors"]:
                print(f"    - {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
