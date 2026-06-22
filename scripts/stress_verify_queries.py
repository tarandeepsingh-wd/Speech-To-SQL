from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
TEMP_DB_DIR = PROJECT_ROOT / "temp_recipe_db"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(TEMP_DB_DIR))

from run_query import DB_PATH, count_for_sql, generated_sql, rows_for_sql, verify_generated_result
from seed_temp_db import build_database


LEGACY_IDENTIFIER_RE = re.compile(r"RecipeDB_|micronutrients_v1|matching_values", re.IGNORECASE)
TABLE_RE = re.compile(r"\b(?:FROM|JOIN)\s+`([^`]+)`", re.IGNORECASE)


@dataclass(frozen=True)
class StressCase:
    case_id: int
    category: str
    query: str
    required_tables: tuple[str, ...] = ()
    forbidden_tables: tuple[str, ...] = ()
    min_rows: int = 1


RECIPE_BLUEPRINTS = [
    {
        "name": "chicken_spinach",
        "ingredients": ("chicken", "spinach", "garlic"),
        "region": "Indian",
        "diet": None,
        "instruction": "simmer",
        "process": "saute",
        "utensil": "skillet",
        "flavor": ("garlic", "allium"),
        "recipe_nutrition": ("protein", "above", 30, "g"),
        "ingredient_nutrition": ("protein", "above", 20, "g"),
        "ingredient_nutrition_ingredient": "chicken",
        "absent_ingredient": "salmon",
    },
    {
        "name": "vegan_spinach",
        "ingredients": ("spinach", "lemon juice", "olive oil"),
        "region": "Mediterranean",
        "diet": "vegan",
        "instruction": "mix",
        "process": "mix",
        "utensil": "bowl",
        "flavor": ("spinach", "leafy green"),
        "recipe_nutrition": ("calories", "under", 300, "calories"),
        "ingredient_nutrition": ("vitamin c", "above", 10, "mg"),
        "ingredient_nutrition_ingredient": "lemon juice",
        "absent_ingredient": "chicken",
    },
    {
        "name": "salmon",
        "ingredients": ("salmon", "garlic", "lemon juice"),
        "region": "American",
        "diet": "pescetarian",
        "instruction": "bake",
        "process": "bake",
        "utensil": "oven",
        "flavor": ("salmon", "fish"),
        "recipe_nutrition": ("cholesterol", "above", 50, "mg"),
        "ingredient_nutrition": ("protein", "above", 20, "g"),
        "ingredient_nutrition_ingredient": "salmon",
        "absent_ingredient": "onion",
    },
    {
        "name": "tomato_pasta",
        "ingredients": ("tomato", "pasta", "basil"),
        "region": "Italian",
        "diet": None,
        "instruction": "boil",
        "process": "saute",
        "utensil": "pot",
        "flavor": ("basil", "herb"),
        "recipe_nutrition": ("vitamin c", "above", 10, "mg"),
        "ingredient_nutrition": ("thiamin", "above", 0.2, "mg"),
        "ingredient_nutrition_ingredient": "pasta",
        "absent_ingredient": "salmon",
    },
    {
        "name": "lentil_stew",
        "ingredients": ("lentil", "turmeric", "onion"),
        "region": "Indian",
        "diet": "vegan",
        "instruction": "simmer",
        "process": "boil",
        "utensil": "pot",
        "flavor": ("turmeric", "spice"),
        "recipe_nutrition": ("fiber", "above", 10, "g"),
        "ingredient_nutrition": ("thiamin", "above", 0.2, "mg"),
        "ingredient_nutrition_ingredient": "lentil",
        "absent_ingredient": "chicken",
    },
    {
        "name": "lemonade",
        "ingredients": ("sugar", "water", "lemon juice"),
        "region": "American",
        "diet": "vegan",
        "instruction": "stir",
        "process": "mix",
        "utensil": "spoon",
        "flavor": ("lemon juice", "citrus"),
        "recipe_nutrition": ("sugar", "above", 20, "g"),
        "ingredient_nutrition": ("water", "above", 80, "g"),
        "ingredient_nutrition_ingredient": "water",
        "absent_ingredient": "chicken",
    },
    {
        "name": "carrot_lentil",
        "ingredients": ("lentil", "carrot", "garlic"),
        "region": "Middle Eastern",
        "diet": "vegan",
        "instruction": "simmer",
        "process": "blend",
        "utensil": "blender",
        "flavor": ("carrot", "root vegetable"),
        "recipe_nutrition": ("magnesium", "above", 80, "mg"),
        "ingredient_nutrition": ("fiber", "above", 3, "g"),
        "ingredient_nutrition_ingredient": "lentil",
        "absent_ingredient": "salmon",
    },
    {
        "name": "chicken_tomato",
        "ingredients": ("chicken", "tomato", "basil"),
        "region": "American",
        "diet": None,
        "instruction": "saute",
        "process": "saute",
        "utensil": "skillet",
        "flavor": ("tomato", "nightshade"),
        "recipe_nutrition": ("protein", "above", 30, "g"),
        "ingredient_nutrition": ("cholesterol", "above", 60, "mg"),
        "ingredient_nutrition_ingredient": "chicken",
        "absent_ingredient": "onion",
    },
    {
        "name": "spinach_pasta",
        "ingredients": ("pasta", "spinach", "basil"),
        "region": "Italian",
        "diet": None,
        "instruction": "boil",
        "process": "toss",
        "utensil": "bowl",
        "flavor": ("spinach", "leafy green"),
        "recipe_nutrition": ("calories", "under", 520, "calories"),
        "ingredient_nutrition": ("vitamin k", "above", 50, "g"),
        "ingredient_nutrition_ingredient": "spinach",
        "absent_ingredient": "salmon",
    },
    {
        "name": "tomato_salmon",
        "ingredients": ("salmon", "tomato", "basil"),
        "region": "Mediterranean",
        "diet": "pescetarian",
        "instruction": "bake",
        "process": "simmer",
        "utensil": "oven",
        "flavor": ("salmon", "fish"),
        "recipe_nutrition": ("fat", "above", 20, "g"),
        "ingredient_nutrition": ("fat", "above", 10, "g"),
        "ingredient_nutrition_ingredient": "salmon",
        "absent_ingredient": "onion",
    },
]


LEAD_INS = (
    "Show",
    "Find",
    "List",
    "Give me",
    "Return",
)


def _two_ingredients(items: tuple[str, ...], offset: int) -> tuple[str, str]:
    first = items[offset % len(items)]
    second = items[(offset + 1) % len(items)]
    return first, second


def generate_cases(count: int) -> list[StressCase]:
    cases: list[StressCase] = []
    templates = [
        _case_ingredient_pair,
        _case_ingredient_exclusion,
        _case_region_ingredient_instruction,
        _case_diet_ingredient_flavor,
        _case_recipe_nutrition,
        _case_ingredient_nutrition,
        _case_recipe_and_ingredient_nutrition,
        _case_instruction_only,
        _case_process_or_utensil,
        _case_time_instruction_ingredient,
    ]

    index = 0
    while len(cases) < count:
        blueprint = RECIPE_BLUEPRINTS[index % len(RECIPE_BLUEPRINTS)]
        template = templates[(index // len(RECIPE_BLUEPRINTS)) % len(templates)]
        lead = LEAD_INS[index % len(LEAD_INS)]
        case = template(len(cases) + 1, blueprint, lead, index)
        cases.append(case)
        index += 1
    return cases


def _case_ingredient_pair(case_id: int, bp: dict[str, Any], lead: str, index: int) -> StressCase:
    first, second = _two_ingredients(bp["ingredients"], index)
    query = f"{lead} recipes containing {first} and {second}."
    return StressCase(
        case_id,
        "ingredient_pair",
        query,
        required_tables=("recipes", "ingredient_phrase"),
        forbidden_tables=("micronutrients", "ingredient_nutrition", "ingredient_flavor", "instructions"),
    )


def _case_ingredient_exclusion(case_id: int, bp: dict[str, Any], lead: str, index: int) -> StressCase:
    first, second = _two_ingredients(bp["ingredients"], index)
    excluded = str(bp["absent_ingredient"])
    query = f"{lead} recipes with {first} and {second} but no {excluded}."
    return StressCase(
        case_id,
        "ingredient_exclusion",
        query,
        required_tables=("recipes", "ingredient_phrase"),
        forbidden_tables=("micronutrients", "ingredient_nutrition", "ingredient_flavor", "instructions"),
    )


def _case_region_ingredient_instruction(case_id: int, bp: dict[str, Any], lead: str, index: int) -> StressCase:
    ingredient = bp["ingredients"][index % len(bp["ingredients"])]
    query = (
        f"{lead} {bp['region']} recipes that use {ingredient} "
        f"and whose instructions mention {bp['instruction']}."
    )
    return StressCase(
        case_id,
        "region_ingredient_instruction",
        query,
        required_tables=("recipes", "ingredient_phrase", "instructions"),
        forbidden_tables=("micronutrients", "ingredient_nutrition", "ingredient_flavor"),
    )


def _case_diet_ingredient_flavor(case_id: int, bp: dict[str, Any], lead: str, index: int) -> StressCase:
    ingredient, flavor = bp["flavor"]
    diet_prefix = f"{bp['diet']} " if bp.get("diet") else ""
    query = f"{lead} {diet_prefix}recipes with {ingredient} where {ingredient} is categorized as {flavor}."
    return StressCase(
        case_id,
        "diet_ingredient_flavor",
        query,
        required_tables=("recipes", "ingredient_phrase", "ingredient_flavor"),
        forbidden_tables=("micronutrients", "ingredient_nutrition", "instructions"),
    )


def _case_recipe_nutrition(case_id: int, bp: dict[str, Any], lead: str, index: int) -> StressCase:
    nutrient, op, value, unit = bp["recipe_nutrition"]
    ingredient = bp["ingredients"][index % len(bp["ingredients"])]
    query = f"{lead} recipes with {ingredient} where the full recipe has {nutrient} {op} {value} {unit}."
    return StressCase(
        case_id,
        "recipe_nutrition",
        query,
        required_tables=("recipes", "ingredient_phrase", "micronutrients"),
        forbidden_tables=("ingredient_nutrition", "ingredient_flavor", "instructions"),
    )


def _case_ingredient_nutrition(case_id: int, bp: dict[str, Any], lead: str, index: int) -> StressCase:
    nutrient, op, value, unit = bp["ingredient_nutrition"]
    ingredient = str(bp["ingredient_nutrition_ingredient"])
    query = f"{lead} recipes with {ingredient} where at least one ingredient has {nutrient} {op} {value} {unit}."
    return StressCase(
        case_id,
        "ingredient_nutrition",
        query,
        required_tables=("recipes", "ingredient_phrase", "ingredient_nutrition"),
        forbidden_tables=("micronutrients", "ingredient_flavor", "instructions"),
    )


def _case_recipe_and_ingredient_nutrition(case_id: int, bp: dict[str, Any], lead: str, index: int) -> StressCase:
    ingredient = str(bp["ingredient_nutrition_ingredient"])
    nutrient, op, value, unit = bp["ingredient_nutrition"]
    query = (
        f"{lead} recipes with {ingredient} where the full recipe has cholesterol below 100 mg "
        f"but at least one ingredient has {nutrient} {op} {value} {unit}."
    )
    return StressCase(
        case_id,
        "recipe_and_ingredient_nutrition",
        query,
        required_tables=("recipes", "ingredient_phrase", "micronutrients", "ingredient_nutrition"),
        forbidden_tables=("ingredient_flavor", "instructions"),
    )


def _case_instruction_only(case_id: int, bp: dict[str, Any], lead: str, index: int) -> StressCase:
    query = f"{lead} recipes where the instruction steps mention {bp['instruction']}."
    return StressCase(
        case_id,
        "instruction_only",
        query,
        required_tables=("recipes", "instructions"),
        forbidden_tables=("micronutrients", "ingredient_phrase", "ingredient_nutrition", "ingredient_flavor"),
    )


def _case_process_or_utensil(case_id: int, bp: dict[str, Any], lead: str, index: int) -> StressCase:
    if index % 2 == 0:
        query = f"{lead} recipes that use a {bp['utensil']}."
    else:
        query = f"{lead} recipes that involve {bp['process']}."
    return StressCase(
        case_id,
        "base_process_or_utensil",
        query,
        required_tables=("recipes",),
        forbidden_tables=("micronutrients", "ingredient_phrase", "ingredient_nutrition", "ingredient_flavor", "instructions"),
    )


def _case_time_instruction_ingredient(case_id: int, bp: dict[str, Any], lead: str, index: int) -> StressCase:
    ingredient = bp["ingredients"][index % len(bp["ingredients"])]
    query = (
        f"{lead} recipes under 60 minutes where the steps mention {bp['instruction']} "
        f"and the ingredient phrase contains {ingredient}."
    )
    return StressCase(
        case_id,
        "time_instruction_ingredient",
        query,
        required_tables=("recipes", "ingredient_phrase", "instructions"),
        forbidden_tables=("micronutrients", "ingredient_nutrition", "ingredient_flavor"),
    )


def extract_tables(sql: str) -> list[str]:
    tables: list[str] = []
    for match in TABLE_RE.finditer(sql):
        table = match.group(1)
        if table not in tables:
            tables.append(table)
    return tables


def sqlite_tables(db_path: Path) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        return {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }


def run_case(case: StressCase, db_path: Path, known_tables: set[str], row_limit: int) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "case_id": case.case_id,
        "category": case.category,
        "query": case.query,
        "required_tables": list(case.required_tables),
        "forbidden_tables": list(case.forbidden_tables),
        "ok": False,
        "errors": [],
    }
    errors: list[str] = result["errors"]

    try:
        generated = generated_sql(case.query)
        compiled = generated["sql"]
        sql = str(compiled.get("query") or "")
        params = list(compiled.get("params") or [])
        tables = extract_tables(sql)
        result.update({
            "sql": sql,
            "params": params,
            "tables": tables,
            "intent": generated.get("intent"),
            "validation": compiled.get("validation", {}),
        })

        if LEGACY_IDENTIFIER_RE.search(sql):
            errors.append("legacy_identifier_in_sql")

        unknown_tables = sorted(set(tables) - known_tables)
        if unknown_tables:
            errors.append(f"unknown_tables={unknown_tables}")

        missing_tables = [table for table in case.required_tables if table not in tables]
        if missing_tables:
            errors.append(f"missing_required_tables={missing_tables}")

        forbidden_tables = [table for table in case.forbidden_tables if table in tables]
        if forbidden_tables:
            errors.append(f"forbidden_tables_present={forbidden_tables}")

        validation = compiled.get("validation") or {}
        if not validation.get("ok"):
            errors.append(f"sql_validation_errors={validation.get('errors')}")

        total_count = count_for_sql(db_path, sql, params)
        columns, rows = rows_for_sql(db_path, sql, params, row_limit)
        verification = verify_generated_result(
            db_path,
            generated.get("verifier_intent", generated.get("intent", {})),
            compiled,
            rows,
            total_count,
        )
        result.update({
            "columns": columns,
            "row_sample_count": len(rows),
            "total_count": total_count,
            "verification": verification,
        })
        if total_count < case.min_rows:
            errors.append(f"row_count_below_min={total_count}<{case.min_rows}")
        if not verification.get("ok"):
            failed_checks = [
                check
                for check in verification.get("checks", [])
                if not check.get("ok")
            ]
            errors.append(f"verification_failed={failed_checks}")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        result["ok"] = not errors
    return result


def write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic stress suite against the speech-to-SQL pipeline.")
    parser.add_argument("--count", type=int, default=1000, help="Number of generated natural-language cases.")
    parser.add_argument("--row-limit", type=int, default=10, help="Rows to sample for semantic verification.")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "docs" / "verification" / "stress_1000_latest.json")
    parser.add_argument("--rebuild-db", action="store_true", help="Rebuild the SQLite dummy database before running.")
    parser.add_argument("--show-failures", type=int, default=10, help="Number of failures to print.")
    args = parser.parse_args()

    started = time.perf_counter()
    if args.rebuild_db or not DB_PATH.exists():
        build_database()

    known_tables = sqlite_tables(DB_PATH)
    cases = generate_cases(args.count)
    results = [
        run_case(case, DB_PATH, known_tables, args.row_limit)
        for case in cases
    ]
    failures = [result for result in results if not result["ok"]]
    by_category = Counter(result["category"] for result in results)
    failed_by_category = Counter(result["category"] for result in failures)
    table_usage = Counter(
        table
        for result in results
        for table in result.get("tables", [])
    )

    payload = {
        "ok": not failures,
        "case_count": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "database": str(DB_PATH),
        "known_tables": sorted(known_tables),
        "by_category": dict(sorted(by_category.items())),
        "failed_by_category": dict(sorted(failed_by_category.items())),
        "table_usage": dict(sorted(table_usage.items())),
        "failures": failures,
        "results": results,
    }
    write_report(args.report, payload)

    print(f"Stress cases: {payload['case_count']}")
    print(f"Passed: {payload['passed']}")
    print(f"Failed: {payload['failed']}")
    print(f"Elapsed: {payload['elapsed_seconds']}s")
    print(f"Report: {args.report}")
    print("By category:")
    for category, total in sorted(by_category.items()):
        failed = failed_by_category.get(category, 0)
        print(f"  {category}: {total - failed}/{total} passed")
    print("Table usage:")
    for table, total in sorted(table_usage.items()):
        print(f"  {table}: {total}")

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
