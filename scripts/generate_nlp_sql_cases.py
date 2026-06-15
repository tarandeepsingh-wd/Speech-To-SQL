from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from speech_to_sql.m2_nlu.data.db_loader import DB_INGREDIENTS
from speech_to_sql.common.config import load_recipe_db_config, load_sql_schema_config


DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "generated_cases"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate NLP-to-SQL test cases without running or grading them."
    )
    parser.add_argument("--count", type=int, default=360, help="Number of cases to generate, max 400.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--stdout", action="store_true", help="Print JSON to terminal instead of writing files.")
    args = parser.parse_args()

    count = max(1, min(args.count, 400))
    cases = generate_cases(count)

    if args.stdout:
        print(json.dumps(cases, indent=2))
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "nlp_sql_cases.jsonl"
    json_path = out_dir / "nlp_sql_cases.json"
    csv_path = out_dir / "nlp_sql_cases.csv"

    json_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=True) + "\n")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "query",
                "expected_intent_json",
                "expected_schema_json",
                "expected_sql_properties_json",
            ],
        )
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "id": case["id"],
                "query": case["query"],
                "expected_intent_json": json.dumps(case["expected"]["intent"], ensure_ascii=True),
                "expected_schema_json": json.dumps(case["expected"]["schema"], ensure_ascii=True),
                "expected_sql_properties_json": json.dumps(case["expected"]["sql_properties"], ensure_ascii=True),
            })

    print(f"Generated {len(cases)} cases")
    print(f"JSON : {json_path}")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV  : {csv_path}")
    return 0


def generate_cases(count: int) -> list[dict[str, Any]]:
    config = load_recipe_db_config()
    schema = load_sql_schema_config()
    ingredients = _available_ingredients()
    cuisines = list(config.get("cuisine_aliases", {}))
    diets = ["vegan", "vegetarian", "pescetarian", "ovo-vegetarian", "lacto-vegetarian"]
    nutrient_specs = [
        ("calories", "under", "<", 300),
        ("protein", "at least", ">=", 20),
        ("carbs", "below", "<", 50),
        ("fat", "under", "<", 12),
        ("sodium", "less than", "<", 500),
    ]
    sort_specs = [
        ("prepared time from most to least", "prep_time", "DESC"),
        ("prepared time from least to most", "prep_time", "ASC"),
        ("cook time from highest to lowest", "cook_time", "DESC"),
        ("calories from lowest to highest", "calories", "ASC"),
        ("protein from highest to lowest", "protein", "DESC"),
    ]

    cases: list[dict[str, Any]] = []

    for ingredient in ingredients:
        _add(cases, schema, f"recipes with {ingredient}", ingredients=[ingredient])

    for a, b in _windows(ingredients, 2, 45):
        _add(cases, schema, f"recipes with {a} and {b}", ingredients=[a, b], include_operator="AND")
        _add(cases, schema, f"recipes with {a} or {b}", ingredients=[a, b], include_operator="OR")

    for a, b, c in _windows(ingredients, 3, 55):
        _add(cases, schema, f"recipes using {a} {b} {c}", ingredients=[a, b, c], include_operator="AND")

    for index, (a, b) in enumerate(_windows(ingredients, 2, 35)):
        exclude = ingredients[(index + 9) % len(ingredients)]
        _add(cases, schema, f"recipes with {a} and {b} without {exclude}", ingredients=[a, b], exclude=[exclude])

    for diet in diets:
        _add(cases, schema, f"show {diet} recipes", dietary=[diet])
    for cuisine in cuisines[:20]:
        _add(cases, schema, f"show {cuisine} recipes", cuisine=cuisine)

    for nutrient, word, operator, value in nutrient_specs:
        _add(cases, schema, f"recipes with {nutrient} {word} {value}", nutrition=[{
            "nutrient": nutrient,
            "operator": operator,
            "value": value,
        }])

    for phrase, field, direction in sort_specs:
        _add(cases, schema, f"show recipes ordered by {phrase}", order_by={"field": field, "direction": direction})
        _add(cases, schema, f"top 5 recipes by {phrase}", order_by={"field": field, "direction": direction}, limit=5)

    for index, ingredient in enumerate(ingredients):
        nutrient, word, operator, value = nutrient_specs[index % len(nutrient_specs)]
        _add(cases, schema, f"{ingredient} recipes with {nutrient} {word} {value}", ingredients=[ingredient], nutrition=[{
            "nutrient": nutrient,
            "operator": operator,
            "value": value,
        }])

    for index, ingredient in enumerate(ingredients[:30]):
        diet = diets[index % len(diets)]
        _add(cases, schema, f"{diet} {ingredient} recipes", ingredients=[ingredient], dietary=[diet])

    for index, ingredient in enumerate(ingredients[:30]):
        cuisine = cuisines[index % len(cuisines)]
        _add(cases, schema, f"{cuisine} recipes with {ingredient}", ingredients=[ingredient], cuisine=cuisine)

    for index, (a, b, c) in enumerate(_windows(ingredients, 3, 60)):
        nutrient, word, operator, value = nutrient_specs[index % len(nutrient_specs)]
        phrase, field, direction = sort_specs[index % len(sort_specs)]
        _add(
            cases,
            schema,
            f"top 10 recipes with {a} {b} {c} and {nutrient} {word} {value} ordered by {phrase}",
            ingredients=[a, b, c],
            nutrition=[{"nutrient": nutrient, "operator": operator, "value": value}],
            order_by={"field": field, "direction": direction},
            limit=10,
        )

    for index, (a, b, c) in enumerate(_windows(ingredients, 3, 60)):
        diet = diets[index % len(diets)]
        cuisine = cuisines[index % len(cuisines)]
        exclude = ingredients[(index + 11) % len(ingredients)]
        nutrient, word, operator, value = nutrient_specs[index % len(nutrient_specs)]
        _add(
            cases,
            schema,
            f"top 7 {diet} {cuisine} recipes with {a} {b} {c} without {exclude} with {nutrient} {word} {value}",
            ingredients=[a, b, c],
            exclude=[exclude],
            dietary=[diet],
            cuisine=cuisine,
            nutrition=[{"nutrient": nutrient, "operator": operator, "value": value}],
            limit=7,
        )

    invalid_or_unsupported = [
        ("low fodmap recipes", {"should_fail": True, "reason": "unsupported dietary term"}),
        ("average calories recipes", {"should_fail": True, "reason": "unsupported aggregate"}),
        ("show ingredients under 300 calories", {"should_fail": True, "reason": "unsupported cross-root filter"}),
    ]
    for query, expected in invalid_or_unsupported:
        _add_raw(cases, query, _expected_payload(expected, schema))

    return cases[:count]


def _add(
    cases: list[dict[str, Any]],
    schema: dict[str, Any],
    query: str,
    *,
    ingredients: list[str] | None = None,
    exclude: list[str] | None = None,
    include_operator: str = "AND",
    dietary: list[str] | None = None,
    cuisine: str | None = None,
    nutrition: list[dict[str, Any]] | None = None,
    order_by: dict[str, str] | None = None,
    limit: int | None = None,
) -> None:
    expected: dict[str, Any] = {}
    if ingredients or exclude:
        expected["ingredient_logic"] = {
            "include": ingredients or [],
            "exclude": exclude or [],
            "include_operator": include_operator,
        }
    if dietary:
        expected["dietary_styles"] = dietary
    if cuisine:
        expected["cuisine"] = cuisine
    if nutrition:
        expected["nutrition_filters"] = nutrition
    if order_by:
        expected["order_by"] = order_by
    if limit is not None:
        expected["limit"] = limit
    _add_raw(cases, query, _expected_payload(expected, schema))


def _add_raw(cases: list[dict[str, Any]], query: str, expected: dict[str, Any]) -> None:
    cases.append({
        "id": f"CASE_{len(cases) + 1:04d}",
        "query": query,
        "expected": expected,
    })


def _expected_payload(intent: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "intent": intent,
        "schema": _expected_schema(intent, schema),
        "sql_properties": _expected_sql_properties(intent, schema),
    }


def _expected_schema(intent: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    if intent.get("should_fail"):
        return {
            "should_fail": True,
            "reason": intent.get("reason"),
            "tables": [],
            "columns": [],
            "joins": [],
        }

    base_table = schema["base_table"]["name"]
    ingredients_relation = schema["relations"]["ingredients"]
    nutrition_columns = schema["relations"]["nutrition"]["columns"]
    dietary_columns = schema["relations"]["dietary"]["columns"]
    order_fields = schema["order_fields"]
    root_alias = schema["base_table"]["alias"]

    tables = [{
        "table": base_table,
        "alias": root_alias,
        "role": "recipe root",
    }]
    columns: list[dict[str, Any]] = []
    joins: list[dict[str, Any]] = []

    ingredient_logic = intent.get("ingredient_logic") or {}
    include = ingredient_logic.get("include") or []
    exclude = ingredient_logic.get("exclude") or []
    if include or exclude:
        tables.append({
            "table": ingredients_relation["table"],
            "alias": ingredients_relation["alias"],
            "role": "ingredient match",
        })
        joins.append({
            "from_table": ingredients_relation["table"],
            "from_column": "recipe_no",
            "to_table": base_table,
            "to_column": "Recipe_id",
            "reason": "connect ingredient rows to recipe rows",
        })
        for value in include:
            columns.extend(_ingredient_columns(ingredients_relation, value, include=True))
        for value in exclude:
            columns.extend(_ingredient_columns(ingredients_relation, value, include=False))

    for item in intent.get("nutrition_filters") or []:
        nutrient = item["nutrient"]
        columns.append({
            "table": base_table,
            "alias": root_alias,
            "column": nutrition_columns.get(nutrient),
            "logical_field": nutrient,
            "operator": item["operator"],
            "value": item["value"],
            "role": "nutrition filter",
        })

    for style in intent.get("dietary_styles") or []:
        column = dietary_columns.get(_diet_column_key(style))
        columns.append({
            "table": base_table,
            "alias": root_alias,
            "column": column,
            "logical_field": style,
            "operator": "IN",
            "value": "truthy_values",
            "role": "dietary filter",
        })

    if intent.get("cuisine"):
        for field in ("region", "sub_region", "continent"):
            columns.append({
                "table": base_table,
                "alias": root_alias,
                "column": schema["base_columns"][field],
                "logical_field": field,
                "operator": "=",
                "value": intent["cuisine"],
                "role": "cuisine/category filter",
            })

    if intent.get("order_by"):
        order = intent["order_by"]
        order_info = order_fields[order["field"]]
        source = order_info["source"]
        logical_field = order_info["field"]
        column = (
            schema["base_columns"][logical_field]
            if source == "base"
            else schema["relations"][source]["columns"][logical_field]
        )
        columns.append({
            "table": base_table,
            "alias": root_alias,
            "column": column,
            "logical_field": order["field"],
            "operator": f"ORDER BY {order['direction']}",
            "value": None,
            "role": "sort column",
        })

    return {
        "tables": _dedupe_dicts(tables),
        "columns": [item for item in columns if item.get("column")],
        "joins": joins,
    }


def _expected_sql_properties(intent: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    if intent.get("should_fail"):
        return {
            "valid": False,
            "expected_error_reason": intent.get("reason"),
        }

    operators = ["SELECT"]
    checks: list[str] = []
    params: list[Any] = []

    ingredient_logic = intent.get("ingredient_logic") or {}
    include = ingredient_logic.get("include") or []
    exclude = ingredient_logic.get("exclude") or []
    include_operator = ingredient_logic.get("include_operator", "AND")
    if include or exclude:
        operators.extend(["JOIN", "WHERE", "=", "LIKE"])
        params.extend(_ingredient_params(include + exclude))
        if include_operator == "OR":
            operators.append("IN")
            checks.append("ingredient OR condition can be rendered as IN/OR semantics")
        elif len(include) > 1:
            operators.extend(["GROUP BY", "HAVING"])
            checks.append("all included ingredients must match same recipe_id")
        if exclude:
            operators.append("NOT EXISTS")
            checks.append("excluded ingredients must be rendered as anti-match")

    for item in intent.get("nutrition_filters") or []:
        operators.extend(["WHERE", item["operator"]])
        params.append(float(item["value"]))

    if intent.get("dietary_styles"):
        operators.extend(["WHERE", "IN"])
        checks.append("diet column must use configured truthy values")

    if intent.get("cuisine"):
        operators.extend(["WHERE", "="])
        checks.append("cuisine may map to region/sub_region/continent predicates")

    if intent.get("order_by"):
        operators.append("ORDER BY")
        checks.append(f"sort direction must be {intent['order_by']['direction']}")

    if intent.get("limit") is not None:
        operators.append("LIMIT")
        params.append(intent["limit"])

    return {
        "valid": True,
        "operators": _dedupe_strings(operators),
        "param_values": params,
        "checks": checks,
        "must_not_compare_exact_sql_string": True,
    }


def _ingredient_columns(relation: dict[str, Any], value: str, *, include: bool) -> list[dict[str, Any]]:
    return [
        {
            "table": relation["table"],
            "alias": relation["alias"],
            "column": relation["value_column"],
            "logical_field": "ingredient",
            "operator": relation["strategy"]["match_operator"],
            "value": value,
            "role": "included ingredient" if include else "excluded ingredient",
        },
        {
            "table": relation["table"],
            "alias": relation["alias"],
            "column": relation["fallback_value_column"],
            "logical_field": "ingredient",
            "operator": relation["strategy"]["fallback_match_operator"],
            "value": f"%{value}%",
            "role": "included ingredient phrase fallback" if include else "excluded ingredient phrase fallback",
        },
    ]


def _ingredient_params(values: list[str]) -> list[str]:
    params = []
    for value in values:
        params.extend([value, f"%{value}%"])
    return params


def _diet_column_key(style: str) -> str:
    return {
        "vegetarian": "ovo_lacto_vegetarian",
        "pescetarian": "pescetarian",
        "pescatarian": "pescetarian",
    }.get(style, style)


def _dedupe_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for item in items:
        key = json.dumps(item, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_strings(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _available_ingredients() -> list[str]:
    preferred = [
        "chicken", "tomato", "potato", "garlic", "paneer", "spinach", "onion", "rice",
        "egg", "milk", "cheese", "carrot", "beef", "corn", "cabbage", "cauliflower",
        "broccoli", "bell pepper", "ginger", "lemon", "yogurt", "butter", "flour",
        "cinnamon", "parsley", "basil", "turmeric", "cumin", "pasta", "apple",
        "banana", "honey", "almond", "cashew", "chickpea", "shrimp", "salmon",
        "pork", "tofu", "coconut", "cream",
    ]
    available = []
    db_terms = set(DB_INGREDIENTS)
    for term in preferred:
        if term in db_terms:
            available.append(term)
    return available


def _windows(items: list[str], size: int, count: int) -> list[tuple[str, ...]]:
    windows = []
    if len(items) < size:
        return windows
    index = 0
    while len(windows) < count:
        group = tuple(items[(index + offset) % len(items)] for offset in range(size))
        if len(set(group)) == size:
            windows.append(group)
        index += 1
    return windows


if __name__ == "__main__":
    raise SystemExit(main())
