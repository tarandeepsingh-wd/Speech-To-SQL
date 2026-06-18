from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

from seed_temp_db import DB_PATH, build_database


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


BRIEF_COLUMNS = [
    "recipe_id",
    "recipe_title",
    "ingredient",
    "ingredient_phrase",
    "calories",
    "protein (g)",
    "carbohydrate, by difference (g)",
    "thiamin (mg)",
    "vegan",
    "region",
]

TRUTHY_TEXT = {"1", "1.0", "true", "yes", "y"}
NUTRITION_COLUMNS = {
    "calories": "calories",
    "energy": "energy (kcal)",
    "energy_kcal": "energy (kcal)",
    "protein": "protein (g)",
    "carbohydrate": "carbohydrate, by difference (g)",
    "carbohydrates": "carbohydrate, by difference (g)",
    "carbs": "carbohydrate, by difference (g)",
    "fat": "total lipid (fat) (g)",
    "fiber": "fiber, total dietary (g)",
    "sodium": "sodium, na (mg)",
    "sugar": "sugars, total (g)",
    "cholesterol": "cholesterol (mg)",
    "magnesium": "magnesium, mg (mg)",
    "folate_total": "folate, total (g)",
    "folate": "folate, total (g)",
    "thiamin": "thiamin (mg)",
    "vitamin_c": "vitamin c, total ascorbic acid (mg)",
    "vitamin_k": "vitamin k (phylloquinone) (g)",
    "water": "water (g)",
}


def generated_sql(query: str) -> dict[str, Any]:
    from speech_to_sql.m2_nlu.entity_tagger import tag_entities
    from speech_to_sql.common.config import load_recipe_db_config
    from speech_to_sql.m3_query_ir.intent_preparer import prepare_intent
    from speech_to_sql.m3_query_ir.time_resolver import resolve_time_filter
    from speech_to_sql.m4_sql.sql_query_builder import build_sql_query

    tags = tag_entities(query)
    tags["transcript"] = query
    prepared = prepare_intent(tags, transcript=query)
    compiled = build_sql_query(prepared, transcript=query)
    time_filters = list(prepared.get("time_filters") or [])
    if not time_filters:
        resolved_time = resolve_time_filter(prepared, query.lower(), load_recipe_db_config())
        if resolved_time:
            field, min_value, max_value = resolved_time
            time_filters.append({
                "field": field,
                "min": min_value,
                "max": max_value,
                "source": "time_resolver",
            })
    public_intent = {
        "ingredients": _normalized_ingredients_for_display(prepared),
        "dietary_styles": prepared.get("dietary_styles", []),
        "utensils": prepared.get("utensils", []),
        "actions": prepared.get("actions", []),
        "category_filters": prepared.get("category_filters", []),
        "nutrition_filters": prepared.get("nutrition_filters", []),
        "time_filters": time_filters,
        "ingredient_logic": prepared.get("ingredient_logic", {}),
    }
    verifier_intent = {
        **public_intent,
        "_transcript": query,
        "raw_spans": prepared.get("raw_spans", []),
    }
    return {
        "intent": public_intent,
        "verifier_intent": verifier_intent,
        "sql": compiled,
    }


def _normalized_ingredients_for_display(intent: dict[str, Any]) -> list[Any]:
    logic = intent.get("ingredient_logic") or {}
    if "include" in logic or "exclude" in logic:
        return list(logic.get("include") or []) + list(logic.get("exclude") or [])
    return list(intent.get("ingredients") or [])


def rows_for_sql(
    db_path: Path,
    sql: str,
    params: list[Any],
    limit: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql, params)
        columns = [column[0] for column in cursor.description or []]
        return columns, [dict(row) for row in cursor.fetchmany(limit)]


def count_for_sql(db_path: Path, sql: str, params: list[Any]) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            f"SELECT COUNT(*) FROM ({sql}) AS temp_query_count",
            params,
        ).fetchone()
        return int(row[0])


def verify_generated_result(
    db_path: Path,
    intent: dict[str, Any],
    compiled: dict[str, Any],
    rows: list[dict[str, Any]],
    total_count: int,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    semantic_conflicts = _semantic_conflict_warnings(compiled)
    _add_check(
        checks,
        "sql_validation",
        bool((compiled.get("validation") or {}).get("ok")),
        "Generated SQL passed the schema validator.",
        (compiled.get("validation") or {}).get("errors", []),
    )
    result_count_ok = total_count > 0 or bool(semantic_conflicts)
    if total_count > 0:
        result_count_message = f"SQL returned {total_count} row(s) from the dummy database."
        result_count_failures = []
    elif semantic_conflicts:
        result_count_message = (
            "SQL returned 0 row(s), which is consistent with the semantic "
            "conflict warning(s)."
        )
        result_count_failures = []
    else:
        result_count_message = "SQL returned 0 row(s) from the dummy database."
        result_count_failures = ["No rows matched the generated SQL."]
    _add_check(
        checks,
        "result_count",
        result_count_ok,
        result_count_message,
        result_count_failures,
    )

    if not rows:
        return {
            "ok": all(item["ok"] for item in checks),
            "checked_rows": 0,
            "total_count": total_count,
            "semantic_conflicts": semantic_conflicts,
            "checks": checks,
        }

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        _verify_ingredients(conn, checks, intent, rows)
        _verify_excluded_ingredients(conn, checks, intent, rows)
        _verify_nutrition_filters(conn, checks, intent, rows)
        _verify_time_filters(conn, checks, intent, rows)
        _verify_category_filters(conn, checks, intent, rows)
        _verify_dietary_styles(conn, checks, intent, rows)
        _verify_instruction_terms(conn, checks, intent, rows)

    _verify_utensils(checks, intent, rows)

    return {
        "ok": all(item["ok"] for item in checks),
        "checked_rows": len(rows),
        "total_count": total_count,
        "semantic_conflicts": semantic_conflicts,
        "checks": checks,
    }


def _semantic_conflict_warnings(compiled: dict[str, Any]) -> list[str]:
    warnings = (compiled.get("validation") or {}).get("warnings") or []
    return [
        str(warning)
        for warning in warnings
        if "semantic conflict" in str(warning).lower()
    ]


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    ok: bool,
    message: str,
    failures: list[Any],
) -> None:
    checks.append({
        "name": name,
        "ok": ok,
        "message": message,
        "failures": failures[:5],
    })


def _verify_ingredients(
    conn: sqlite3.Connection,
    checks: list[dict[str, Any]],
    intent: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    required = _required_ingredients(intent)
    if not required:
        return
    include_operator = _required_ingredient_operator(intent)
    failures = []
    for row in rows:
        recipe_id = _recipe_id(row)
        ingredients = _ingredients_for_recipe(conn, recipe_id)
        if include_operator == "OR":
            present = [
                ingredient
                for ingredient in required
                if _ingredient_present(ingredient, ingredients)
            ]
            if not present:
                failures.append({
                    "recipe_id": recipe_id,
                    "missing_any_of": required,
                    "available": ingredients,
                })
            continue
        missing = [
            ingredient
            for ingredient in required
            if not _ingredient_present(ingredient, ingredients)
        ]
        if missing:
            failures.append({
                "recipe_id": recipe_id,
                "missing": missing,
                "available": ingredients,
            })
    _add_check(
        checks,
        "required_ingredients",
        not failures,
        (
            "Every returned row contains at least one required ingredient."
            if include_operator == "OR"
            else "Every returned row contains the required ingredient(s)."
        ),
        failures,
    )


def _verify_excluded_ingredients(
    conn: sqlite3.Connection,
    checks: list[dict[str, Any]],
    intent: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    excluded = _excluded_ingredients(intent)
    if not excluded:
        return
    failures = []
    for row in rows:
        recipe_id = _recipe_id(row)
        ingredients = _ingredients_for_recipe(conn, recipe_id)
        present = [
            ingredient
            for ingredient in excluded
            if _ingredient_present(ingredient, ingredients)
        ]
        if present:
            failures.append({
                "recipe_id": recipe_id,
                "present": present,
                "available": ingredients,
            })
    _add_check(
        checks,
        "excluded_ingredients",
        not failures,
        "No returned row contains excluded ingredient(s).",
        failures,
    )


def _verify_dietary_styles(
    conn: sqlite3.Connection,
    checks: list[dict[str, Any]],
    intent: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    styles = [
        str(style).strip()
        for style in intent.get("dietary_styles", [])
        if str(style).strip()
    ]
    if not styles:
        return
    failures = []
    for row in rows:
        missing = [
            style
            for style in styles
            if not _truthy(row.get(style, _dietary_value_for_recipe(conn, _recipe_id(row), style)))
        ]
        if missing:
            failures.append({
                "recipe_id": _recipe_id(row),
                "missing": missing,
            })
    _add_check(
        checks,
        "dietary_styles",
        not failures,
        "Every returned row satisfies the dietary style filter(s).",
        failures,
    )


def _dietary_value_for_recipe(
    conn: sqlite3.Connection,
    recipe_id: int | None,
    style: str,
) -> Any:
    if recipe_id is None:
        return None
    if style not in {"vegan", "pescetarian", "ovo_vegetarian", "lacto_vegetarian", "ovo_lacto_vegetarian"}:
        return None
    row = conn.execute(
        f"SELECT `{style}` FROM recipes WHERE recipe_id = ?",
        [recipe_id],
    ).fetchone()
    return row[0] if row else None


def _verify_nutrition_filters(
    conn: sqlite3.Connection,
    checks: list[dict[str, Any]],
    intent: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    filters = [
        item
        for item in intent.get("nutrition_filters", [])
        if isinstance(item, dict)
    ]
    if not filters:
        return
    failures = []
    skipped = []
    for item in filters:
        nutrient = str(item.get("nutrient") or "").strip()
        operator = str(item.get("operator") or "").strip()
        expected = item.get("value")
        source = _nutrition_filter_source(item)
        column = (
            _ingredient_nutrition_column(nutrient)
            if source == "ingredient_nutrition"
            else _nutrition_column(nutrient, rows[0])
        )
        if not column or expected is None or not operator:
            skipped.append(item)
            continue
        if source == "ingredient_nutrition":
            _verify_ingredient_nutrition_filter(
                conn,
                failures,
                rows,
                nutrient,
                column,
                operator,
                float(expected),
            )
        else:
            for row in rows:
                actual_value = row.get(column)
                actual = _number(actual_value)
                if actual is None:
                    actual = _recipe_nutrition_value(conn, _recipe_id(row), column)
                    actual_value = actual
                if actual is None or not _compare(actual, operator, float(expected)):
                    failures.append({
                        "recipe_id": _recipe_id(row),
                        "source": source,
                        "nutrient": nutrient,
                        "column": column,
                        "actual": actual_value,
                        "expected": f"{operator} {expected}",
                    })
    _add_check(
        checks,
        "nutrition_filters",
        not failures,
        "Every returned row satisfies the numeric nutrition filter(s).",
        failures,
    )
    if skipped:
        _add_check(
            checks,
            "nutrition_filters_skipped",
            True,
            "Some non-numeric or presence-style nutrition filters were not row-checked.",
            skipped,
        )


def _verify_time_filters(
    conn: sqlite3.Connection,
    checks: list[dict[str, Any]],
    intent: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    filters = [
        item
        for item in intent.get("time_filters", []) or []
        if isinstance(item, dict)
    ]
    if not filters:
        return

    failures = []
    for item in filters:
        field = str(item.get("field") or "total_time")
        min_value = _number(item.get("min"))
        max_value = _number(item.get("max"))
        for row in rows:
            actual_value = row.get(field)
            actual = _number(actual_value)
            if actual is None:
                actual = _recipe_nutrition_value(conn, _recipe_id(row), field)
                actual_value = actual
            if actual is None:
                failures.append({
                    "recipe_id": _recipe_id(row),
                    "field": field,
                    "actual": actual_value,
                    "expected": item,
                })
                continue
            if min_value is not None and actual < min_value:
                failures.append({
                    "recipe_id": _recipe_id(row),
                    "field": field,
                    "actual": actual,
                    "expected": f">= {min_value}",
                })
            if max_value is not None and actual > max_value:
                failures.append({
                    "recipe_id": _recipe_id(row),
                    "field": field,
                    "actual": actual,
                    "expected": f"<= {max_value}",
                })

    _add_check(
        checks,
        "time_filters",
        not failures,
        "Every returned row satisfies the time filter(s).",
        failures,
    )


def _verify_ingredient_nutrition_filter(
    conn: sqlite3.Connection,
    failures: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    nutrient: str,
    column: str,
    operator: str,
    expected: float,
) -> None:
    for row in rows:
        recipe_id = _recipe_id(row)
        values = _ingredient_nutrition_values(conn, recipe_id, column)
        if any(_compare(value, operator, expected) for value in values):
            continue
        failures.append({
            "recipe_id": recipe_id,
            "source": "ingredient_nutrition",
            "nutrient": nutrient,
            "column": column,
            "actual_values": values[:8],
            "expected": f"any ingredient {operator} {expected}",
        })


def _ingredient_nutrition_values(
    conn: sqlite3.Connection,
    recipe_id: int | None,
    column: str,
) -> list[float]:
    if recipe_id is None:
        return []
    values: list[float] = []
    for row in conn.execute(
        f"""
        SELECT `{column}`
        FROM ingredient_nutrition
        WHERE recipe_id = ?
        """,
        [recipe_id],
    ):
        value = _number(row[0])
        if value is not None:
            values.append(value)
    return values


def _recipe_nutrition_value(
    conn: sqlite3.Connection,
    recipe_id: int | None,
    column: str,
) -> float | None:
    if recipe_id is None:
        return None
    try:
        row = conn.execute(
            f"""
            SELECT `{column}`
            FROM micronutrients
            WHERE recipe_id = ?
            """,
            [recipe_id],
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return _number(row[0]) if row else None


def _nutrition_filter_source(item: dict[str, Any]) -> str:
    for key in ("table_source", "relation", "relation_source", "source"):
        value = str(item.get(key) or "").strip()
        if value in {"nutrition", "ingredient_nutrition"}:
            return value
    return "nutrition"


def _verify_category_filters(
    conn: sqlite3.Connection,
    checks: list[dict[str, Any]],
    intent: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    filters = [
        item
        for item in intent.get("category_filters", [])
        if isinstance(item, dict) and item.get("field")
    ]
    if not filters:
        return
    failures = []
    for item in filters:
        source = str(item.get("source") or "base")
        field = str(item.get("field"))
        expected = str(item.get("value", "")).lower()
        if source == "ingredient_flavor":
            _verify_ingredient_flavor_filter(conn, failures, rows, field, expected, item)
        else:
            for row in rows:
                actual = str(row.get(field, "")).lower()
                if actual != expected:
                    failures.append({
                        "recipe_id": _recipe_id(row),
                        "source": source,
                        "field": field,
                        "actual": row.get(field),
                        "expected": item.get("value"),
                    })
    _add_check(
        checks,
        "category_filters",
        not failures,
        "Every returned row satisfies cuisine/category filter(s).",
        failures,
    )


def _verify_ingredient_flavor_filter(
    conn: sqlite3.Connection,
    failures: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    field: str,
    expected: str,
    item: dict[str, Any],
) -> None:
    column = {
        "dietrx_category": "dietrx_category",
        "flavordb_category": "flavordb_category",
        "generic_name": "generic_name",
        "ingredient": "ingredient",
    }.get(field, field)
    for row in rows:
        recipe_id = _recipe_id(row)
        values = _ingredient_flavor_values(conn, recipe_id, column)
        if expected in {value.lower() for value in values}:
            continue
        failures.append({
            "recipe_id": recipe_id,
            "source": "ingredient_flavor",
            "field": field,
            "actual_values": values[:8],
            "expected": item.get("value"),
        })


def _ingredient_flavor_values(
    conn: sqlite3.Connection,
    recipe_id: int | None,
    column: str,
) -> list[str]:
    if recipe_id is None:
        return []
    values: list[str] = []
    for row in conn.execute(
        f"""
        SELECT ifl.`{column}`
        FROM ingredient_phrase AS ip
        JOIN ingredient_flavor AS ifl
          ON ip.ing_id = ifl.ingid
        WHERE ip.recipe_no = ?
        """,
        [recipe_id],
    ):
        value = str(row[0] or "").strip()
        if value and value not in values:
            values.append(value)
    return values


def _verify_utensils(
    checks: list[dict[str, Any]],
    intent: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    scoped_terms = _instruction_scoped_terms(intent)
    utensils = [
        str(item).strip().lower()
        for item in intent.get("utensils", [])
        if str(item).strip()
        and str(item).strip().lower() not in scoped_terms["utensils"]
    ]
    if not utensils:
        return
    failures = []
    for row in rows:
        row_utensils = str(row.get("utensils", "")).lower()
        missing = [
            utensil
            for utensil in utensils
            if utensil not in row_utensils
        ]
        if missing:
            failures.append({
                "recipe_id": _recipe_id(row),
                "missing": missing,
                "available": row.get("utensils", ""),
            })
    _add_check(
        checks,
        "utensils",
        not failures,
        "Every returned row satisfies utensil filter(s).",
        failures,
    )


def _verify_instruction_terms(
    conn: sqlite3.Connection,
    checks: list[dict[str, Any]],
    intent: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    scoped_terms = _instruction_scoped_terms(intent)
    terms = sorted(scoped_terms["actions"] | scoped_terms["utensils"])
    if not terms:
        return

    failures = []
    for row in rows:
        recipe_id = _recipe_id(row)
        steps = _instruction_steps_for_recipe(conn, recipe_id)
        missing = [
            term
            for term in terms
            if term not in steps
        ]
        if missing:
            failures.append({
                "recipe_id": recipe_id,
                "missing": missing,
                "available": steps[:160],
            })
    _add_check(
        checks,
        "instructions",
        not failures,
        "Every returned row satisfies instruction-step term filter(s).",
        failures,
    )


def _instruction_steps_for_recipe(conn: sqlite3.Connection, recipe_id: int | None) -> str:
    if recipe_id is None:
        return ""
    rows = conn.execute(
        """
        SELECT steps
        FROM instructions
        WHERE recipe_id = ?
        """,
        [recipe_id],
    ).fetchall()
    return " ".join(str(row[0] or "").lower() for row in rows)


def _instruction_scoped_terms(intent: dict[str, Any]) -> dict[str, set[str]]:
    scoped = {"actions": set(), "utensils": set()}
    text = str(intent.get("_transcript") or intent.get("transcript") or intent.get("query") or "")
    if not text:
        return scoped

    for item in intent.get("actions") or []:
        if not isinstance(item, dict) or not item.get("action"):
            continue
        if _span_is_instruction_scoped(text, item.get("span")):
            scoped["actions"].add(str(item["action"]).strip().lower())

    utensil_values = {
        str(value).strip().lower()
        for value in intent.get("utensils") or []
        if str(value).strip()
    }
    for item in intent.get("raw_spans") or []:
        if not isinstance(item, dict):
            continue
        canonical = str(item.get("canonical") or "").strip().lower()
        if canonical not in utensil_values:
            continue
        if _span_is_instruction_scoped(text, item.get("span")):
            scoped["utensils"].add(canonical)
    if len(scoped["actions"]) > 1:
        scoped["actions"].discard("cook")
    return scoped


def _span_is_instruction_scoped(text: str, span: Any) -> bool:
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return False
    try:
        start = int(span[0])
        end = int(span[1]) + 1
    except (TypeError, ValueError):
        return False

    text_lower = text.lower()
    prefix = text_lower[max(0, start - 96):start]
    suffix = text_lower[end:min(len(text_lower), end + 72)]
    instruction_pattern = r"\b(?:instruction|instructions|step|steps|cooking\s+steps?)\b"
    mention_pattern = r"\b(?:mention|mentions|mentioned|contain|contains|containing|include|includes|including|with|in)\b"
    return (
        bool(re.search(instruction_pattern, prefix) and re.search(mention_pattern, prefix))
        or bool(re.search(instruction_pattern, suffix) and re.search(mention_pattern, suffix))
    )


def _required_ingredients(intent: dict[str, Any]) -> list[str]:
    logic = intent.get("ingredient_logic") or {}
    values = logic.get("include") if "include" in logic else intent.get("ingredients") or []
    return _unique_lower(values)


def _required_ingredient_operator(intent: dict[str, Any]) -> str:
    logic = intent.get("ingredient_logic") or {}
    return str(logic.get("include_operator") or "AND").upper()


def _excluded_ingredients(intent: dict[str, Any]) -> list[str]:
    logic = intent.get("ingredient_logic") or {}
    return _unique_lower(logic.get("exclude") or [])


def _unique_lower(values: list[Any]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        item = str(value).strip().lower()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _recipe_id(row: dict[str, Any]) -> int | None:
    value = row.get("recipe_id")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ingredients_for_recipe(conn: sqlite3.Connection, recipe_id: int | None) -> list[str]:
    if recipe_id is None:
        return []
    values = []
    for row in conn.execute(
        """
        SELECT ingredient, ingredient_phrase
        FROM ingredient_phrase
        WHERE recipe_no = ?
        """,
        [recipe_id],
    ):
        for value in row:
            item = str(value or "").strip().lower()
            if item and item not in values:
                values.append(item)
    return values


def _ingredient_present(ingredient: str, available: list[str]) -> bool:
    needle = ingredient.lower()
    return any(
        needle == item or needle in item or item in needle
        for item in available
    )


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in TRUTHY_TEXT


def _nutrition_column(nutrient: str, row: dict[str, Any]) -> str | None:
    lowered_keys = {key.lower(): key for key in row}
    expected = NUTRITION_COLUMNS.get(nutrient.lower())
    if expected:
        return expected
    if nutrient in row:
        return nutrient
    return lowered_keys.get(nutrient.lower())


def _ingredient_nutrition_column(nutrient: str) -> str | None:
    return NUTRITION_COLUMNS.get(nutrient.lower(), nutrient)


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compare(actual: float, operator: str, expected: float) -> bool:
    if operator in {"<", "lt"}:
        return actual < expected
    if operator in {"<=", "lte"}:
        return actual <= expected
    if operator in {">", "gt"}:
        return actual > expected
    if operator in {">=", "gte"}:
        return actual >= expected
    if operator in {"=", "==", "eq"}:
        return actual == expected
    return False


def brief_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows:
        compact_row = {
            column: row[column]
            for column in BRIEF_COLUMNS
            if column in row
        }
        compact.append(compact_row or row)
    return compact


def brief_columns(columns: list[str]) -> dict[str, Any]:
    return {
        "count": len(columns),
        "sample": columns[:12],
    }


def schema_snapshot(db_path: Path) -> dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        result = {}
        for table in tables:
            columns = [
                {
                    "name": row[1],
                    "type": row[2],
                    "notnull": bool(row[3]),
                    "default": row[4],
                    "primary_key": bool(row[5]),
                }
                for row in conn.execute(f"PRAGMA table_info(`{table}`)")
            ]
            count = conn.execute(f"SELECT COUNT(*) FROM `{table}`").fetchone()[0]
            result[table] = {
                "row_count": count,
                "column_count": len(columns),
                "columns": columns,
            }
        return result


def print_result(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=True))


def print_explanation(payload: dict[str, Any]) -> None:
    intent = payload.get("intent") or {}
    sql_info = payload.get("sql") or {}
    verification = payload.get("verification") or {}

    print()
    print("=" * 72)
    print("Speech-to-SQL Dummy DB Verification")
    print("=" * 72)
    print(f"Query       : {payload.get('query')}")
    print(f"Ingredients : {intent.get('ingredients', [])}")
    print(f"Diet        : {intent.get('dietary_styles', [])}")
    print(f"Utensils    : {intent.get('utensils', [])}")
    print(f"Nutrition   : {_format_nutrition_filters(intent.get('nutrition_filters', []))}")
    print()
    print(f"SQL valid   : {_status((sql_info.get('validation') or {}).get('ok'))}")
    print(f"Rows matched: {payload.get('total_count', payload.get('count', 0))}")
    print(f"Rows checked: {verification.get('checked_rows', 0)}")
    print(f"Overall     : {_status(verification.get('ok'))}")
    warnings = (sql_info.get("validation") or {}).get("warnings") or []
    if warnings:
        print()
        print("Warnings")
        for warning in warnings:
            print(f"  - {warning}")
    print()
    print("Checks")
    for check in verification.get("checks", []):
        print(f"  {_status(check.get('ok')):<6} {check.get('name')}: {check.get('message')}")
        for failure in check.get("failures", [])[:3]:
            print(f"         {failure}")
    print()
    print("Generated SQL")
    print(f"  {sql_info.get('query')}")
    print(f"  params = {sql_info.get('params', [])}")
    print()
    print("Sample Rows")
    for row in payload.get("rows", [])[:5]:
        print(
            "  "
            f"{row.get('recipe_id')}: "
            f"{row.get('recipe_title')} | "
            f"calories={row.get('calories')} | "
            f"vegan={row.get('vegan')} | "
            f"region={row.get('region')}"
        )


def _format_nutrition_filters(filters: list[dict[str, Any]]) -> str:
    parts = []
    for item in filters:
        if not isinstance(item, dict):
            continue
        source = item.get("table_source") or item.get("relation") or item.get("source") or "nutrition"
        parts.append(
            f"{source}.{item.get('nutrient')} {item.get('operator')} {item.get('value')}"
        )
    return ", ".join(parts) if parts else "[]"


def _status(ok: Any) -> str:
    return "PASS" if bool(ok) else "FAIL"


def parse_params(raw: str | None) -> list[Any]:
    if not raw:
        return []
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError("--params must be a JSON list")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run temp RecipeDB SQL tests against a local SQLite fixture."
    )
    parser.add_argument("query", nargs="*", help="Natural-language query to compile and execute.")
    parser.add_argument("--sql", help="Run raw SQL directly instead of compiling natural language.")
    parser.add_argument("--params", help="JSON list of params for --sql.")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite DB path.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild and reseed the temp DB first.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum rows to print.")
    parser.add_argument("--schema", action="store_true", help="Print actual temp table columns and row counts.")
    parser.add_argument("--brief", action="store_true", help="Print short rows and column summary instead of full rows.")
    parser.add_argument("--explain", action="store_true", help="Print a readable PASS/FAIL verification summary.")
    parser.add_argument("--count-only", action="store_true", help="Print only the total SQL result count.")
    args = parser.parse_args()

    db_path = args.db
    if args.rebuild or not db_path.exists():
        if db_path != DB_PATH:
            raise ValueError("--rebuild currently writes the default fixture path only")
        db_path = build_database()

    if args.schema:
        payload = {
            "db": str(db_path),
            "schema": schema_snapshot(db_path),
        }
        print_result(payload)
        return

    if args.sql:
        sql = args.sql
        params = parse_params(args.params)
        if args.count_only:
            print_result({
                "db": str(db_path),
                "sql": sql,
                "params": params,
                "total_count": count_for_sql(db_path, sql, params),
            })
            return
        columns, rows = rows_for_sql(db_path, sql, params, args.limit)
        payload = {
            "db": str(db_path),
            "sql": sql,
            "params": params,
            "columns": brief_columns(columns) if args.brief else columns,
            "count": len(rows),
            "rows": brief_rows(rows) if args.brief else rows,
        }
        print_result(payload)
        return

    natural_query = " ".join(args.query).strip()
    if not natural_query:
        parser.error("provide a natural-language query or --sql")

    generated = generated_sql(natural_query)
    compiled = generated["sql"]
    sql = compiled.get("query")
    if not sql:
        print_result({
            "db": str(db_path),
            "query": natural_query,
            "intent": generated["intent"],
            "sql": compiled,
            "count": 0,
            "rows": [],
        })
        return

    if args.count_only:
        total_count = count_for_sql(db_path, sql, compiled.get("params", []))
        print_result({
            "db": str(db_path),
            "query": natural_query,
            "intent": generated["intent"],
            "sql": {
                "query": sql,
                "params": compiled.get("params", []),
                "validation": compiled.get("validation", {}),
            },
            "total_count": total_count,
        })
        return

    params = compiled.get("params", [])
    total_count = count_for_sql(db_path, sql, params)
    columns, rows = rows_for_sql(db_path, sql, params, args.limit)
    verification = verify_generated_result(
        db_path,
        generated.get("verifier_intent", generated["intent"]),
        compiled,
        rows,
        total_count,
    )
    payload = {
        "db": str(db_path),
        "query": natural_query,
        "intent": generated["intent"],
        "sql": {
            "query": sql,
            "params": params,
            "validation": compiled.get("validation", {}),
        },
        "verification": verification,
        "columns": brief_columns(columns) if args.brief else columns,
        "count": len(rows),
        "total_count": total_count,
        "rows": brief_rows(rows) if args.brief else rows,
    }
    if args.explain:
        print_explanation(payload)
    else:
        print_result(payload)


if __name__ == "__main__":
    main()
