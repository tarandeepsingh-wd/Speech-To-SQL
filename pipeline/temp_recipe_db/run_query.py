from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from seed_temp_db import DB_PATH, build_database


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


BRIEF_COLUMNS = [
    "Recipe_id",
    "recipe_id",
    "recipe_title",
    "ingredient",
    "ingredient_lower",
    "matching_values",
    "calories",
    "protein (g)",
    "carbohydrate, by difference (g)",
    "thiamin (mg)",
    "vegan",
    "region",
]


def generated_sql(query: str) -> dict[str, Any]:
    from module1_asr.entity_tagger import tag_entities
    from module3.sql_query_builder import build_sql_query

    tags = tag_entities(query)
    tags["transcript"] = query
    compiled = build_sql_query(tags, transcript=query)
    return {
        "intent": {
            "ingredients": tags.get("ingredients", []),
            "dietary_styles": tags.get("dietary_styles", []),
            "utensils": tags.get("utensils", []),
            "actions": tags.get("actions", []),
            "category_filters": tags.get("category_filters", []),
            "nutrition_filters": tags.get("nutrition_filters", []),
            "ingredient_logic": tags.get("ingredient_logic", {}),
        },
        "sql": compiled,
    }


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
    parser.add_argument("--count-only", action="store_true", help="Print only the total SQL result count.")
    args = parser.parse_args()

    db_path = args.db
    if args.rebuild or not db_path.exists():
        if db_path != DB_PATH:
            raise ValueError("--rebuild currently writes the default fixture path only")
        db_path = build_database()

    if args.schema:
        print_result({
            "db": str(db_path),
            "schema": schema_snapshot(db_path),
        })
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
        print_result({
            "db": str(db_path),
            "sql": sql,
            "params": params,
            "columns": brief_columns(columns) if args.brief else columns,
            "count": len(rows),
            "rows": brief_rows(rows) if args.brief else rows,
        })
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
        print_result({
            "db": str(db_path),
            "query": natural_query,
            "intent": generated["intent"],
            "sql": {
                "query": sql,
                "params": compiled.get("params", []),
                "validation": compiled.get("validation", {}),
            },
            "total_count": count_for_sql(db_path, sql, compiled.get("params", [])),
        })
        return

    columns, rows = rows_for_sql(db_path, sql, compiled.get("params", []), args.limit)
    print_result({
        "db": str(db_path),
        "query": natural_query,
        "intent": generated["intent"],
        "sql": {
            "query": sql,
            "params": compiled.get("params", []),
            "validation": compiled.get("validation", {}),
        },
        "columns": brief_columns(columns) if args.brief else columns,
        "count": len(rows),
        "rows": brief_rows(rows) if args.brief else rows,
    })


if __name__ == "__main__":
    main()
