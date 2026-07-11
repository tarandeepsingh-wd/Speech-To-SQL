from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paste-ready AWS Athena SQL from a natural language recipe query."
    )
    parser.add_argument("query", nargs="+", help="Natural language recipe query.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also print the normalized entities, parameterized SQL, and params.",
    )
    args = parser.parse_args()

    text = " ".join(args.query).strip()
    if not text:
        raise SystemExit("No query entered.")

    from speech_to_sql.m2_nlu.entity_tagger import tag_entities
    from speech_to_sql.m4_sql.athena_renderer import render_athena_sql
    from speech_to_sql.m4_sql.sql_query_builder import build_sql_query

    nlu_output = tag_entities(text)
    sql_result = build_sql_query(nlu_output, transcript=text)
    athena_result = render_athena_sql(sql_result)

    if args.debug:
        print("INPUT:", text)
        print("NLU:", json.dumps(_debug_nlu(nlu_output), ensure_ascii=False, default=str))
        print("PARAMETERIZED_SQL:", sql_result.get("query") or "None")
        print("PARAMS:", json.dumps(sql_result.get("params", []), ensure_ascii=False, default=str))
        print("VALID:", bool((sql_result.get("validation") or {}).get("ok")))
        print()

    print(athena_result.get("query") or "None")


def _debug_nlu(output: dict) -> dict:
    return {
        "ingredients": output.get("ingredients", []),
        "ingredient_logic": output.get("ingredient_logic", {}),
        "dietary_styles": output.get("dietary_styles", []),
        "categories": output.get("categories", []),
        "nutrition_filters": output.get("nutrition_filters", []),
        "time_filters": output.get("time_filters", []),
    }


if __name__ == "__main__":
    main()
