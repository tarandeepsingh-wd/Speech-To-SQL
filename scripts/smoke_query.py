from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    args = sys.argv[1:]
    verbose = "--verbose" in args or "-v" in args
    athena = "--athena" in args
    athena_only = "--athena-only" in args
    args = [
        arg
        for arg in args
        if arg not in {"--verbose", "-v", "--athena", "--athena-only"}
    ]

    if args:
        run_query(
            " ".join(args).strip(),
            verbose=verbose,
            athena=athena or athena_only,
            athena_only=athena_only,
        )
        return

    print("Speech-to-SQL tester")
    print("Use --verbose for full NLU / QueryIR / SQL plan JSON.")
    print("Use --athena to also print paste-ready AWS Athena SQL.")
    print("Type a natural-language recipe query. Press Enter on blank input to exit.")
    while True:
        try:
            text = input("\nNLP> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text or text.lower() in {"exit", "quit", "q"}:
            return
        run_query(text, verbose=verbose, athena=athena)


def run_query(
    text: str,
    *,
    verbose: bool = False,
    athena: bool = False,
    athena_only: bool = False,
) -> None:
    if not text:
        print("No query entered.")
        return

    try:
        from speech_to_sql.m2_nlu.entity_tagger import tag_entities
        from speech_to_sql.m4_sql.athena_renderer import render_athena_sql
        from speech_to_sql.m4_sql.sql_query_builder import build_sql_query
    except Exception as exc:
        print_section("IMPORT ERROR")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    try:
        start = time.perf_counter()
        nlu_output = tag_entities(text)
        nlu_seconds = time.perf_counter() - start
    except Exception as exc:
        print_section("NLU ERROR")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    if not athena_only:
        print("\nINPUT:", text)
        print("NLU:", _nlu_summary(nlu_output), f"({nlu_seconds:.2f}s)")

    try:
        start = time.perf_counter()
        sql_result = build_sql_query(nlu_output, transcript=text)
        sql_seconds = time.perf_counter() - start
    except Exception as exc:
        print_section("SQL ERROR")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    if athena_only:
        print(render_athena_sql(sql_result).get("query") or "None")
        return

    query = sql_result.get("query")
    print("SQL:", query or "None", f"({sql_seconds:.2f}s)")
    print("PARAMS:", json.dumps(sql_result.get("params", []), ensure_ascii=False, default=str))
    if athena:
        print("ATHENA_SQL:", render_athena_sql(sql_result).get("query") or "None")
    print("VALID:", bool((sql_result.get("validation") or {}).get("ok")))

    errors = (sql_result.get("validation") or {}).get("errors") or []
    if errors:
        print("ERRORS:", json.dumps(errors, ensure_ascii=False, default=str))

    if verbose:
        print_section("FULL NLU OUTPUT")
        print_json(nlu_output)
        print_section("FULL SQL OUTPUT")
        print_json(sql_result)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _nlu_summary(output: dict[str, Any]) -> str:
    summary = {
        "ingredients": output.get("ingredients", []),
        "diet": output.get("dietary_styles", []),
        "categories": output.get("categories", []),
        "nutrition": output.get("nutrition_filters", []),
        "sql_ops": [
            item.get("canonical")
            for item in output.get("sql_operators", [])
            if isinstance(item, dict)
        ],
    }
    return json.dumps(summary, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
