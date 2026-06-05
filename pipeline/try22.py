from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


PROJECT_PIPELINE = Path(__file__).resolve().parent
if str(PROJECT_PIPELINE) not in sys.path:
    sys.path.insert(0, str(PROJECT_PIPELINE))


def main() -> None:
    args = sys.argv[1:]
    verbose = "--verbose" in args or "-v" in args
    args = [arg for arg in args if arg not in {"--verbose", "-v"}]

    if args:
        run_query(" ".join(args).strip(), verbose=verbose)
        return

    print("Speech-to-SQL tester")
    print("Use --verbose for full Module 1 / QueryIR / plan JSON.")
    print("Type a natural-language recipe query. Press Enter on blank input to exit.")
    while True:
        try:
            text = input("\nNLP> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text or text.lower() in {"exit", "quit", "q"}:
            return
        run_query(text, verbose=verbose)


def run_query(text: str, *, verbose: bool = False) -> None:
    if not text:
        print("No query entered.")
        return

    try:
        from module1_asr.entity_tagger import tag_entities
        from module3.sql_query_builder import build_sql_query
    except Exception as exc:
        print_section("IMPORT ERROR")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    try:
        start = time.perf_counter()
        module1_output = tag_entities(text)
        module1_seconds = time.perf_counter() - start
    except Exception as exc:
        print_section("MODULE 1 ERROR")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    print("\nINPUT:", text)
    print("MODULE 1:", _module1_summary(module1_output), f"({module1_seconds:.2f}s)")

    try:
        start = time.perf_counter()
        sql_result = build_sql_query(module1_output, transcript=text)
        module3_seconds = time.perf_counter() - start
    except Exception as exc:
        print_section("MODULE 3 ERROR")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    query = sql_result.get("query")
    print("SQL:", query or "None", f"({module3_seconds:.2f}s)")
    print("PARAMS:", json.dumps(sql_result.get("params", []), ensure_ascii=False, default=str))
    print("VALID:", bool((sql_result.get("validation") or {}).get("ok")))

    errors = (sql_result.get("validation") or {}).get("errors") or []
    if errors:
        print("ERRORS:", json.dumps(errors, ensure_ascii=False, default=str))

    if verbose:
        print_section("FULL MODULE 1 OUTPUT")
        print_json(module1_output)
        print_section("FULL MODULE 3 OUTPUT")
        print_json(sql_result)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _module1_summary(output: dict[str, Any]) -> str:
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
