import io
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any


os.environ.setdefault("TQDM_DISABLE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
TEMP_DB_DIR = PROJECT_ROOT / "temp_recipe_db"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(TEMP_DB_DIR) not in sys.path:
    sys.path.insert(0, str(TEMP_DB_DIR))

import numpy as np

from speech_to_sql.m2_nlu.entity_tagger import tag_entities
from speech_to_sql.m2_nlu.feedback_logger import log_ingredient_feedback, log_sql_clause_feedback
from speech_to_sql.m2_nlu.sql_ops.label_utils import normalize_sql_label
from speech_to_sql.m2_nlu.sql_ops import score_sql_clause_labels
from speech_to_sql.m3_query_ir.intent_preparer import prepare_intent
from speech_to_sql.m3_query_ir.sql_intent_builder import build_query_ir
from speech_to_sql.m4_sql.sql_query_builder import build_sql_query
from speech_to_sql.m5_execution.query_planner import plan_queries


SAMPLE_RATE = 16000
CHANNELS    = 1


def read_terminal_input(prompt: str, default: str = "") -> str:
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        return default


def record_until_enter() -> np.ndarray:
    """
    Records audio from the default microphone until the user
    presses Enter. Returns a numpy array of float32 samples.
    """
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - depends on local audio setup
        raise RuntimeError(
            "Microphone mode needs the sounddevice package and an accessible input device."
        ) from exc

    print("\n Recording... Press ENTER to stop.\n")

    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate = SAMPLE_RATE,
        channels   = CHANNELS,
        dtype      = "float32",
        callback   = callback,
    ):
        input()   # blocks until user presses Enter

    if not frames:
        return np.array([], dtype="float32")

    return np.concatenate(frames, axis=0)


def numpy_to_wav_bytes(audio: np.ndarray) -> bytes:
    """
    Converts a float32 numpy array to in-memory WAV bytes.
    Whisper's transcribe() needs a file path or file-like WAV.
    We write to a BytesIO buffer so no temp file is needed here
    (transcriber.py writes its own temp file from these bytes).
    """
    # Convert float32 [-1, 1] to int16 for standard WAV.
    pcm = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)          # 2 bytes = int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    return buf.getvalue()


def process_audio(audio_bytes: bytes) -> dict:
    """Full pipeline: WAV bytes -> transcript -> structured entities."""
    from speech_to_sql.m1_speech.transcriber import transcribe_audio

    timings: dict[str, float] = {}
    start = time.perf_counter()
    text = transcribe_audio(audio_bytes)
    timings["m1_speech"] = time.perf_counter() - start

    start = time.perf_counter()
    entities = tag_entities(text)
    timings["m2_nlu"] = time.perf_counter() - start
    return {
        "transcript": text,
        "_input_mode": "speech",
        "_timings": timings,
        **entities,
    }


def process_text(text: str) -> dict:
    """Text-only path; skips ASR. Useful for quick testing."""
    timings: dict[str, float] = {"m1_speech": 0.0}
    start = time.perf_counter()
    entities = tag_entities(text)
    timings["m2_nlu"] = time.perf_counter() - start
    return {
        "transcript": text,
        "_input_mode": "text",
        "_timings": timings,
        **entities,
    }


def _format_sql_operators(sql_operators: list[dict]) -> str:
    if not sql_operators:
        return "[]"

    parts = []
    for op in sql_operators:
        confidence = op.get("confidence")
        confidence_text = f"{confidence:.2f}" if isinstance(confidence, float) else str(confidence)
        layer = op.get("layer", "")
        text = op.get("text") or "implicit"
        parts.append(f"{op['canonical']} ({layer}, {confidence_text}, {text})")
    return "[" + "; ".join(parts) + "]"


def _format_sql_clause_model(sql_operators: list[dict]) -> str:
    clause_predictions = [
        op for op in sql_operators
        if op.get("layer") == "ml_clause"
    ]
    if not clause_predictions:
        return "[]"

    parts = []
    for op in clause_predictions:
        confidence = op.get("confidence")
        confidence_text = f"{confidence:.2f}" if isinstance(confidence, float) else str(confidence)
        parts.append(f"{op['canonical']} ({confidence_text})")
    return "[" + "; ".join(parts) + "]"


def _format_sql_clause_scores(text: str, top_n: int = 12) -> str:
    scores = sorted(
        score_sql_clause_labels(text),
        key=lambda item: item["confidence"],
        reverse=True,
    )
    if not scores:
        return "[]"

    return "[" + "; ".join(
        f"{item['canonical']} ({item['confidence']:.3f})"
        for item in scores[:top_n]
    ) + "]"


def print_result(result: dict):
    sql_operators = result.get("sql_operators", [])
    explicit_sql_ops = [
        op for op in sql_operators
        if op.get("layer") != "ml_clause"
    ]

    print(f"\n  Transcript       : {result['transcript']}")
    print(f"  Ingredients      : {result['ingredients']}")
    print(f"  Ingredient logic : {result.get('ingredient_logic', {})}")
    print(f"  Utensils         : {result['utensils']}")
    state_actions = [a['action'] for a in result['actions'] if a['type'] == 'state_change']
    print(f"  Cooking actions  : {state_actions}")
    print(f"  Dietary styles   : {result['dietary_styles']}")
    print(f"  Categories       : {[c['category'] for c in result['categories']]}")
    print(f"  Nutrition filters: {result['nutrition_filters']}")
    print(f"  SQL clause model : {_format_sql_clause_model(sql_operators)}")
    if os.getenv("SPEECH_TO_SQL_CLAUSE_DEBUG") == "1":
        print(f"  SQL clause scores: {_format_sql_clause_scores(result['transcript'])}")
    print(f"  SQL rule ops     : {_format_sql_operators(explicit_sql_ops)}")
    print(f"  SQL filter plan  : {result.get('sql_filter_plan', {})}")
    print(f"  Unified SQL plan : {result.get('unified_sql_plan', {})}")


def print_sql_result(result: dict) -> None:
    sql_result = build_sql_query(result, transcript=result.get("transcript"))
    validation = sql_result.get("validation") or {}
    plan = sql_result.get("plan") or {}

    print("\n  SQL")
    print(f"  Meaning         : {_plain_meaning(result)}")
    if plan:
        print(f"  Root table      : {plan.get('from_table')} AS {plan.get('from_alias')}")
        print(f"  Strategy        : {_sql_strategy(plan, sql_result.get('query'))}")
    print(f"  SQL             : {sql_result.get('query')}")
    print(f"  Params          : {sql_result.get('params', [])}")
    print(f"  Valid           : {validation.get('ok')}")
    if validation.get("warnings"):
        print(f"  Warnings        : {validation['warnings']}")
    if validation.get("errors"):
        print(f"  Errors          : {validation['errors']}")


def print_pipeline_report(result: dict) -> None:
    report = build_pipeline_report(result)
    timings = report.get("timings", {})

    _print_banner("Speech-to-SQL Full Pipeline")
    _print_kv("Input mode", report["input_mode"])
    _print_kv("Transcript", report["transcript"])

    _print_section("M1 Speech")
    if report["input_mode"] == "text":
        _print_kv("Status", "SKIP - typed text was used")
    else:
        _print_kv("Status", "PASS")
    _print_kv("Time", _seconds(timings.get("m1_speech")))

    _print_section("M2 NLU")
    _print_kv("Status", "PASS")
    _print_kv("Time", _seconds(timings.get("m2_nlu")))
    _print_kv("Ingredients", _json_inline(result.get("ingredients", [])))
    _print_kv("Ingredient logic", _ingredient_logic_summary(result.get("ingredient_logic") or {}))
    _print_kv("Utensils", _json_inline(result.get("utensils", [])))
    _print_kv("Actions", _format_actions(result.get("actions", [])))
    _print_kv("Dietary styles", _json_inline(result.get("dietary_styles", [])))
    _print_kv("Categories", _format_categories(result.get("categories", [])))
    _print_kv("Nutrition", _format_nutrition_filters(result.get("nutrition_filters", [])))
    _print_kv("SQL operators", _format_sql_operators(result.get("sql_operators", [])))
    _print_spans(result.get("raw_spans", []))

    _print_section("M3 QueryIR")
    m3_error = report.get("m3_error")
    _print_kv("Status", "FAIL" if m3_error else "PASS")
    _print_kv("Time", _seconds(timings.get("m3_query_ir")))
    if m3_error:
        _print_kv("Error", m3_error)
    else:
        _print_prepared_intent(report.get("prepared_intent") or {})
        _print_query_ir(report.get("query_ir") or {})

    _print_section("M4 SQL")
    sql_result = report.get("sql") or {}
    sql_validation = sql_result.get("validation") or {}
    _print_kv("Status", _status_text(sql_validation.get("ok")))
    _print_kv("Time", _seconds(timings.get("m4_sql")))
    if report.get("m4_error"):
        _print_kv("Error", report["m4_error"])
    else:
        _print_kv("Valid", str(bool(sql_validation.get("ok"))))
        _print_kv("SQL", sql_result.get("query"))
        _print_kv("Params", _json_inline(sql_result.get("params", [])))
        _print_sql_plan(sql_result.get("plan") or {})
        if sql_validation.get("warnings"):
            _print_kv("Warnings", _json_inline(sql_validation["warnings"]))
        if sql_validation.get("errors"):
            _print_kv("Errors", _json_inline(sql_validation["errors"]))

    _print_section("M5 Execution / Verification")
    _print_api_plans(report.get("api_plans", []))
    _print_dummy_db(report.get("dummy_db") or {})

    _print_section("Overall")
    _print_kv("Pipeline", _overall_status(report))
    print()


def build_pipeline_report(result: dict) -> dict[str, Any]:
    transcript = result.get("transcript", "")
    timings = dict(result.get("_timings") or {})
    report: dict[str, Any] = {
        "input_mode": result.get("_input_mode", "text"),
        "transcript": transcript,
        "timings": timings,
    }

    prepared_intent = None
    query_ir = None

    try:
        start = time.perf_counter()
        prepared_intent = prepare_intent(
            _public_tag_result(result),
            transcript=transcript,
        )
        query_ir = build_query_ir(prepared_intent, transcript=transcript)
        report["prepared_intent"] = prepared_intent
        report["query_ir"] = query_ir.as_dict()
        timings["m3_query_ir"] = time.perf_counter() - start
    except Exception as exc:
        timings["m3_query_ir"] = time.perf_counter() - start if "start" in locals() else 0.0
        report["m3_error"] = f"{type(exc).__name__}: {exc}"

    try:
        start = time.perf_counter()
        source_intent = prepared_intent or _public_tag_result(result)
        report["api_plans"] = [
            plan.as_dict()
            for plan in plan_queries(source_intent, transcript=transcript)
        ]
    except Exception as exc:
        report["api_plan_error"] = f"{type(exc).__name__}: {exc}"

    try:
        start = time.perf_counter()
        source_intent = prepared_intent or _public_tag_result(result)
        report["sql"] = build_sql_query(
            source_intent,
            transcript=transcript,
            query_ir=query_ir,
        )
        timings["m4_sql"] = time.perf_counter() - start
    except Exception as exc:
        timings["m4_sql"] = time.perf_counter() - start if "start" in locals() else 0.0
        report["m4_error"] = f"{type(exc).__name__}: {exc}"
        report["sql"] = {"validation": {"ok": False, "errors": [report["m4_error"]]}}

    try:
        start = time.perf_counter()
        report["dummy_db"] = run_dummy_db_verification(
            prepared_intent or result,
            report.get("sql") or {},
        )
        timings["m5_dummy_db"] = time.perf_counter() - start
    except Exception as exc:
        timings["m5_dummy_db"] = time.perf_counter() - start if "start" in locals() else 0.0
        report["dummy_db"] = {
            "ok": False,
            "skipped": True,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    return report


def run_dummy_db_verification(intent: dict, sql_result: dict) -> dict[str, Any]:
    query = sql_result.get("query")
    if not query:
        return {
            "ok": False,
            "skipped": True,
            "reason": "No SQL query was generated.",
        }

    from run_query import (
        DB_PATH,
        brief_rows,
        build_database,
        count_for_sql,
        rows_for_sql,
        verify_generated_result,
    )

    db_path = DB_PATH if DB_PATH.exists() else build_database()
    params = list(sql_result.get("params") or [])
    columns, rows = rows_for_sql(db_path, query, params, limit=10)
    total_count = count_for_sql(db_path, query, params)
    verifier_intent = _intent_for_verifier(intent)
    verification = verify_generated_result(
        db_path,
        verifier_intent,
        sql_result,
        rows,
        total_count,
    )
    return {
        "ok": bool(verification.get("ok")),
        "skipped": False,
        "db": str(db_path),
        "columns": columns,
        "rows": brief_rows(rows),
        "total_count": total_count,
        "verification": verification,
    }


def _public_tag_result(result: dict) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if not str(key).startswith("_")
    }


def _intent_for_verifier(intent: dict) -> dict[str, Any]:
    return {
        "ingredients": intent.get("ingredients", []),
        "dietary_styles": intent.get("dietary_styles", []),
        "utensils": intent.get("utensils", []),
        "actions": intent.get("actions", []),
        "category_filters": intent.get("category_filters", []),
        "nutrition_filters": intent.get("nutrition_filters", []),
        "ingredient_logic": intent.get("ingredient_logic", {}),
    }


def _print_banner(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _print_section(title: str) -> None:
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def _print_kv(label: str, value: Any) -> None:
    print(f"  {label:<18}: {value if value is not None else ''}")


def _json_inline(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=str)


def _seconds(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return "-"


def _status_text(ok: Any) -> str:
    return "PASS" if bool(ok) else "FAIL"


def _ingredient_logic_summary(logic: dict) -> str:
    include = logic.get("include", [])
    exclude = logic.get("exclude", [])
    include_op = logic.get("include_operator", "AND")
    exclude_op = logic.get("exclude_operator", "AND")
    ambiguous = bool(logic.get("ambiguous"))
    return (
        f"include={include} ({include_op}), "
        f"exclude={exclude} ({exclude_op}), ambiguous={ambiguous}"
    )


def _format_actions(actions: list[dict]) -> str:
    if not actions:
        return "[]"
    compact = [
        {
            "action": item.get("action"),
            "type": item.get("type"),
        }
        for item in actions[:12]
    ]
    return _json_inline(compact)


def _format_categories(categories: list[dict]) -> str:
    if not categories:
        return "[]"
    compact = [
        {
            "term": item.get("term"),
            "category": item.get("category"),
        }
        for item in categories[:12]
    ]
    return _json_inline(compact)


def _format_nutrition_filters(filters: list[dict]) -> str:
    if not filters:
        return "[]"
    compact = []
    for item in filters[:12]:
        if not isinstance(item, dict):
            continue
        compact.append({
            "nutrient": item.get("nutrient"),
            "operator": item.get("operator"),
            "value": item.get("value"),
            "group": item.get("nutrient_group"),
            "table_source": item.get("table_source"),
        })
    return _json_inline(compact)


def _print_spans(spans: list[dict]) -> None:
    if not spans:
        _print_kv("Raw spans", "[]")
        return
    _print_kv("Raw spans", f"{len(spans)} span(s)")
    for span in spans[:8]:
        text = span.get("text")
        canonical = span.get("canonical")
        layer = span.get("layer")
        source = span.get("source")
        confidence = span.get("reranker_confidence", span.get("score", ""))
        print(f"    - {text!r} -> {canonical!r} | layer={layer} | source={source} | conf={confidence}")


def _print_prepared_intent(intent: dict) -> None:
    _print_kv("Prepared", str(bool(intent.get("_intent_prepared"))))
    _print_kv("Schema links", _json_inline(intent.get("_schema_links", [])))
    _print_kv("Enrichments", _json_inline(intent.get("_intent_enrichments", [])))
    _print_kv("Category filters", _json_inline(intent.get("category_filters", [])))


def _print_query_ir(query_ir: dict) -> None:
    _print_kv("Root", query_ir.get("root"))
    _print_kv("Select", _json_inline(query_ir.get("select", [])))
    predicates = query_ir.get("predicates", [])
    _print_kv("Predicates", f"{len(predicates)} predicate(s)")
    for predicate in predicates:
        print(
            "    - "
            f"{predicate.get('source')}.{predicate.get('field')} "
            f"{predicate.get('operator')} "
            f"{predicate.get('values')} "
            f"({predicate.get('quantifier')})"
        )
    _print_kv("Order by", _json_inline(query_ir.get("order_by", [])))
    _print_kv("Limit", query_ir.get("limit"))


def _print_sql_plan(plan: dict) -> None:
    if not plan:
        return
    _print_kv("Root table", f"{plan.get('from_table')} AS {plan.get('from_alias')}")
    _print_kv("Strategy", _sql_strategy(plan, "SELECT DISTINCT" if plan.get("distinct") else ""))
    joins = plan.get("joins") or []
    _print_kv("Joins", f"{len(joins)} join(s)")
    for join in joins[:6]:
        print(
            "    - "
            f"{join.get('join_type')} {join.get('table')} AS {join.get('alias')} "
            f"ON {join.get('on')} [{join.get('source')}]"
        )
    where = plan.get("where") or []
    _print_kv("Where", f"{len(where)} condition(s)")
    for condition in where[:8]:
        print(f"    - {condition.get('sql')} | params={condition.get('params')}")
    assumptions = plan.get("join_assumptions") or []
    if assumptions:
        _print_kv("Join assumptions", f"{len(assumptions)}")
        for item in assumptions[:5]:
            print(
                "    - "
                f"{item.get('name')} | confidence={item.get('confidence')}"
            )


def _print_api_plans(plans: list[dict]) -> None:
    _print_kv("RecipeDB API", "planned only; no external call is made")
    if not plans:
        _print_kv("API plans", "[]")
        return
    _print_kv("API plans", f"{len(plans)} plan(s)")
    for plan in plans[:5]:
        print(
            "    - "
            f"{plan.get('role')} {plan.get('method')} {plan.get('path')} "
            f"label={plan.get('label')} params={plan.get('params')}"
        )


def _print_dummy_db(dummy: dict) -> None:
    if dummy.get("skipped"):
        _print_kv("Dummy DB", f"SKIP - {dummy.get('reason')}")
        return
    verification = dummy.get("verification") or {}
    _print_kv("Dummy DB", dummy.get("db"))
    _print_kv("SQL execution", _status_text(dummy.get("ok")))
    _print_kv("Rows matched", dummy.get("total_count"))
    _print_kv("Rows checked", verification.get("checked_rows"))
    semantic_conflicts = verification.get("semantic_conflicts") or []
    if semantic_conflicts:
        _print_kv("Semantic conflicts", f"{len(semantic_conflicts)} warning(s)")
        for warning in semantic_conflicts[:5]:
            print(f"    - {warning}")
    for check in verification.get("checks", []):
        print(
            "    - "
            f"{_status_text(check.get('ok')):<4} "
            f"{check.get('name')}: {check.get('message')}"
        )
        for failure in check.get("failures", [])[:2]:
            print(f"      failure: {failure}")
    rows = dummy.get("rows") or []
    if rows:
        _print_kv("Sample rows", f"{len(rows[:5])} shown")
        for row in rows[:5]:
            print(
                "    - "
                f"{row.get('Recipe_id', row.get('recipe_id'))}: "
                f"{row.get('recipe_title', row.get('ingredient', ''))} | "
                f"calories={row.get('calories')} | "
                f"vegan={row.get('vegan')} | "
                f"region={row.get('region')}"
            )
    if dummy.get("total_count") == 0 and semantic_conflicts:
        print("    Note: zero rows are expected for this contradictory intent.")
    elif not dummy.get("ok") and dummy.get("total_count") == 0:
        print("    Note: zero rows can mean the tiny dummy DB lacks that example, not necessarily that the SQL is invalid.")


def _overall_status(report: dict) -> str:
    sql_ok = bool(((report.get("sql") or {}).get("validation") or {}).get("ok"))
    dummy = report.get("dummy_db") or {}
    dummy_ok = bool(dummy.get("ok")) if not dummy.get("skipped") else False
    verification = dummy.get("verification") or {}
    has_expected_empty_conflict = (
        dummy_ok
        and dummy.get("total_count") == 0
        and bool(verification.get("semantic_conflicts"))
    )
    if report.get("m3_error") or report.get("m4_error"):
        return "FAIL - see module errors above"
    if sql_ok and has_expected_empty_conflict:
        return "PASS - SQL is valid; no dummy rows matched because a semantic conflict was detected"
    if sql_ok and dummy_ok:
        return "PASS - NLU, QueryIR, SQL validation, and dummy DB verification passed"
    if sql_ok:
        return "PARTIAL - SQL is valid, but dummy DB verification did not fully pass"
    return "FAIL - SQL validation failed"


def _plain_meaning(result: dict) -> str:
    parts = []
    logic = result.get("ingredient_logic") or {}
    include = logic.get("include") or result.get("ingredients") or []
    exclude = logic.get("exclude") or []
    if include:
        connector = " OR " if logic.get("include_operator") == "OR" else " AND "
        parts.append("contains " + connector.join(include))
    if exclude:
        parts.append("excludes " + ", ".join(exclude))
    if result.get("dietary_styles"):
        parts.append("diet " + ", ".join(result["dietary_styles"]))
    if result.get("categories"):
        parts.append("category " + ", ".join(
            str(item.get("term") or item.get("category"))
            for item in result["categories"]
        ))
    if result.get("nutrition_filters"):
        parts.append("nutrition " + ", ".join(
            _nutrition_filter_meaning(item)
            for item in result["nutrition_filters"]
        ))
    return "; ".join(parts) if parts else "no filters"


def _nutrition_filter_meaning(item: dict) -> str:
    if item.get("nutrient_group"):
        return f"group {item.get('nutrient_group')} present"
    source = item.get("table_source") or "nutrition"
    return f"{source}.{item.get('nutrient')} {item.get('operator')} {item.get('value')}"


def _sql_strategy(plan: dict, query: str | None) -> str:
    joins = plan.get("joins") or []
    ingredient_joins = [
        join for join in joins
        if join.get("source") == "ingredients"
    ]
    if len(ingredient_joins) > 1 and query and "SELECT DISTINCT" in query and "ingredient_lower" in query:
        return "filtered ingredient AND JOINs"
    if len(ingredient_joins) > 1 and query and "ingredient_lower" in query:
        return "ingredient AND JOINs"
    if query and "GROUP BY" in query and ("COUNT(DISTINCT" in query or "COUNT(*)" in query):
        return "grouped ingredient JOIN"
    if joins:
        return "JOIN"
    if query and "EXISTS" in query:
        return "EXISTS"
    return "base table filter"


def collect_ingredient_feedback(result: dict) -> None:
    answer = read_terminal_input(
        "\nAre the ingredients correct? [Enter=skip, y=yes, n=no]: "
    ).strip().lower()
    if answer not in {"n", "no"}:
        return

    corrected = read_terminal_input("Correct ingredients, comma-separated (empty if none): ").strip()
    corrected_ingredients = []
    for raw_item in corrected.split(","):
        item = raw_item.strip().lower()
        if item and item not in corrected_ingredients:
            corrected_ingredients.append(item)

    log_ingredient_feedback(
        query=result["transcript"],
        predicted=result.get("ingredients", []),
        corrected_ingredients=corrected_ingredients,
        source="main.py",
    )
    print("Saved ingredient feedback for the next reranker training run.")


def collect_sql_clause_feedback(result: dict) -> None:
    clause_predictions = [
        op for op in result.get("sql_operators", [])
        if op.get("layer") == "ml_clause"
    ]
    if not clause_predictions:
        return

    answer = read_terminal_input(
        "\nAre the SQL labels correct? [Enter=skip, y=yes, n=no]: "
    ).strip().lower()
    if answer not in {"n", "no"}:
        return

    corrected = read_terminal_input("Correct SQL labels, comma-separated: ").strip()
    corrected_labels = []
    for raw_label in corrected.split(","):
        label = normalize_sql_label(raw_label)
        if label and label not in corrected_labels:
            corrected_labels.append(label)

    if not corrected_labels:
        print("No feedback saved; no labels were entered.")
        return

    log_sql_clause_feedback(
        query=result["transcript"],
        predicted=clause_predictions,
        corrected_labels=corrected_labels,
        source="main.py",
    )
    print("Saved SQL-label feedback for the next training run.")


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------

def main() -> None:
    os.environ.setdefault("SPEECH_TO_SQL_ENABLE_SQL_CLAUSE_ML", "1")

    print("=" * 55)
    print("  Speech-to-SQL - Full Pipeline CLI")
    print("=" * 55)

    while True:
        print("\nOptions:")
        print("  [1] Speak a query (microphone)")
        print("  [2] Type a query  (text)")
        print("  [q] Quit")
        choice = read_terminal_input("\nChoice: ", "q").strip().lower()

        if choice == "q":
            print("Bye!")
            break

        elif choice == "1":
            try:
                audio_np  = record_until_enter()
                if audio_np.size == 0:
                    print("No audio captured. Try again.")
                    continue
                wav_bytes = numpy_to_wav_bytes(audio_np)
                print("\nTranscribing with Whisper...")
                result = process_audio(wav_bytes)
                print_pipeline_report(result)
                collect_ingredient_feedback(result)
                collect_sql_clause_feedback(result)
            except Exception as exc:
                print(f"\nPipeline failed: {type(exc).__name__}: {exc}")

        elif choice == "2":
            text   = read_terminal_input("Type your query: ").strip()
            if not text:
                print("No text entered.")
                continue
            try:
                result = process_text(text)
                print_pipeline_report(result)
                collect_ingredient_feedback(result)
                collect_sql_clause_feedback(result)
            except Exception as exc:
                print(f"\nPipeline failed: {type(exc).__name__}: {exc}")

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
