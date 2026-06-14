import io
import os
import wave
import numpy as np
import sounddevice as sd

from module1_asr.transcriber import transcribe_audio
from module1_asr.entity_tagger import tag_entities
from module1_asr.feedback_logger import log_ingredient_feedback, log_sql_clause_feedback
from module1_asr.sql_ops.label_utils import normalize_sql_label
from module1_asr.sql_ops import score_sql_clause_labels
from module3.sql_query_builder import build_sql_query


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
    Converts a float32 numpy array → in-memory WAV bytes.
    Whisper's transcribe() needs a file path or file-like WAV.
    We write to a BytesIO buffer so no temp file is needed here
    (transcriber.py writes its own temp file from these bytes).
    """
    # Convert float32 [-1, 1] → int16 for standard WAV
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
    text     = transcribe_audio(audio_bytes)
    entities = tag_entities(text)
    return {"transcript": text, **entities}


def process_text(text: str) -> dict:
    """Text-only path — skips ASR. Useful for quick testing."""
    entities = tag_entities(text)
    return {"transcript": text, **entities}


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
    if os.getenv("TADKA_SQL_CLAUSE_DEBUG") == "1":
        print(f"  SQL clause scores: {_format_sql_clause_scores(result['transcript'])}")
    print(f"  SQL rule ops     : {_format_sql_operators(explicit_sql_ops)}")
    print(f"  SQL filter plan  : {result.get('sql_filter_plan', {})}")
    print(f"  Unified SQL plan : {result.get('unified_sql_plan', {})}")


def print_module3_sql(result: dict) -> None:
    sql_result = build_sql_query(result, transcript=result.get("transcript"))
    validation = sql_result.get("validation") or {}
    plan = sql_result.get("plan") or {}

    print("\n  Module 3 SQL")
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
    return f"{item.get('nutrient')} {item.get('operator')} {item.get('value')}"


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


# ─────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────

def main() -> None:
    os.environ.setdefault("TADKA_ENABLE_SQL_CLAUSE_ML", "1")

    print("=" * 55)
    print("  TADKA - ASR + Entity Tagger + SQL Generator")
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
            audio_np  = record_until_enter()
            if audio_np.size == 0:
                print("No audio captured. Try again.")
                continue
            wav_bytes = numpy_to_wav_bytes(audio_np)
            print("\nTranscribing with Whisper...")
            result    = process_audio(wav_bytes)
            print_result(result)
            print_module3_sql(result)
            collect_ingredient_feedback(result)
            collect_sql_clause_feedback(result)

        elif choice == "2":
            text   = read_terminal_input("Type your query: ").strip()
            if not text:
                print("No text entered.")
                continue
            result = process_text(text)
            print_result(result)
            print_module3_sql(result)
            collect_ingredient_feedback(result)
            collect_sql_clause_feedback(result)

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
