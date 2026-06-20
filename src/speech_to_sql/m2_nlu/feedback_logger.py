import json
import os
from datetime import datetime, timezone
from pathlib import Path


FEEDBACK_DIR = Path(__file__).parent / "data" / "feedback"
SQL_FEEDBACK_FILE = FEEDBACK_DIR / "sql_operator_feedback.jsonl"
SQL_CLAUSE_FEEDBACK_FILE = FEEDBACK_DIR / "sql_clause_feedback.jsonl"
INGREDIENT_FEEDBACK_FILE = FEEDBACK_DIR / "ingredient_feedback.jsonl"
TRANSCRIPT_FEEDBACK_FILE = FEEDBACK_DIR / "transcript_feedback.jsonl"
TRANSCRIPT_FEEDBACK_FILE_ENV = "SPEECH_TO_SQL_TRANSCRIPT_FEEDBACK_FILE"


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **row,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def log_sql_operator_feedback(
    query: str,
    predicted: list[dict],
    corrected_spans: list[dict],
    source: str = "manual",
) -> None:
    """
    Store a corrected SQL-operator example for future NER training.

    corrected_spans format:
      [{"start": 10, "end": 22, "type": "COMPARE"}]
    End is exclusive, matching the training script.
    """
    _append_jsonl(SQL_FEEDBACK_FILE, {
        "source": source,
        "query": query,
        "predicted": predicted,
        "text": query,
        "spans": corrected_spans,
    })


def log_sql_clause_feedback(
    query: str,
    predicted: list[dict],
    corrected_labels: list[str],
    source: str = "manual",
) -> None:
    """
    Store corrected whole-query SQL labels for the clause classifier.

    corrected_labels format:
      ["SELECT", "WHERE", "COUNT", "="]
    """
    _append_jsonl(SQL_CLAUSE_FEEDBACK_FILE, {
        "source": source,
        "query": query,
        "predicted": predicted,
        "text": query,
        "labels": corrected_labels,
    })


def log_ingredient_feedback(
    query: str,
    predicted: list[str],
    corrected_ingredients: list[str],
    source: str = "manual",
) -> None:
    _append_jsonl(INGREDIENT_FEEDBACK_FILE, {
        "source": source,
        "query": query,
        "predicted": predicted,
        "corrected_ingredients": corrected_ingredients,
    })


def log_transcript_feedback(
    original_query: str,
    corrected_query: str,
    predicted_ingredients: list[str] | None = None,
    corrected_ingredients: list[str] | None = None,
    corrections: list[dict] | None = None,
    source: str = "manual",
) -> None:
    """
    Store a user-approved transcript correction.

    The adaptive lexicon consumes these rows at runtime. This keeps the
    correction layer data-driven: the pipeline learns from explicit user
    corrections instead of adding new hardcoded spelling maps.
    """
    target_file = Path(os.getenv(TRANSCRIPT_FEEDBACK_FILE_ENV) or TRANSCRIPT_FEEDBACK_FILE)
    _append_jsonl(target_file, {
        "source": source,
        "original_query": original_query,
        "corrected_query": corrected_query,
        "predicted_ingredients": predicted_ingredients or [],
        "corrected_ingredients": corrected_ingredients or [],
        "corrections": corrections or [],
        "approved": True,
    })
