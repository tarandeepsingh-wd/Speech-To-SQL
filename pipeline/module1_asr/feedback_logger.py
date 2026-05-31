import json
from datetime import datetime, timezone
from pathlib import Path


FEEDBACK_DIR = Path(__file__).parent / "data" / "feedback"
SQL_FEEDBACK_FILE = FEEDBACK_DIR / "sql_operator_feedback.jsonl"
SQL_CLAUSE_FEEDBACK_FILE = FEEDBACK_DIR / "sql_clause_feedback.jsonl"
INGREDIENT_FEEDBACK_FILE = FEEDBACK_DIR / "ingredient_feedback.jsonl"


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
