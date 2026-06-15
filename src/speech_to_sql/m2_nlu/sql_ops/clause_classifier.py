from functools import lru_cache
import json
import os
from pathlib import Path

import torch
from transformers.utils import logging as transformers_logging

from speech_to_sql.m2_nlu.sql_ops.label_utils import canonical_label_type_map, load_label_list
from speech_to_sql.m2_nlu.sql_ops.train import LEXICON_PATH, _load_json


MODEL_DIR = Path(__file__).parents[1] / "models" / "sql_clause_classifier"
LABELS_FILE = MODEL_DIR / "labels.json"
META_FILE = MODEL_DIR / "training_meta.json"
DEFAULT_THRESHOLD = 0.5


def _fallback_label_type(label: str) -> str:
    if label in {"SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY", "HAVING", "LIMIT"}:
        return "CLAUSE"
    if label in {"AND", "OR", "NOT"}:
        return "LOGIC"
    if label in {"COUNT", "SUM", "AVG", "MIN", "MAX", "DISTINCT"}:
        return "AGG"
    if label in {"IN", "NOT IN", "IS NULL", "IS NOT NULL"}:
        return "SET"
    return "COMPARE"


def _load_threshold() -> float:
    env_value = os.getenv("SPEECH_TO_SQL_CLAUSE_THRESHOLD")
    if env_value:
        try:
            return float(env_value)
        except ValueError:
            pass
    if META_FILE.exists():
        with open(META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return float(meta.get("threshold", DEFAULT_THRESHOLD))
    return DEFAULT_THRESHOLD


@lru_cache(maxsize=1)
def _load_classifier():
    if not MODEL_DIR.exists() or not LABELS_FILE.exists():
        return None

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return None

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    transformers_logging.set_verbosity_error()
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model.eval()
    labels = load_label_list(LABELS_FILE)
    label_types = canonical_label_type_map(_load_json(LEXICON_PATH))
    return tokenizer, model, labels, label_types, _load_threshold()


def predict_sql_clause_labels(text: str, threshold: float | None = None) -> list[dict]:
    loaded = _load_classifier()
    if not loaded or not text.strip():
        return []

    _tokenizer, _model, _labels, _label_types, saved_threshold = loaded
    threshold = saved_threshold if threshold is None else threshold
    return [
        prediction for prediction in score_sql_clause_labels(text)
        if prediction["confidence"] >= threshold
    ]


def score_sql_clause_labels(text: str) -> list[dict]:
    loaded = _load_classifier()
    if not loaded or not text.strip():
        return []

    tokenizer, model, labels, label_types, _saved_threshold = loaded
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits[0]
        scores = torch.sigmoid(logits).tolist()

    scored = []
    for label, score in zip(labels, scores):
        scored.append({
            "canonical": label,
            "type": label_types.get(label, _fallback_label_type(label)),
            "confidence": round(float(score), 3),
        })
    return scored
