from functools import lru_cache
import os
from pathlib import Path

import torch
from transformers.utils import logging as transformers_logging


MODEL_DIR = Path(__file__).parents[1] / "models" / "sql_operator_ner"
MIN_CONFIDENCE = 0.60


@lru_cache(maxsize=1)
def _load_pipeline():
    if not MODEL_DIR.exists():
        return None

    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
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
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def _merge_bio_tokens(text: str, labels: list[tuple[str, tuple[int, int], float]]) -> list[dict]:
    spans = []
    current = None

    for label, offset, confidence in labels:
        if label == "O" or offset[0] == offset[1]:
            if current:
                spans.append(current)
                current = None
            continue

        prefix, entity_type = label.split("-", 1)
        if prefix == "B" or not current or current["type"] != entity_type:
            if current:
                spans.append(current)
            current = {
                "type": entity_type,
                "start": offset[0],
                "end": offset[1],
                "scores": [confidence],
            }
        else:
            current["end"] = offset[1]
            current["scores"].append(confidence)

    if current:
        spans.append(current)

    return [
        {
            "text": text[item["start"]:item["end"]],
            "type": item["type"],
            "span": [item["start"], item["end"] - 1],
            "confidence": round(sum(item["scores"]) / len(item["scores"]), 3),
        }
        for item in spans
        if item["scores"] and (sum(item["scores"]) / len(item["scores"])) >= MIN_CONFIDENCE
    ]


def extract_sql_operator_ner_spans(text: str) -> list[dict]:
    loaded = _load_pipeline()
    if not loaded or not text.strip():
        return []

    tokenizer, model = loaded
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=128,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        logits = model(**encoded).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        scores, predictions = torch.max(probabilities, dim=-1)

    id2label = model.config.id2label
    labels = [
        (id2label[int(predictions[i])], tuple(offsets[i]), float(scores[i]))
        for i in range(len(offsets))
    ]
    return _merge_bio_tokens(text, labels)
