import ast
import csv
import json
from pathlib import Path


TEXT_FIELDS = ("text", "query", "question", "utterance", "sentence")


def canonical_label_type_map(lexicon: dict) -> dict[str, str]:
    label_types = {}
    for payload in lexicon.get("symbols", {}).values():
        label_types[payload["canonical"]] = payload["type"]
    for payload in lexicon.get("terms", {}).values():
        label_types[payload["canonical"]] = payload["type"]
    return label_types


def normalize_sql_label(label: str) -> str | None:
    label = str(label).strip().strip("\"'")
    if not label:
        return None
    if label in {"<", ">", "<=", ">=", "=", "!=", "<>"}:
        return "!=" if label == "<>" else label
    return " ".join(label.upper().split())


def parse_label_cell(value: str | None) -> list[str]:
    if not value or not str(value).strip():
        return []

    raw = str(value).strip()
    parsed = None
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            parsed = None

    if parsed is None:
        parsed = raw.split(",")

    labels = []
    for item in parsed:
        label = normalize_sql_label(str(item))
        if label and label not in labels:
            labels.append(label)
    return labels


def text_from_row(row: dict) -> str | None:
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def labels_from_row(row: dict) -> list[str]:
    labels = parse_label_cell(row.get("labels"))
    if labels:
        return labels
    return parse_label_cell(row.get("label_string"))


def read_labeled_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = text_from_row(row)
            labels = labels_from_row(row)
            if text and labels:
                rows.append({"text": text, "labels": labels})
    return rows


def save_label_list(labels: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


def load_label_list(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
