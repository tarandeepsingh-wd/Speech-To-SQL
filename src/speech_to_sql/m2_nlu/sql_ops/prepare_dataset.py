import argparse
import csv
import json
from pathlib import Path

from speech_to_sql.m2_nlu.sql_ops.train import LEXICON_PATH, _label_text, _load_json


TEXT_FIELDS = ("text", "query", "question", "utterance", "sentence")


def _read_rows(path: Path) -> list[dict]:
    if path.is_dir():
        rows = []
        for child in sorted(path.rglob("*")):
            if child.suffix.lower() in {".json", ".jsonl"}:
                rows.extend(_read_rows(child))
        return rows

    if path.suffix.lower() == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "train", "validation", "examples"):
                if isinstance(payload.get(key), list):
                    return payload[key]
            return [payload]

    if path.suffix.lower() == ".csv":
        rows = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                spans = row.get("spans")
                if spans and spans.strip():
                    row["spans"] = json.loads(spans)
                rows.append(row)
        return rows

    raise ValueError(f"Unsupported dataset file: {path}")


def _text_from_row(row: dict) -> str | None:
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def convert_dataset(input_path: Path, output_path: Path, keep_empty: bool = False) -> int:
    lexicon = _load_json(LEXICON_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for row in _read_rows(input_path):
            text = _text_from_row(row)
            if not text:
                continue

            if isinstance(row.get("spans"), list):
                example = {"text": text, "spans": row["spans"]}
            else:
                example = _label_text(text, lexicon)

            if not keep_empty and not example["spans"]:
                continue

            f.write(json.dumps(example, ensure_ascii=True) + "\n")
            count += 1

    return count


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert manual/Spider/WikiSQL-style JSON into SQL NER JSONL."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep-empty", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    written = convert_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        keep_empty=args.keep_empty,
    )
    print(f"wrote {written} examples to {args.output}")
