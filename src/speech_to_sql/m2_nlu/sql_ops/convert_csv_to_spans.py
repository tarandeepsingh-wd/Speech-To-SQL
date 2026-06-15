"""
Converts wikisql_labeled CSVs → BIO-ready JSONL for training.
Works for BOTH the old generic CSV and the new food/nutrition CSV.

Usage:
    # New food CSV (primary - use all)
    python -m speech_to_sql.m2_nlu.sql_ops.convert_csv_to_spans \
        --input wikisql_food_nutrition_final_v3.csv \
        --output src/speech_to_sql/m2_nlu/datasets/processed/food_spans.jsonl \
        --max-rows 999999

    # Old generic CSV (supplement - limit to 3000)
    python -m speech_to_sql.m2_nlu.sql_ops.convert_csv_to_spans \
        --input wikisql_labeled.csv \
        --output src/speech_to_sql/m2_nlu/datasets/processed/generic_spans.jsonl \
        --max-rows 3000
"""

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd

BASE_DIR     = Path(__file__).parents[1]
LEXICON_PATH = BASE_DIR / "data" / "sql_operator_lexicon.json"


# Map CSV operator tokens → BIO label types
# Used for cross-validation (confirm what we find matches what should be there)
CSV_LABEL_TO_BIO_TYPE = {
    "COUNT": "AGG",   "SUM": "AGG",   "AVG": "AGG",
    "MAX": "AGG",     "MIN": "AGG",
    ">": "COMPARE",   "<": "COMPARE", "=": "COMPARE",
    ">=": "COMPARE",  "<=": "COMPARE","<>": "COMPARE",
    "BETWEEN": "COMPARE", "LIKE": "COMPARE",
    "AND": "LOGIC",   "OR": "LOGIC",  "NOT": "LOGIC",
    "WHERE": "CLAUSE","SELECT": "CLAUSE","FROM": "CLAUSE",
    "ORDER BY": "CLAUSE","GROUP BY": "CLAUSE","HAVING": "CLAUSE",
    "LIMIT": "CLAUSE","DISTINCT": "CLAUSE",
    "IS NULL": "CLAUSE","IS NOT NULL": "CLAUSE",
    "IN": "SET",      "NOT IN": "SET",
}


def _load_lexicon() -> dict:
    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_spans(text: str, lexicon: dict) -> list:
    """Find operator spans in natural language text using the lexicon."""
    spans = []
    lowered = text.lower()

    # Multi-word terms first (longest match wins)
    terms = sorted(
        lexicon["terms"].items(),
        key=lambda x: len(x[0]),
        reverse=True,
    )
    for term, payload in terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        for m in re.finditer(pattern, lowered):
            spans.append((m.start(), m.end(), payload["type"]))

    # Symbol operators
    for symbol, payload in lexicon["symbols"].items():
        for m in re.finditer(re.escape(symbol), text):
            spans.append((m.start(), m.end(), payload["type"]))

    # Resolve overlaps: prefer longer spans
    spans = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
    accepted, used = [], set()
    for start, end, label in spans:
        chars = set(range(start, end))
        if chars & used:
            continue
        accepted.append({"start": start, "end": end, "type": label})
        used |= chars

    return sorted(accepted, key=lambda s: s["start"])


def convert(
    csv_path: str,
    output_path: str,
    max_rows: int = 999_999,
):
    lexicon = _load_lexicon()
    df = pd.read_csv(csv_path)
    if max_rows < len(df):
        # Sample evenly across the dataset to preserve operator diversity
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    total, kept, no_span, mismatch = 0, 0, 0, 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            total += 1
            question = str(row.get("question", "")).strip()
            if not question:
                continue

            try:
                csv_labels = set(ast.literal_eval(str(row["labels"])))
            except Exception:
                csv_labels = set()

            spans = _find_spans(question, lexicon)

            if not spans:
                no_span += 1
                continue

            # Cross-validate: found span types must overlap with expected types
            found_types    = {s["type"] for s in spans}
            expected_types = {
                CSV_LABEL_TO_BIO_TYPE[lbl]
                for lbl in csv_labels
                if lbl in CSV_LABEL_TO_BIO_TYPE
            }

            if expected_types and not (found_types & expected_types):
                mismatch += 1
                continue

            kept += 1
            f.write(json.dumps({"text": question, "spans": spans}) + "\n")

    print(f"\n{'─'*50}")
    print(f"  File    : {Path(csv_path).name}")
    print(f"  Total   : {total:,}")
    print(f"  Kept    : {kept:,}  ({100*kept//max(total,1)}%)")
    print(f"  No span : {no_span:,}")
    print(f"  Mismatch: {mismatch:,}")
    print(f"  Output  : {output_path}")
    print(f"{'─'*50}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--max-rows", type=int, default=999_999)
    args = parser.parse_args()
    convert(args.input, args.output, args.max_rows)


if __name__ == "__main__":
    main()