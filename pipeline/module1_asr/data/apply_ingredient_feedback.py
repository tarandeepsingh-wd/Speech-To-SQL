"""
Suggest or apply ingredient synonym additions from logged feedback.

Usage:
    python -m module1_asr.data.apply_ingredient_feedback --dry-run
    python -m module1_asr.data.apply_ingredient_feedback --apply
"""

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path


FEEDBACK_FILE = Path(__file__).parent / "feedback" / "ingredient_feedback.jsonl"
SYNONYMS_FILE = Path(__file__).parent / "ingredient_synonyms.py"
MIN_OCCURRENCES = 3

STOPWORDS = {
    "show", "find", "give", "recipe", "recipes", "with", "without", "and",
    "or", "the", "for", "some", "good", "mujhe", "wali", "chahiye",
    "please", "want", "need", "make",
}


def load_feedback() -> list[dict]:
    if not FEEDBACK_FILE.exists():
        return []

    rows = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _synonyms_assignment():
    content = SYNONYMS_FILE.read_text(encoding="utf-8")
    tree = ast.parse(content)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SYNONYMS":
                    return content, node
    raise RuntimeError("Could not find SYNONYMS assignment.")


def load_existing_synonyms() -> set[str]:
    _, node = _synonyms_assignment()
    return {
        ast.literal_eval(key).lower()
        for key in node.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _query_candidates(query: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", query.lower())
    candidates = []
    for width in (3, 2, 1):
        for i in range(0, len(tokens) - width + 1):
            phrase = " ".join(tokens[i:i + width]).replace("-", " ")
            if phrase in STOPWORDS:
                continue
            if any(part in STOPWORDS for part in phrase.split()):
                continue
            candidates.append(phrase)
    return candidates


def extract_correction_candidates(rows: list[dict]) -> list[tuple[str, str, int]]:
    existing_synonyms = load_existing_synonyms()
    counter = Counter()

    for row in rows:
        query = row.get("query", "")
        predicted = {item.lower() for item in row.get("predicted", [])}
        corrected = {item.lower() for item in row.get("corrected_ingredients", [])}
        missed = corrected - predicted
        if not missed:
            continue

        for token in _query_candidates(query):
            if token in existing_synonyms or token in predicted or token in corrected:
                continue
            for ingredient in missed:
                counter[(token, ingredient)] += 1

    return [
        (token, canonical, count)
        for (token, canonical), count in counter.most_common()
        if count >= MIN_OCCURRENCES
    ]


def apply_candidates(candidates: list[tuple[str, str, int]]) -> None:
    content, node = _synonyms_assignment()
    lines = content.splitlines(keepends=True)
    insert_at = node.end_lineno - 1

    section = ["\n    # Auto-generated from ingredient feedback\n"]
    for token, canonical, count in candidates:
        section.append(f'    "{token}": "{canonical}",  # seen {count}x\n')

    updated = "".join(lines[:insert_at] + section + lines[insert_at:])
    SYNONYMS_FILE.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print suggestions only")
    parser.add_argument("--apply", action="store_true", help="Append suggestions to SYNONYMS")
    args = parser.parse_args()

    if args.apply == args.dry_run:
        parser.error("Choose exactly one of --dry-run or --apply.")

    rows = load_feedback()
    if not rows:
        print("No feedback found.")
        return

    candidates = extract_correction_candidates(rows)
    if not candidates:
        print(f"No candidates with >= {MIN_OCCURRENCES} occurrences.")
        return

    print(f"Suggested new synonyms ({len(candidates)}):")
    for token, canonical, count in candidates:
        print(f'    "{token}": "{canonical}",  # seen {count}x')

    if args.apply:
        apply_candidates(candidates)
        print(f"Added {len(candidates)} entries to ingredient_synonyms.py")


if __name__ == "__main__":
    main()
