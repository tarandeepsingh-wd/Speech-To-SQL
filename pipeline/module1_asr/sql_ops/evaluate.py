"""
Evaluates the full SQL operator extraction pipeline.
Tests both the rule system and the ML model separately.

Run from pipeline/ directory:
    python -m module1_asr.sql_ops.evaluate
"""

import os
from module1_asr.sql_ops.extractor import extract_sql_operators

# ── Test cases ────────────────────────────────────────────────────────────────
# Format: (query, expected_types_that_must_be_found)
# We check that every expected type appears in the output.
# We don't penalize for extra types found.

TEST_CASES = [

    # ── COUNT / AGG ───────────────────────────────────────────────────────────
    ("how many vegan recipes have fiber more than 5 grams",
     ["AGG", "COMPARE"]),

    ("what is the average protein in salad recipes",
     ["AGG"]),

    ("total number of recipes with less than 200 calories",
     ["AGG", "COMPARE"]),

    ("what is the maximum iron content in breakfast dishes",
     ["AGG"]),

    ("find the minimum prep time for gluten free recipes",
     ["AGG"]),

    # ── COMPARE ───────────────────────────────────────────────────────────────
    ("show recipes where calories are under 400",
     ["COMPARE"]),

    ("recipes with protein greater than 25 grams",
     ["COMPARE"]),

    ("dishes where fat is between 5 and 15 grams",
     ["COMPARE"]),

    ("recipes with rating above 4",
     ["COMPARE"]),

    ("find meals where carbs are at most 30 grams",
     ["COMPARE"]),

    ("list dishes where sodium is not more than 500 milligrams",
     ["COMPARE"]),

    # ── CLAUSE ────────────────────────────────────────────────────────────────
    ("list recipes sorted by calories from lowest to highest",
     ["CLAUSE"]),

    ("show dishes grouped by cuisine type",
     ["CLAUSE"]),

    ("find recipes ordered by preparation time descending",
     ["CLAUSE"]),

    ("show top 5 recipes by protein content",
     ["CLAUSE"]),

    # ── LOGIC ─────────────────────────────────────────────────────────────────
    ("recipes with chicken and garlic but without onion",
     ["LOGIC"]),

    ("show vegan or vegetarian recipes under 300 calories",
     ["LOGIC", "COMPARE"]),

    # ── COMBINED ─────────────────────────────────────────────────────────────
    ("how many recipes have protein greater than 30 and calories less than 500",
     ["AGG", "COMPARE", "LOGIC"]),

    ("show recipes where fat is under 10 sorted by protein descending",
     ["COMPARE", "CLAUSE"]),

    ("find dishes grouped by meal type where fiber is more than 8 grams",
     ["CLAUSE", "COMPARE"]),

    # ── SHOULD RETURN EMPTY (no operator) ────────────────────────────────────
    ("show me something healthy",          []),
    ("I want a quick dinner",              []),
    ("give me a good recipe",              []),
]


def run_evaluation(force_ml: bool = False):
    if force_ml:
        os.environ["TADKA_ENABLE_SQL_ML"]    = "1"
        os.environ["TADKA_FORCE_SQL_ML"]     = "1"
        label = "ML MODEL"
    else:
        os.environ.pop("TADKA_ENABLE_SQL_ML", None)
        os.environ.pop("TADKA_FORCE_SQL_ML",  None)
        label = "RULE SYSTEM"

    print(f"\n{'═'*60}")
    print(f"  EVALUATING: {label}")
    print(f"{'═'*60}\n")

    passed   = 0
    failed   = 0
    failures = []

    for query, expected_types in TEST_CASES:
        ops          = extract_sql_operators(query)
        found_types  = {o["type"] for o in ops}

        if not expected_types:
            # Should return empty
            if not ops:
                passed += 1
                print(f"  ✅ PASS  | {query[:55]}")
            else:
                failed += 1
                false_positives = [o["text"] for o in ops]
                failures.append((query, expected_types, found_types))
                print(f"  ❌ FAIL  | {query[:55]}")
                print(f"           | Expected: nothing")
                print(f"           | Got:      {false_positives}")
        else:
            missing = [t for t in expected_types if t not in found_types]
            if not missing:
                passed += 1
                found_texts = [(o["type"], o["text"]) for o in ops]
                print(f"  ✅ PASS  | {query[:55]}")
                print(f"           | Found: {found_texts}")
            else:
                failed += 1
                failures.append((query, expected_types, found_types))
                print(f"  ❌ FAIL  | {query[:55]}")
                print(f"           | Missing types: {missing}")
                print(f"           | Found types:   {list(found_types)}")
        print()

    total    = passed + failed
    accuracy = 100 * passed // total

    print(f"{'─'*60}")
    print(f"  RESULT: {passed}/{total} passed  ({accuracy}%)")
    print(f"{'─'*60}\n")

    return passed, failed


if __name__ == "__main__":
    # Run 1: Rule system only
    rule_pass, rule_fail = run_evaluation(force_ml=False)

    # Run 2: ML model forced on
    ml_pass, ml_fail = run_evaluation(force_ml=True)

    # Compare
    print(f"\n{'═'*60}")
    print(f"  COMPARISON")
    print(f"{'═'*60}")
    print(f"  Rule system : {rule_pass}/{rule_pass+rule_fail}")
    print(f"  ML model    : {ml_pass}/{ml_pass+ml_fail}")
    winner = "Rule system" if rule_pass >= ml_pass else "ML model"
    print(f"  Better      : {winner}")
    print(f"{'═'*60}\n")