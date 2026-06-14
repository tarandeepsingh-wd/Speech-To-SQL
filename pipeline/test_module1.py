"""
Module 1 test suite — run from pipeline/ directory:
    python test_module1.py
"""
import sys
sys.path.insert(0, ".")

from module1_asr.entity_tagger import tag_entities

# ── Test cases ────────────────────────────────────────────
# Each entry: (query, expected_ingredients, expected_dietary, expected_utensils)
TESTS = [
    # Basic English ingredients
    ("show me recipes with chicken and garlic",
     ["chicken", "garlic"], [], []),

    # Hindi ingredients
    ("mujhe adrak aur haldi wali recipe chahiye",
     ["ginger", "turmeric"], [], []),

    # Mixed Hindi-English
    ("chicken curry with adrak, lehsun and lal mirch",
     ["chicken", "ginger", "garlic", "red chili pepper"], [], []),

    # Dietary style
    ("vegan recipes with spinach and lemon juice",
     ["spinach", "lemon juice"], ["vegan"], []),

    # Utensil
    ("recipes that use a skillet with olive oil and garlic",
     ["olive oil", "garlic"], [], ["skillet"]),

    # Nutrition filter
    ("high protein recipes under 500 calories with chicken",
     ["chicken"], [], []),

    # Common English words that should NOT be ingredients
    ("give me some good recipes",
     [], [], []),

    # ASR-style noisy input
    ("lemonata chicken masala",
     ["chicken", "masala"], [], []),

    # Multi-ingredient
    ("biryani with basmati rice, saffron, onion and whole spices",
     ["biryani", "basmati rice", "saffron", "onion"], [], []),

    # Dirty DB / user typo normalization
    ("recipes with tomatoe chicken and milk",
     ["tomato", "chicken", "milk"], [], []),

    # CSV-backed morphology, not a hand-written synonym
    ("recipes with grapes",
     ["grape"], [], []),

    # Should NOT match random words
    ("tarandeep singh wadhwa honolulu words",
     [], [], []),
]

# ── Runner ────────────────────────────────────────────────
passed = 0
failed = 0

for query, exp_ing, exp_diet, exp_uten in TESTS:
    result = tag_entities(query)
    got_ing   = result["ingredients"]
    got_diet  = result["dietary_styles"]
    got_uten  = result["utensils"]

    # Check: all expected ingredients present
    ing_ok   = all(e in got_ing for e in exp_ing)
    diet_ok  = all(e in got_diet for e in exp_diet)
    uten_ok  = all(e in got_uten for e in exp_uten)

    # Check: no obvious false positives (got_ing shouldn't have
    # things that are clearly NOT food if expected is empty)
    fp_ok = True
    if exp_ing == [] and got_ing:
        fp_ok = False

    ok = ing_ok and diet_ok and uten_ok and fp_ok

    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1

    print(f"[{status}] {query[:55]:<55}")
    if not ok:
        if not ing_ok:
            print(f"       Ingredients — expected: {exp_ing}")
            print(f"                    got:      {got_ing}")
        if not diet_ok:
            print(f"       Dietary    — expected: {exp_diet}, got: {got_diet}")
        if not uten_ok:
            print(f"       Utensils   — expected: {exp_uten}, got: {got_uten}")
        if not fp_ok:
            print(f"       FALSE POSITIVES: {got_ing}")

print(f"\n{'='*60}")
print(f"Results: {passed}/{len(TESTS)} passed, {failed} failed")
print(f"{'='*60}")

extra_failed = 0

vitamin_query = "show me recipes that have vitamin B12 and vitamin C"
vitamin_result = tag_entities(vitamin_query)
got_nutrients = {
    item.get("nutrient")
    for item in vitamin_result["nutrition_filters"]
    if item.get("nutrient")
}
expected_nutrients = {"vitamin_b_12", "vitamin_c"}
if expected_nutrients.issubset(got_nutrients):
    print("[PASS] vitamin presence filters from schema aliases")
else:
    print("[FAIL] vitamin presence filters from schema aliases")
    print(f"       Expected nutrients: {sorted(expected_nutrients)}")
    print(f"       Got nutrients:      {sorted(got_nutrients)}")
    extra_failed += 1

nutrient_presence_cases = [
    (
        "show me recipes that contain vitamin k and chicken",
        {"vitamin_k"},
        {"chicken"},
    ),
    (
        "show recipes that contain thiamine",
        {"thiamin"},
        set(),
    ),
    (
        "show recipes with cholesterol",
        {"cholesterol"},
        set(),
    ),
    (
        "show recipes that have magnesium",
        {"magnesium"},
        set(),
    ),
    (
        "recipes rich in folate",
        {"folate_total"},
        set(),
    ),
    (
        "recipes with sugar",
        set(),
        {"sugar"},
    ),
]

for query, expected_nutrients, expected_ingredients in nutrient_presence_cases:
    result = tag_entities(query)
    got_nutrients = {
        item.get("nutrient")
        for item in result["nutrition_filters"]
        if item.get("nutrient")
    }
    got_ingredients = set(result["ingredients"])
    nutrient_ok = expected_nutrients.issubset(got_nutrients)
    ingredient_ok = expected_ingredients.issubset(got_ingredients)
    if expected_nutrients:
        ingredient_ok = ingredient_ok and not (expected_nutrients & got_ingredients)
    if not expected_nutrients:
        nutrient_ok = not got_nutrients
    ok = nutrient_ok and ingredient_ok
    print(f"[{'PASS' if ok else 'FAIL'}] schema nutrition presence: {query[:45]:<45}")
    if not ok:
        extra_failed += 1
        print(f"       Expected nutrients:   {sorted(expected_nutrients)}")
        print(f"       Got nutrients:        {sorted(got_nutrients)}")
        print(f"       Expected ingredients: {sorted(expected_ingredients)}")
        print(f"       Got ingredients:      {sorted(got_ingredients)}")

if extra_failed:
    sys.exit(1)
