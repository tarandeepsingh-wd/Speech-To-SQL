"""
Generated Module 1 stress suite.

This intentionally creates broad combinations instead of hand-picking cases.
Run from pipeline/:
    python test_module1_stress.py
    python test_module1_stress.py --verbose
"""
import argparse
import sys
import time

sys.path.insert(0, ".")

from module1_asr.entity_tagger import tag_entities


EN_INGREDIENTS = [
    "chicken", "garlic", "onion", "tomato", "spinach", "salt",
    "butter", "sugar", "milk", "egg", "carrot", "potato",
    "ginger", "turmeric", "cumin", "coriander", "cinnamon",
    "pepper", "flour", "rice", "lemon", "cream", "yogurt",
    "cheese", "mushroom", "salmon", "tuna", "beef", "lamb",
    "shrimp", "basil", "thyme", "oregano", "paprika", "saffron",
    "vanilla", "honey", "vinegar", "pasta", "paneer", "olive oil",
    "soy sauce", "ghee", "chickpea flour", "lentil", "kidney bean",
    "green chilies", "red chili pepper", "cardamom", "clove",
]

HI_MAP = {
    "adrak": "ginger",
    "haldi": "turmeric",
    "jeera": "cumin",
    "lehsun": "garlic",
    "pyaz": "onion",
    "tamatar": "tomato",
    "namak": "salt",
    "dahi": "yogurt",
    "ghee": "ghee",
    "aloo": "potato",
    "gajar": "carrot",
    "palak": "spinach",
    "dhania": "coriander",
    "makhan": "butter",
    "cheeni": "sugar",
    "murgh": "chicken",
    "chawal": "rice",
    "nimbu": "lemon",
    "paneer": "paneer",
    "besan": "chickpea flour",
    "hing": "asafoetida",
    "elaichi": "cardamom",
    "dalchini": "cinnamon",
}

ASR_MAP = {
    "chick": "chicken",
    "chiken": "chicken",
    "beaf": "beef",
    "garlc": "garlic",
    "onoin": "onion",
    "tomatoe": "tomato",
    "shrmp": "shrimp",
    "bazil": "basil",
    "vegn": "spinach",
    "lehsoon": "garlic",
    "lesson": "garlic",
    "tamater": "tomato",
}

DIETS = [
    "vegan", "vegetarian", "gluten-free", "dairy-free",
    "keto", "halal", "kosher", "paleo", "pescatarian",
]

UTENSILS = ["skillet", "wok", "oven", "pan", "pot", "blender", "saucepan", "mixer"]

FALSE_POSITIVES = [
    "show me all recipes",
    "what can I cook today",
    "find me something delicious",
    "I am hungry",
    "help me cook dinner",
    "what should I make for lunch",
    "hello world test query",
    "random words here please",
    "give me options please",
    "I want to explore recipes",
    "what is trending in food",
    "something easy to prepare",
    "recommend a meal",
    "browse all available recipes",
    "just tell me something",
]


def case(query, ingredients=None, operator=None, exclude=None,
         dietary=None, utensils=None, nutrients=None, empty=False):
    return {
        "query": query,
        "ingredients": ingredients or [],
        "operator": operator,
        "exclude": exclude or [],
        "dietary": dietary or [],
        "utensils": utensils or [],
        "nutrients": nutrients or [],
        "empty": empty,
    }


def generate_cases():
    cases = []

    # Implicit AND: no explicit connector, SQL should still require all.
    for i in range(220):
        a = EN_INGREDIENTS[i % len(EN_INGREDIENTS)]
        b = EN_INGREDIENTS[(i + 7) % len(EN_INGREDIENTS)]
        c = EN_INGREDIENTS[(i + 13) % len(EN_INGREDIENTS)]
        cases.append(case(
            f"recipes containing {a} {b} {c}",
            [a, b, c],
            "AND",
        ))

    # Explicit AND in different natural-language shapes.
    and_templates = [
        "recipes with {} and {}",
        "show dishes containing {} plus {}",
        "find meals that have {}, {} and {}",
        "I need food using {} with {} and {}",
    ]
    for i in range(260):
        tmpl = and_templates[i % len(and_templates)]
        a = EN_INGREDIENTS[(i * 2) % len(EN_INGREDIENTS)]
        b = EN_INGREDIENTS[(i * 2 + 5) % len(EN_INGREDIENTS)]
        c = EN_INGREDIENTS[(i * 2 + 11) % len(EN_INGREDIENTS)]
        values = [a, b] if tmpl.count("{}") == 2 else [a, b, c]
        cases.append(case(tmpl.format(*values), values, "AND"))

    # OR intent: useful for SQL IN without the HAVING all-ingredients check.
    or_templates = [
        "recipes with {} or {}",
        "either {} or {} recipes",
        "any of {}, {}, or {} dishes",
        "one of {}, {} or {} meals",
    ]
    for i in range(240):
        tmpl = or_templates[i % len(or_templates)]
        a = EN_INGREDIENTS[(i + 1) % len(EN_INGREDIENTS)]
        b = EN_INGREDIENTS[(i + 9) % len(EN_INGREDIENTS)]
        c = EN_INGREDIENTS[(i + 17) % len(EN_INGREDIENTS)]
        values = [a, b] if tmpl.count("{}") == 2 else [a, b, c]
        cases.append(case(tmpl.format(*values), values, "OR"))

    # Exclusion / NOT intent.
    exclude_templates = [
        "recipes with {} without {}",
        "show {} recipes no {}",
        "find dishes containing {} except {}",
        "no {} with {}",
    ]
    for i in range(220):
        tmpl = exclude_templates[i % len(exclude_templates)]
        include = EN_INGREDIENTS[(i + 3) % len(EN_INGREDIENTS)]
        exclude = EN_INGREDIENTS[(i + 19) % len(EN_INGREDIENTS)]
        if tmpl.startswith("no "):
            query = tmpl.format(exclude, include)
        else:
            query = tmpl.format(include, exclude)
        cases.append(case(query, [include, exclude], "AND", [exclude]))

    # Hindi and Hinglish ingredient logic.
    hi_items = list(HI_MAP.items())
    hi_templates = [
        "{} aur {} wali recipe",
        "mujhe {} {} aur {} chahiye",
        "recipe with {} and {}",
        "{} ke saath {} banana hai",
    ]
    for i in range(260):
        tmpl = hi_templates[i % len(hi_templates)]
        h1, e1 = hi_items[i % len(hi_items)]
        h2, e2 = hi_items[(i + 5) % len(hi_items)]
        h3, e3 = hi_items[(i + 11) % len(hi_items)]
        if tmpl.count("{}") == 2:
            query = tmpl.format(h1, h2)
            expected = [e1, e2]
        else:
            query = tmpl.format(h1, h2, h3)
            expected = [e1, e2, e3]
        cases.append(case(query, expected, "AND"))

    # Diet, utensil, and nutrition mixed with ingredients.
    nutrient_templates = [
        "{} {} recipe with {} under {} calories",
        "{} dishes using a {} with {} over {}g protein",
        "{} food in a {} with {} below {} calories",
    ]
    calories = [150, 200, 250, 300, 350, 400, 450, 500]
    for i in range(260):
        tmpl = nutrient_templates[i % len(nutrient_templates)]
        diet = DIETS[i % len(DIETS)]
        utensil = UTENSILS[(i + 2) % len(UTENSILS)]
        ingredient = EN_INGREDIENTS[(i + 4) % len(EN_INGREDIENTS)]
        value = calories[i % len(calories)]
        nutrient = "protein" if "protein" in tmpl else "calories"
        cases.append(case(
            tmpl.format(diet, utensil, ingredient, value),
            [ingredient],
            "AND",
            dietary=[diet],
            utensils=[utensil],
            nutrients=[nutrient],
        ))

    # ASR-ish aliases and truncations.
    asr_items = list(ASR_MAP.items())
    for i in range(160):
        surface, expected = asr_items[i % len(asr_items)]
        partner = EN_INGREDIENTS[(i + 8) % len(EN_INGREDIENTS)]
        if surface == "vegn":
            cases.append(case(
                f"vegn recipes with {partner}",
                [partner],
                "AND",
                dietary=["vegan"],
            ))
        else:
            cases.append(case(
                f"{surface} and {partner} recipe",
                [expected, partner],
                "AND",
            ))

    # SQL-specific combined intent.
    for i in range(220):
        a = EN_INGREDIENTS[(i + 2) % len(EN_INGREDIENTS)]
        b = EN_INGREDIENTS[(i + 6) % len(EN_INGREDIENTS)]
        c = EN_INGREDIENTS[(i + 10) % len(EN_INGREDIENTS)]
        d = EN_INGREDIENTS[(i + 14) % len(EN_INGREDIENTS)]
        if i % 2 == 0:
            cases.append(case(
                f"give recipes where ingredients are {a} {b} {c}",
                [a, b, c],
                "AND",
            ))
        else:
            cases.append(case(
                f"recipes with either {a} or {b} without {d}",
                [a, b, d],
                "OR",
                [d],
            ))

    # Empty/generic utterances, repeated with suffixes so each query is unique.
    for i in range(120):
        base = FALSE_POSITIVES[i % len(FALSE_POSITIVES)]
        cases.append(case(f"{base} {i}", empty=True))

    return cases


def check_case(test_case):
    result = tag_entities(test_case["query"])
    got_ing = result["ingredients"]
    got_diet = result["dietary_styles"]
    got_uten = result["utensils"]
    got_nutr = [item["nutrient"] for item in result["nutrition_filters"]]
    logic = result["ingredient_logic"]
    sql_plan = result["sql_filter_plan"]

    failures = []

    if test_case["empty"] and got_ing:
        failures.append(f"expected empty ingredients, got {got_ing}")

    for ingredient in test_case["ingredients"]:
        if ingredient not in got_ing:
            failures.append(f"missing ingredient {ingredient!r}; got {got_ing}")

    for dietary in test_case["dietary"]:
        if dietary not in got_diet:
            failures.append(f"missing dietary {dietary!r}; got {got_diet}")

    for utensil in test_case["utensils"]:
        if utensil not in got_uten:
            failures.append(f"missing utensil {utensil!r}; got {got_uten}")

    for nutrient in test_case["nutrients"]:
        if nutrient not in got_nutr:
            failures.append(f"missing nutrient {nutrient!r}; got {got_nutr}")

    expected_operator = test_case["operator"]
    if expected_operator and logic["include_operator"] != expected_operator:
        failures.append(
            f"expected include operator {expected_operator}, "
            f"got {logic['include_operator']} ({logic})"
        )

    for excluded in test_case["exclude"]:
        if excluded not in logic["exclude"]:
            failures.append(f"missing exclude {excluded!r}; got {logic['exclude']}")

    if expected_operator and sql_plan["include_operator"] != expected_operator:
        failures.append(
            f"SQL plan operator mismatch: expected {expected_operator}, "
            f"got {sql_plan['include_operator']}"
        )

    if expected_operator and logic["include"] and not sql_plan["where_clause"]:
        failures.append("SQL plan missing WHERE clause")

    if expected_operator == "AND" and len(logic["include"]) > 1:
        if not sql_plan["params"] or sql_plan["params"][-1] != len(logic["include"]):
            failures.append(f"AND SQL plan missing HAVING count param: {sql_plan}")

    return failures, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cases = generate_cases()
    passed = failed = 0
    failures = []
    t0 = time.time()

    for idx, test_case in enumerate(cases, start=1):
        case_failures, result = check_case(test_case)
        if case_failures:
            failed += 1
            failures.append((idx, test_case, case_failures, result))
            if args.verbose:
                print(f"[FAIL] #{idx} {test_case['query']}")
                for failure in case_failures:
                    print(f"       -> {failure}")
        else:
            passed += 1
            if args.verbose:
                print(f"[PASS] #{idx} {test_case['query']}")

    print(f"Generated cases: {len(cases)}")
    print(f"Passed: {passed}/{len(cases)} ({100 * passed / max(len(cases), 1):.1f}%)")
    print(f"Failed: {failed}")
    print(f"Runtime: {time.time() - t0:.1f}s")

    if failures and not args.verbose:
        for idx, test_case, case_failures, result in failures[:10]:
            print(f"FAIL #{idx}: {test_case['query']}")
            for failure in case_failures[:3]:
                print(f"  -> {failure}")
            print(f"  ingredients={result['ingredients']}")
            print(f"  logic={result['ingredient_logic']}")
        if len(failures) > 10:
            print(f"... and {len(failures) - 10} more")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
