import re

from module1_asr.extractor import extract_ingredients, extract_utensils
from module1_asr.data.db_loader import DB_ACTIONS
from module1_asr.data.ingredient_synonyms import CATEGORY_SYNONYMS


# ─────────────────────────────────────────────────────────
# DIETARY STYLE TERMS
# ─────────────────────────────────────────────────────────

DIETARY_STYLES = {
    "vegan":               "vegan",
    "vegetarian":          "vegetarian",
    "pescatarian":         "pescatarian",
    "ovo-vegetarian":      "ovo-vegetarian",
    "lacto-vegetarian":    "lacto-vegetarian",
    "non-vegetarian":      "non-vegetarian",
    "non vegetarian":      "non-vegetarian",
    "nonvegetarian":       "non-vegetarian",
    "non-veg":             "non-vegetarian",
    "nonveg":              "non-vegetarian",
    "gluten-free":         "gluten-free",
    "gluten free":         "gluten-free",
    "dairy-free":          "dairy-free",
    "dairy free":          "dairy-free",
    "nut-free":            "nut-free",
    "nut free":            "nut-free",
    "keto":                "keto",
    "ketogenic":           "keto",
    "paleo":               "paleo",
    "halal":               "halal",
    "kosher":              "kosher",
    "diabetic":            "diabetic",
    "diabetic-friendly":   "diabetic",
    "low sodium":          "low-sodium",
    "low-sodium":          "low-sodium",
    "low salt":            "low-sodium",
    "sugar free":          "sugar-free",
    "sugar-free":          "sugar-free",
    "high protein":        "high-protein",
    "high-protein":        "high-protein",
    "low fat":             "low-fat",
    "low-fat":             "low-fat",
    "low calorie":         "low-calorie",
    "low-calorie":         "low-calorie",
}


# ─────────────────────────────────────────────────────────
# NUTRITION FILTER PATTERNS
# ─────────────────────────────────────────────────────────

_NUTRIENT_MAP = {
    "calorie": "calories", "calories": "calories", "cal": "calories",
    "kcal": "calories",
    "protein": "protein", "proteins": "protein",
    "fat": "fat", "fats": "fat",
    "carb": "carbs", "carbs": "carbs", "carbohydrate": "carbs",
    "carbohydrates": "carbs",
    "fiber": "fiber", "fibre": "fiber",
    "sugar": "sugar", "sugars": "sugar",
    "sodium": "sodium", "salt": "sodium",
}

_NUTRITION_PATTERN = re.compile(
    r"(?P<op>under|below|less than|at most|"
    r"over|above|more than|at least|"
    r"around|about|approximately)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>g|mg|kcal|cal|calories|calorie)?\s*"
    r"(?P<nutrient>calorie|calories|cal|kcal|"
    r"protein|proteins|fat|fats|"
    r"carb|carbs|carbohydrate|carbohydrates|"
    r"fiber|fibre|sugar|sugars|sodium)",
    re.IGNORECASE,
)

_OP_MAP = {
    "under": "<", "below": "<", "less than": "<", "at most": "<=",
    "over": ">",  "above": ">", "more than": ">", "at least": ">=",
    "around": "~", "about": "~", "approximately": "~",
    None: "=",
}


def _detect_nutrition_filters(text: str) -> list:
    results = []
    for m in _NUTRITION_PATTERN.finditer(text.lower()):
        nutrient = _NUTRIENT_MAP.get(m.group("nutrient").lower())
        if not nutrient:
            continue
        operator = _OP_MAP.get(
            m.group("op").lower() if m.group("op") else None, "="
        )
        results.append({
            "nutrient": nutrient,
            "operator": operator,
            "value":    float(m.group("value")),
        })
    return results


# ─────────────────────────────────────────────────────────
# SPAN MASKING
#
# The ingredient extractor must not see text that has already
# been claimed by the nutrition filter or dietary style detector.
#
# Without this:
#   "less than 555 calor" → nutrition filter takes it,
#   BUT "calor" still reaches ingredient extractor → fuzzy
#   matches "calorie mayonnaise" (a real 20k DB entry).
#
#   "vegan" → dietary detector takes it,
#   BUT "vegan" still reaches ingredient bigram pass →
#   matches "vegan nacho cheese" (a real 20k DB entry).
#
# Fix: replace matched spans with spaces (same length, so
# all other character offsets remain valid) before calling
# extract_ingredients().
# ─────────────────────────────────────────────────────────

def _build_dietary_regex():
    """Build a single regex that matches any dietary style term."""
    parts = []
    for term in sorted(DIETARY_STYLES.keys(), key=len, reverse=True):
        # Allow hyphen or space interchangeably in the pattern
        p = re.escape(term)
        p = p.replace(r"\ ", r"[\s-]")
        p = p.replace("\\-", "[\\s-]")
        parts.append(p)
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


_DIETARY_REGEX = _build_dietary_regex()


def _mask_claimed_spans(text: str) -> str:
    """
    Replace nutrition-filter and dietary-style spans with spaces.
    This prevents the ingredient extractor from processing words
    that have already been extracted by other detectors.
    """
    chars = list(text)
    for pattern in [_NUTRITION_PATTERN, _DIETARY_REGEX]:
        for m in pattern.finditer(text.lower()):
            for i in range(m.start(), m.end()):
                if i < len(chars):
                    chars[i] = " "
    return "".join(chars)


# ─────────────────────────────────────────────────────────
# ACTION / PROCESS DETECTION
# ─────────────────────────────────────────────────────────

_ACTION_TERMS = sorted(DB_ACTIONS.keys(), key=len, reverse=True)
_ING_SUFFIX   = re.compile(r"^(\w+)ing$")


def _detect_actions(text: str) -> list:
    text_lower     = text.lower()
    found          = []
    used_positions = set()

    for action in _ACTION_TERMS:
        pattern = re.compile(r"\b" + re.escape(action) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text_lower):
            positions = set(range(m.start(), m.end()))
            if positions & used_positions:
                continue
            used_positions |= positions
            found.append({
                "action": action,
                "type":   DB_ACTIONS[action]["type"],
                "span":   (m.start(), m.end() - 1),
            })

    for token_m in re.finditer(r"\b\w+\b", text_lower):
        token = token_m.group()
        if token in DB_ACTIONS:
            continue
        ing_m = _ING_SUFFIX.match(token)
        if ing_m:
            base = ing_m.group(1)
            if base in DB_ACTIONS:
                positions = set(range(token_m.start(), token_m.end()))
                if positions & used_positions:
                    continue
                used_positions |= positions
                found.append({
                    "action": base,
                    "type":   DB_ACTIONS[base]["type"],
                    "span":   (token_m.start(), token_m.end() - 1),
                })

    return found


# ─────────────────────────────────────────────────────────
# CATEGORY DETECTION
# ─────────────────────────────────────────────────────────

_CATEGORY_TERMS = sorted(CATEGORY_SYNONYMS.keys(), key=len, reverse=True)


def _detect_categories(text: str) -> list:
    text_lower = text.lower()
    found      = []
    used       = set()

    for term in _CATEGORY_TERMS:
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text_lower):
            positions = set(range(m.start(), m.end()))
            if positions & used:
                continue
            used |= positions
            found.append({"term": term, "category": CATEGORY_SYNONYMS[term]})

    return found


# ─────────────────────────────────────────────────────────
# DIETARY STYLE DETECTION
# ─────────────────────────────────────────────────────────

_STYLE_TERMS = sorted(DIETARY_STYLES.keys(), key=len, reverse=True)


def _detect_dietary_styles(text: str) -> list:
    text_lower = text.lower()
    found      = []
    used       = set()

    for term in _STYLE_TERMS:
        p   = re.escape(term).replace(r"\ ", r"[\s-]").replace("\\-", "[\\s-]")
        pat = re.compile(r"\b" + p + r"\b", re.IGNORECASE)
        for m in pat.finditer(text_lower):
            positions = set(range(m.start(), m.end()))
            if positions & used:
                continue
            used |= positions
            found.append(DIETARY_STYLES[term])

    return list(dict.fromkeys(found))


# ─────────────────────────────────────────────────────────
# MAIN TAG FUNCTION
# ─────────────────────────────────────────────────────────

def tag_entities(text: str) -> dict:
    """
    Extract all entity types from a natural language query.

    Returns:
        {
            "ingredients":       list[str]
            "utensils":          list[str]
            "actions":           list[dict]
            "dietary_styles":    list[str]
            "categories":        list[dict]
            "nutrition_filters": list[dict]
            "raw_spans":         list[dict]
        }
    """
    # Run non-ingredient detectors first
    dietary_styles    = _detect_dietary_styles(text)
    nutrition_filters = _detect_nutrition_filters(text)
    action_matches    = _detect_actions(text)
    categories        = _detect_categories(text)

    # Mask nutrition + dietary spans so ingredient extractor
    # never processes words already claimed by other detectors
    masked = _mask_claimed_spans(text)

    ingredient_matches = extract_ingredients(masked)
    utensil_matches    = extract_utensils(text)

    return {
        "ingredients":       [m["canonical"] for m in ingredient_matches],
        "utensils":          [m["canonical"] for m in utensil_matches],
        "actions":           action_matches,
        "dietary_styles":    dietary_styles,
        "categories":        categories,
        "nutrition_filters": nutrition_filters,
        "raw_spans":         ingredient_matches + utensil_matches,
    }