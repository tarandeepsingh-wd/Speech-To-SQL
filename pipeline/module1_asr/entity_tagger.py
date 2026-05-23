# from module1_asr.extractor import extract_ingredients

# FLAVOR_MOLECULES = {"limonene", "capsaicin", "vanillin", "menthol", "cinnamaldehyde"}
# DIETARY_MARKERS  = {"vegan", "vegetarian", "gluten-free", "keto", "diabetic", "low-sodium"}

# def tag_entities(text: str) -> dict:
#     tokens = text.lower().split()

#     ingredient_matches = extract_ingredients(text)

#     flavor  = []
#     dietary = []
#     for token in tokens:
#         clean = token.strip(".,?!")
#         if clean in FLAVOR_MOLECULES:
#             flavor.append(clean)
#         if clean in DIETARY_MARKERS:
#             dietary.append(clean)

#     return {
#         "ingredients":      [m["canonical"] for m in ingredient_matches],
#         "flavor_molecules": flavor,
#         "dietary_markers":  dietary,
#         "raw_spans":        ingredient_matches
#     }
import re

from module1_asr.extractor import extract_ingredients, extract_utensils
from module1_asr.data.db_loader import DB_ACTIONS
from module1_asr.data.ingredient_synonyms import CATEGORY_SYNONYMS

# ─────────────────────────────────────────────────────────
# COMPLETE REWRITE — original had 3 problems:
#
# PROBLEM 1: Only 5 flavor molecules, only 6 dietary markers,
#   both hardcoded. A system targeting RecipeDB + FlavorDB
#   + DietRx needs to detect all entity types relevant to
#   those databases.
#
# PROBLEM 2: Tokenizer was `text.lower().split()` — naive
#   whitespace split. This missed:
#     - hyphenated terms: "gluten-free" → ["gluten-free"]
#       but "gluten-free" != "gluten-free" after split
#       actually stays as one token, BUT "low-calorie"
#       would never match "low calorie" from the marker set.
#     - Multi-word markers from the category synonyms dict
#       e.g. "dry fruits" → two tokens, never matched as one.
#
# PROBLEM 3: Zero utensil detection, zero action/process
#   detection, zero nutrition filter detection. A user
#   saying "recipes using a skillet" or "under 500 calories"
#   would produce no entities for those dimensions, so the
#   SQL generator could not construct the right WHERE clauses.
#
# FIX: entity_tagger now detects 5 entity types:
#   1. ingredients  — via extract_ingredients() (Layer 1-4 pipeline)
#   2. utensils     — via extract_utensils() (exact Aho-Corasick)
#   3. actions      — via _detect_actions() (word-boundary scan)
#   4. categories   — via _detect_categories() (CATEGORY_SYNONYMS)
#   5. nutrition    — via _detect_nutrition_filters() (regex patterns)
# ─────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────
# DIETARY STYLE TERMS
# These map to the dietary_style column in RecipeDB recipes.
# Distinct from DietRx ingredient categories.
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
# Detects quantitative nutrition constraints in queries.
# e.g. "under 500 calories", "at least 30g protein",
#      "less than 10g fat", "more than 200 calories"
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
    """
    Extract quantitative nutrition constraints.
    Returns list of {"nutrient", "operator", "value"} dicts.
    e.g. "under 500 calories" → [{"nutrient": "calories",
                                   "operator": "<",
                                   "value": 500.0}]
    """
    results = []
    for m in _NUTRITION_PATTERN.finditer(text.lower()):
        nutrient = _NUTRIENT_MAP.get(m.group("nutrient").lower())
        if not nutrient:
            continue
        operator = _OP_MAP.get(
            m.group("op").lower() if m.group("op") else None,
            "="
        )
        results.append({
            "nutrient": nutrient,
            "operator": operator,
            "value":    float(m.group("value")),
        })
    return results


# ─────────────────────────────────────────────────────────
# ACTION / PROCESS DETECTION
# Only state_change actions are flagged as SQL-useful since
# procedural actions (stir, add, mix) appear in almost
# every recipe and don't meaningfully filter anything.
# ─────────────────────────────────────────────────────────

# Build a sorted list (longest first) so "slow cook" is
# matched before "cook" if we add multi-word actions later
_ACTION_TERMS = sorted(DB_ACTIONS.keys(), key=len, reverse=True)

# Also handle lemma variants: "stirring" → "stir"
# Simple heuristic: if token ends in "ing" and base is in DB_ACTIONS
_ING_SUFFIX = re.compile(r"^(\w+)ing$")


def _detect_actions(text: str) -> list:
    """
    Detect cooking process mentions in text.
    Returns list of {"action", "type", "span"} dicts.
    type is "state_change" or "procedural".
    """
    text_lower = text.lower()
    found = []
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

    # Handle "-ing" suffix variants not in DB (stirring → stir etc.)
    for token_m in re.finditer(r"\b\w+\b", text_lower):
        token = token_m.group()
        if token in DB_ACTIONS:
            continue  # already matched above
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
# CATEGORY DETECTION (DietRx ingredient categories)
# Uses CATEGORY_SYNONYMS which is now a SEPARATE dict
# from SYNONYMS — fixes the bug where category names
# like "Plant Derivative" were appearing as extracted
# ingredient names.
# ─────────────────────────────────────────────────────────

# Sorted longest-first so multi-word phrases match first
_CATEGORY_TERMS = sorted(CATEGORY_SYNONYMS.keys(), key=len, reverse=True)


def _detect_categories(text: str) -> list:
    """
    Detect DietRx ingredient category mentions.
    e.g. "dairy-free" → excludes "Dairy" category
         "vegan" → handled by dietary_styles
         "spices" → includes "Spice" category
    Returns list of {"term", "category"} dicts.
    """
    text_lower = text.lower()
    found = []
    used = set()

    for term in _CATEGORY_TERMS:
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text_lower):
            positions = set(range(m.start(), m.end()))
            if positions & used:
                continue
            used |= positions
            found.append({
                "term":     term,
                "category": CATEGORY_SYNONYMS[term],
            })

    return found


# ─────────────────────────────────────────────────────────
# DIETARY STYLE DETECTION
# Uses word-boundary regex on normalized text.
# Handles hyphenated terms ("gluten-free") and
# multi-word terms ("low sodium") correctly —
# unlike the original split() approach which would
# only match single tokens.
# ─────────────────────────────────────────────────────────

_STYLE_TERMS = sorted(DIETARY_STYLES.keys(), key=len, reverse=True)


def _detect_dietary_styles(text: str) -> list:
    """
    Returns list of canonical dietary style strings.
    """
    text_lower = text.lower()
    found = []
    used = set()

    for term in _STYLE_TERMS:
        # allow hyphen or space to match interchangeably
        pattern_str = re.escape(term).replace(r"\ ", r"[\s\-]").replace(r"\-", r"[\s\-]")
        pattern = re.compile(r"\b" + pattern_str + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text_lower):
            positions = set(range(m.start(), m.end()))
            if positions & used:
                continue
            used |= positions
            found.append(DIETARY_STYLES[term])

    return list(dict.fromkeys(found))  # deduplicate, preserve order


# ─────────────────────────────────────────────────────────
# MAIN TAG FUNCTION
# ─────────────────────────────────────────────────────────

def tag_entities(text: str) -> dict:
    """
    Extract all entity types from a natural language query.

    Returns:
        {
            "ingredients":      list of canonical ingredient name strings
            "utensils":         list of canonical utensil name strings
            "actions":          list of {"action", "type", "span"} dicts
            "dietary_styles":   list of canonical dietary style strings
            "categories":       list of {"term", "category"} dicts
            "nutrition_filters":list of {"nutrient", "operator", "value"} dicts
            "raw_spans":        full match dicts with span + layer + score
        }
    """
    ingredient_matches = extract_ingredients(text)
    utensil_matches    = extract_utensils(text)
    action_matches     = _detect_actions(text)
    dietary_styles     = _detect_dietary_styles(text)
    categories         = _detect_categories(text)
    nutrition_filters  = _detect_nutrition_filters(text)

    return {
        "ingredients":       [m["canonical"] for m in ingredient_matches],
        "utensils":          [m["canonical"] for m in utensil_matches],
        "actions":           action_matches,
        "dietary_styles":    dietary_styles,
        "categories":        categories,
        "nutrition_filters": nutrition_filters,
        "raw_spans":         ingredient_matches + utensil_matches,
    }