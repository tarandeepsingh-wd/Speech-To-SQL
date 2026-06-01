import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from rapidfuzz import fuzz, process

from module1_asr.extractor import extract_ingredients, extract_utensils
from module1_asr.data.db_loader import DB_ACTIONS, DB_INGREDIENTS
from module1_asr.data.ingredient_synonyms import CATEGORY_SYNONYMS, canonicalize_ingredient
from module1_asr.intent_sql_mapper import build_sql_filter_plan
from module1_asr.ingredient_reranker import rerank_ingredient_matches
from module1_asr.query_decomposer import decompose_ingredient_logic
from module1_asr.sql_plan_builder import build_unified_sql_plan
from module1_asr.sql_ops import extract_sql_operators


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
    r"(?P<unit>g|mg|gram|grams|kcal|cal|calories|calorie)?\s*"
    r"(?P<nutrient>[a-z][a-z-]*)",
    re.IGNORECASE,
)

_NUTRITION_NUTRIENT_FIRST_PATTERN = re.compile(
    r"(?P<nutrient>[a-z][a-z-]*)\s*"
    r"(?:se\s+)?"
    r"(?P<op>under|below|less than|at most|"
    r"over|above|more than|at least|"
    r"around|about|approximately)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>g|mg|gram|grams|kcal|cal|calories|calorie)?",
    re.IGNORECASE,
)

_NUTRITION_PATTERNS = [_NUTRITION_PATTERN, _NUTRITION_NUTRIENT_FIRST_PATTERN]
_CALORIE_TERMS = {"calorie", "calories", "cal", "kcal"}
_NUTRIENT_AFTER_PATTERN = re.compile(
    r"^\s*([a-z][a-z-]*)\b",
    re.IGNORECASE,
)

_OP_MAP = {
    "under": "<", "below": "<", "less than": "<", "at most": "<=",
    "over": ">",  "above": ">", "more than": ">", "at least": ">=",
    "around": "~", "about": "~", "approximately": "~",
    None: "=",
}


def _normalize_nutrient(term: str) -> str | None:
    term = term.lower().strip()
    if term in _NUTRIENT_MAP:
        return _NUTRIENT_MAP[term]
    if len(term) < 4:
        return None

    match = process.extractOne(
        term,
        list(_NUTRIENT_MAP.keys()),
        scorer=fuzz.WRatio,
        score_cutoff=84,
    )
    if not match:
        return None
    return _NUTRIENT_MAP[match[0]]


def _valid_nutrition_match(pattern, match) -> bool:
    if pattern is not _NUTRITION_NUTRIENT_FIRST_PATTERN:
        return True

    leading = _normalize_nutrient(match.group("nutrient"))
    if not leading:
        return False
    trailing_unit = (match.group("unit") or "").lower()
    if trailing_unit in _CALORIE_TERMS and leading != "calories":
        return False
    trailing_text = match.string[match.end():match.end() + 16]
    trailing_nutrient = _NUTRIENT_AFTER_PATTERN.match(trailing_text)
    if trailing_nutrient:
        next_nutrient = _normalize_nutrient(trailing_nutrient.group(1))
        if next_nutrient and next_nutrient != leading:
            return False
    return True


def _detect_nutrition_filters(text: str) -> list:
    results = []
    seen_spans = set()
    seen_semantic = set()
    for pattern in _NUTRITION_PATTERNS:
        for m in pattern.finditer(text.lower()):
            if not _valid_nutrition_match(pattern, m):
                continue
            if m.span() in seen_spans:
                continue
            seen_spans.add(m.span())
            nutrient = _normalize_nutrient(m.group("nutrient"))
            if not nutrient:
                continue
            operator = _OP_MAP.get(
                m.group("op").lower() if m.group("op") else None, "="
            )
            value = float(m.group("value"))
            semantic_key = (nutrient, operator, value)
            if semantic_key in seen_semantic:
                continue
            seen_semantic.add(semantic_key)
            results.append({
                "nutrient": nutrient,
                "operator": operator,
                "value":    value,
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


def _mask_claimed_spans(text: str, extra_spans: list | None = None) -> str:
    """
    Replace nutrition-filter and dietary-style spans with spaces.
    This prevents the ingredient extractor from processing words
    that have already been extracted by other detectors.
    """
    chars = list(text)
    for pattern in [*_NUTRITION_PATTERNS, _DIETARY_REGEX]:
        for m in pattern.finditer(text.lower()):
            if pattern in _NUTRITION_PATTERNS and not _valid_nutrition_match(pattern, m):
                continue
            for i in range(m.start(), m.end()):
                if i < len(chars):
                    chars[i] = " "
    for start, end in extra_spans or []:
        for i in range(start, end + 1):
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


def _is_known_ingredient_term(text: str) -> bool:
    term = text.lower().strip()
    canonical = canonicalize_ingredient(term)
    return term in DB_INGREDIENTS or canonical in DB_INGREDIENTS


def _detect_dietary_style_matches(text: str) -> list:
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
            found.append({
                "style": DIETARY_STYLES[term],
                "span": (m.start(), m.end() - 1),
            })

    tokens = list(re.finditer(r"\b[a-z][a-z-]*\b", text_lower))
    for width in (3, 2, 1):
        for i in range(0, len(tokens) - width + 1):
            start = tokens[i].start()
            end = tokens[i + width - 1].end() - 1
            positions = set(range(start, end + 1))
            if positions & used:
                continue
            phrase = text_lower[start:end + 1]
            if len(phrase.replace(" ", "").replace("-", "")) < 4:
                continue
            if width == 1 and _is_known_ingredient_term(phrase):
                continue
            match = process.extractOne(
                phrase,
                _STYLE_TERMS,
                scorer=fuzz.WRatio,
                score_cutoff=88,
            )
            if not match:
                continue
            used |= positions
            found.append({
                "style": DIETARY_STYLES[match[0]],
                "span": (start, end),
            })

    return found


def _detect_dietary_styles(text: str) -> list:
    return list(dict.fromkeys(
        match["style"] for match in _detect_dietary_style_matches(text)
    ))


# ─────────────────────────────────────────────────────────
# MAIN TAG FUNCTION
# ─────────────────────────────────────────────────────────

def tag_entities(text: str) -> dict:
    """
    Extract all entity types from a natural language query.

    Returns:
        {
            "ingredients":       list[str]   - canonical ingredient names
            "utensils":          list[str]   - canonical utensil names
            "actions":           list[dict]  - action/type/span matches
            "dietary_styles":    list[str]   - e.g. ["vegan"]
            "categories":        list[dict]  - category synonym matches
            "nutrition_filters": list[dict]  - nutrient/operator/value filters
            "sql_operators":     list[dict]  - SQL operator spans
            "ingredient_logic":  dict        - include/exclude/connectors
            "sql_filter_plan":   dict        - ingredient-only WHERE plan
            "unified_sql_plan":  dict        - ingredients + nutrition + dietary
            "raw_spans":         list[dict]  - matched ingredient/utensil spans
        }

    Module 3 should prefer unified_sql_plan. For precise mixed AND/OR ingredient
    logic, inspect ingredient_logic["connectors"] when sql_filter_plan is marked
    ambiguous.
    """
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_detect_dietary_style_matches, text): "dietary",
            executor.submit(_detect_nutrition_filters, text): "nutrition",
            executor.submit(extract_sql_operators, text): "sql_ops",
            executor.submit(_detect_actions, text): "actions",
            executor.submit(_detect_categories, text): "categories",
            executor.submit(extract_utensils, text): "utensils",
        }
        detected = {}
        for future in as_completed(futures):
            detected[futures[future]] = future.result()

    dietary_matches = detected["dietary"]
    dietary_styles = list(dict.fromkeys(m["style"] for m in dietary_matches))
    nutrition_filters = detected["nutrition"]
    sql_operators = detected["sql_ops"]
    action_matches = detected["actions"]
    categories = detected["categories"]
    utensil_matches = detected["utensils"]

    # Mask nutrition, dietary, and utensil spans so ingredient extraction stays
    # independent from the SQL operator layer while avoiding known false hits.
    claimed_spans = [m["span"] for m in utensil_matches]
    claimed_spans.extend(m["span"] for m in dietary_matches)
    masked = _mask_claimed_spans(text, claimed_spans)
    ingredient_matches = extract_ingredients(masked)
    ingredient_matches = rerank_ingredient_matches(text, ingredient_matches)
    ingredient_logic = decompose_ingredient_logic(text, ingredient_matches)
    sql_filter_plan = build_sql_filter_plan(ingredient_logic)
    unified_sql_plan = build_unified_sql_plan(
        sql_filter_plan,
        nutrition_filters,
        dietary_styles,
    )
    ingredients = list(dict.fromkeys(
        ingredient_logic["include"] + ingredient_logic["exclude"]
    ))

    return {
        "ingredients":       ingredients,
        "utensils":          [m["canonical"] for m in utensil_matches],
        "actions":           action_matches,
        "dietary_styles":    dietary_styles,
        "categories":        categories,
        "nutrition_filters": nutrition_filters,
        "sql_operators":     sql_operators,
        "ingredient_logic":  ingredient_logic,
        "sql_filter_plan":   sql_filter_plan,
        "unified_sql_plan":  unified_sql_plan,
        "raw_spans":         ingredient_matches + utensil_matches,
    }
