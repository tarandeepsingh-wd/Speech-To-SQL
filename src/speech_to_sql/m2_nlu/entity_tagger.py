import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from rapidfuzz import fuzz, process

from speech_to_sql.m2_nlu.extractor import extract_ingredients, extract_utensils
from speech_to_sql.m2_nlu.adaptive_lexicon import normalize_noisy_text
from speech_to_sql.m2_nlu.data.db_loader import DB_ACTIONS, DB_INGREDIENTS
from speech_to_sql.m2_nlu.data.ingredient_synonyms import CATEGORY_SYNONYMS, canonicalize_ingredient
from speech_to_sql.m2_nlu.intent_sql_mapper import build_sql_filter_plan
from speech_to_sql.m2_nlu.ingredient_reranker import rerank_ingredient_matches
from speech_to_sql.m2_nlu.query_decomposer import decompose_ingredient_logic
from speech_to_sql.m2_nlu.sql_plan_builder import build_unified_sql_plan
from speech_to_sql.m2_nlu.sql_ops import extract_sql_operators


# ---------------------------------------------------------
# DIETARY STYLE TERMS
# ---------------------------------------------------------

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

DIETARY_STYLE_ALIASES = {
    # Common ASR/typing slips seen in CLI input. Keep this explicit so the
    # dietary detector does not become too loose around ingredient names.
    "vigan": "vegan",
    "vegen": "vegan",
    "vegn": "vegan",
}

DIETARY_STYLE_LOOKUP = {
    **DIETARY_STYLES,
    **DIETARY_STYLE_ALIASES,
}


# ---------------------------------------------------------
# NUTRITION FILTER PATTERNS
# ---------------------------------------------------------

_NUTRIENT_MAP = {
    "calorie": "calories", "calories": "calories", "cal": "calories",
    "cals": "calories", "calor": "calories",
    "kcal": "calories",
    "protein": "protein", "proteins": "protein",
    "fat": "fat", "fats": "fat",
    "carb": "carbs", "carbs": "carbs", "carbohydrate": "carbs",
    "carbohydrates": "carbs",
    "fiber": "fiber", "fibre": "fiber",
    "sugar": "sugar", "sugars": "sugar",
    "sodium": "sodium", "salt": "sodium",
}


@lru_cache(maxsize=1)
def _schema_nutrient_map() -> dict[str, str]:
    try:
        from speech_to_sql.common.config import load_sql_schema_config
    except Exception:
        return {}

    try:
        schema = load_sql_schema_config()
    except Exception:
        return {}

    relation_columns = schema.get("relations", {}).get("nutrition", {}).get("columns", {})
    aliases = schema.get("semantic_aliases", {}).get("nutrition", {})
    result = {
        _normalize_schema_key(key): str(key)
        for key in relation_columns
        if str(key).strip()
    }
    result.update({
        _normalize_schema_key(alias): str(target)
        for alias, target in aliases.items()
        if str(alias).strip() and str(target).strip()
    })
    return result


def _nutrient_map() -> dict[str, str]:
    result = dict(_NUTRIENT_MAP)
    result.update(_schema_nutrient_map())
    return result


@lru_cache(maxsize=1)
def _schema_nutrition_groups() -> dict[str, dict]:
    try:
        from speech_to_sql.common.config import load_sql_schema_config
    except Exception:
        return {}

    try:
        schema = load_sql_schema_config()
    except Exception:
        return {}

    groups = {}
    for name, config in schema.get("nutrition_groups", {}).items():
        if not isinstance(config, dict):
            continue
        aliases = [
            str(alias)
            for alias in config.get("aliases", [])
            if str(alias).strip()
        ]
        aliases.append(str(name).replace("_", " "))
        groups[str(name)] = {
            "aliases": aliases,
            "operator": config.get("presence_operator") or schema.get("nutrition_presence", {}).get("default_operator", "ANY_GT"),
            "value": config.get("presence_value", schema.get("nutrition_presence", {}).get("default_value", 0)),
            "negation_cues": schema.get("nutrition_presence", {}).get("negation_cues", []),
        }
    return groups


@lru_cache(maxsize=1)
def _schema_nutrition_presence_config() -> dict:
    try:
        from speech_to_sql.common.config import load_sql_schema_config
    except Exception:
        return {}

    try:
        schema = load_sql_schema_config()
    except Exception:
        return {}

    config = schema.get("nutrition_presence", {})
    if not isinstance(config, dict):
        return {}
    cues = [
        str(cue).lower().strip()
        for cue in config.get("cues", [])
        if str(cue).strip()
    ]
    broad_cues = [
        str(cue).lower().strip()
        for cue in config.get("broad_alias_cues", cues)
        if str(cue).strip()
    ]
    return {
        "cues": cues,
        "broad_cues": broad_cues,
        "negation_cues": [
            str(cue).lower().strip()
            for cue in config.get("negation_cues", [])
            if str(cue).strip()
        ],
        "required_terms": [
            _normalize_schema_key(str(term))
            for term in config.get("presence_alias_required_terms", [])
            if str(term).strip()
        ],
        "suffix_terms": [
            _normalize_schema_key(str(term))
            for term in config.get("implicit_presence_suffix_terms", [])
            if str(term).strip()
        ],
        "operator": str(config.get("individual_operator", ">")),
        "value": float(config.get("individual_value", 0)),
    }


def _normalize_schema_key(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]+", " ", value.lower())).strip()

_NUTRITION_PATTERN = re.compile(
    r"(?P<op>under|below|less than|at most|"
    r"over|above|more than|greater than|higher than|larger than|at least|"
    r"around|about|approximately)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>g|mg|gram|grams|kcal|cal|calories|calorie)?\s*"
    r"(?P<nutrient>[a-z][a-z-]*)",
    re.IGNORECASE,
)

_NUTRITION_NUTRIENT_FIRST_PATTERN = re.compile(
    r"(?P<nutrient>[a-z][a-z-]*)\s*"
    r"(?:se\s+)?(?:is|are|was|were|has|have|contains?|with|of)?\s*"
    r"(?P<op>under|below|less than|at most|"
    r"over|above|more than|greater than|higher than|larger than|at least|"
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
    "over": ">",  "above": ">", "more than": ">", "greater than": ">",
    "higher than": ">", "larger than": ">", "at least": ">=",
    "around": "~", "about": "~", "approximately": "~",
    None: "=",
}
_OPERATOR_WORD_PATTERN = (
    r"under|below|less\s+than|at\s+most|"
    r"over|above|more\s+than|greater\s+than|higher\s+than|larger\s+than|at\s+least|"
    r"around|about|approximately"
)


def _normalize_nutrient(term: str) -> str | None:
    term = term.lower().strip()
    nutrient_map = _nutrient_map()
    normalized_term = _normalize_schema_key(term)
    if normalized_term in nutrient_map:
        return nutrient_map[normalized_term]
    if len(term) < 4:
        return None

    match = process.extractOne(
        normalized_term,
        list(nutrient_map.keys()),
        scorer=fuzz.WRatio,
        score_cutoff=84,
    )
    if not match:
        return None
    return nutrient_map[match[0]]


@lru_cache(maxsize=1)
def _dynamic_nutrient_patterns() -> tuple[tuple[str, str, re.Pattern, re.Pattern], ...]:
    patterns = []
    nutrient_map = _nutrient_map()
    for alias, nutrient in sorted(nutrient_map.items(), key=lambda item: len(item[0]), reverse=True):
        tokens = re.findall(r"[a-z0-9]+", alias.lower())
        if not tokens:
            continue
        alias_pattern = r"\b" + r"[\s,()+-]+".join(re.escape(token) for token in tokens) + r"\b"
        nutrient_first = re.compile(
            rf"(?P<nutrient>{alias_pattern})\s+"
            rf"(?:(?:is|are|was|were|has|have|contains?|with|of)\s+)?"
            rf"(?P<op>{_OPERATOR_WORD_PATTERN})\s+"
            rf"(?P<value>\d+(?:\.\d+)?)\s*"
            rf"(?P<unit>g|mg|gram|grams|kcal|cal|calories|calorie)?",
            re.IGNORECASE,
        )
        value_first = re.compile(
            rf"(?P<op>{_OPERATOR_WORD_PATTERN})\s+"
            rf"(?P<value>\d+(?:\.\d+)?)\s*"
            rf"(?P<unit>g|mg|gram|grams|kcal|cal|calories|calorie)?\s+"
            rf"(?:of\s+)?"
            rf"(?P<nutrient>{alias_pattern})",
            re.IGNORECASE,
        )
        patterns.append((alias, nutrient, value_first, nutrient_first))
    return tuple(patterns)


def _dynamic_nutrition_matches(text: str) -> list[dict]:
    found = []
    used_positions: set[int] = set()
    for _alias, nutrient, *patterns in _dynamic_nutrient_patterns():
        for pattern in patterns:
            for match in pattern.finditer(text.lower()):
                positions = set(range(match.start(), match.end()))
                if positions & used_positions:
                    continue
                op_text = re.sub(r"\s+", " ", (match.group("op") or "").lower()).strip()
                operator = _OP_MAP.get(op_text, "=")
                try:
                    value = float(match.group("value"))
                except (TypeError, ValueError):
                    continue
                used_positions |= positions
                found.append({
                    "span": match.span(),
                    "nutrient": nutrient,
                    "operator": operator,
                    "value": value,
                })
    found.sort(key=lambda item: item["span"])
    return found


_NUTRITION_CONTINUATION_PATTERN = re.compile(
    rf"(?:^|\b(?:but|and|or)\b|[,;])\s+"
    rf"(?:(?:also|still|should|must|be|is|are|to|remain|have|has)\s+){{0,4}}"
    rf"(?P<op>{_OPERATOR_WORD_PATTERN})\s+"
    rf"(?P<value>\d+(?:\.\d+)?)\s*"
    rf"(?P<unit>g|mg|gram|grams|kcal|cal|calories|calorie)?",
    re.IGNORECASE,
)

_TIME_SCOPE_BOUNDARY_PATTERN = re.compile(
    rf"\b(?:{_OPERATOR_WORD_PATTERN})\s+\d+(?:\.\d+)?\s*"
    r"(?:min|mins|minute|minutes|hr|hrs|hour|hours)\b|"
    r"\b(?:prep|prepare|prepares|prepared|preparing|preparation|"
    r"cook|cooks|cooked|cooking|ready|time|duration)\b",
    re.IGNORECASE,
)


def _nutrition_scope_boundaries(text: str) -> list[tuple[int, int]]:
    text_lower = text.lower()
    boundaries = []

    for _alias, _nutrient, pattern, _requires_broad_cue in _dynamic_nutrient_presence_patterns():
        for match in pattern.finditer(text_lower):
            boundaries.append(match.span())

    for match in _TIME_SCOPE_BOUNDARY_PATTERN.finditer(text_lower):
        boundaries.append(match.span())

    boundaries.sort()
    return boundaries


def _nutrition_scope_end(text: str, start: int) -> int:
    for boundary_start, _boundary_end in _nutrition_scope_boundaries(text):
        if boundary_start > start:
            return boundary_start
    return len(text)


def _looks_like_time_continuation(text: str, end: int) -> bool:
    suffix = text[end:end + 16]
    return bool(
        re.match(
            r"\s*(?:min|mins|minute|minutes|hr|hrs|hour|hours)\b",
            suffix,
            re.IGNORECASE,
        )
    )


def _looks_like_quantity_of_entity(text: str, end: int) -> bool:
    suffix = text[end:end + 32]
    return bool(
        re.match(
            r"\s*(?:ingredients?|recipes?|dishes?|items?)\b",
            suffix,
            re.IGNORECASE,
        )
    )


def _dynamic_nutrition_continuation_matches(
    text: str,
    base_matches: list[dict] | None = None,
) -> list[dict]:
    found = []
    text_lower = text.lower()
    base_matches = base_matches or _dynamic_nutrition_matches(text)
    for base in base_matches:
        scope_start = base["span"][1]
        scope_end = _nutrition_scope_end(text_lower, scope_start)
        scope_text = text_lower[scope_start:scope_end]
        for match in _NUTRITION_CONTINUATION_PATTERN.finditer(scope_text):
            end = scope_start + match.end()
            if _looks_like_time_continuation(text_lower, end):
                continue
            if not match.group("unit") and _looks_like_quantity_of_entity(text_lower, end):
                continue
            op_text = re.sub(r"\s+", " ", match.group("op").lower()).strip()
            try:
                value = float(match.group("value"))
            except (TypeError, ValueError):
                continue
            found.append({
                "span": (scope_start + match.start(), end),
                "nutrient": base["nutrient"],
                "operator": _OP_MAP.get(op_text, "="),
                "value": value,
            })
    found.sort(key=lambda item: item["span"])
    return found


@lru_cache(maxsize=1)
def _dynamic_nutrition_group_patterns() -> tuple[tuple[str, dict, re.Pattern], ...]:
    patterns = []
    for group, config in _schema_nutrition_groups().items():
        for alias in config.get("aliases", []):
            tokens = re.findall(r"[a-z0-9]+", alias.lower())
            if not tokens:
                continue
            alias_pattern = r"\b" + r"[\s,()+-]+".join(re.escape(token) for token in tokens) + r"\b"
            patterns.append((alias, group, config, re.compile(alias_pattern, re.IGNORECASE)))
    patterns.sort(key=lambda item: len(item[0]), reverse=True)
    return tuple(
        (group, config, pattern)
        for _alias, group, config, pattern in patterns
    )


def _dynamic_nutrition_group_matches(text: str) -> list[dict]:
    found = []
    text_lower = text.lower()
    used_positions: set[int] = set()
    for group, config, pattern in _dynamic_nutrition_group_patterns():
        for match in pattern.finditer(text_lower):
            positions = set(range(match.start(), match.end()))
            if positions & used_positions:
                continue
            if _has_recent_negation(text_lower, match.start(), config.get("negation_cues", [])):
                continue
            used_positions |= positions
            found.append({
                "span": match.span(),
                "nutrient_group": group,
                "operator": str(config.get("operator") or "ANY_GT"),
                "value": float(config.get("value", 0)),
            })
    found.sort(key=lambda item: item["span"])
    return found


def _alias_contains_required_term(alias: str, required_terms: list[str]) -> bool:
    if not required_terms:
        return False
    normalized_alias = _normalize_schema_key(alias)
    return any(
        re.search(rf"\b{re.escape(term)}\b", normalized_alias)
        for term in required_terms
    )


@lru_cache(maxsize=1)
def _dynamic_nutrient_presence_patterns() -> tuple[tuple[str, str, re.Pattern, bool], ...]:
    config = _schema_nutrition_presence_config()
    required_terms = config.get("required_terms", [])
    patterns = []
    for alias, nutrient in _nutrient_map().items():
        required_alias = _alias_contains_required_term(alias, required_terms)
        requires_broad_cue = (
            bool(required_terms)
            and not required_alias
            and _is_known_ingredient_term(alias)
        )
        tokens = re.findall(r"[a-z0-9]+", alias.lower())
        if not tokens:
            continue
        alias_pattern = r"\b" + r"[\s,()+-]*".join(re.escape(token) for token in tokens) + r"\b"
        patterns.append((alias, nutrient, re.compile(alias_pattern, re.IGNORECASE), requires_broad_cue))
    patterns.sort(key=lambda item: len(item[0]), reverse=True)
    return tuple(patterns)


def _cue_in_prefix(prefix: str, cues: list[str]) -> bool:
    prefix = _normalize_schema_key(prefix)
    for cue in cues:
        tokens = re.findall(r"[a-z0-9]+", cue)
        if not tokens:
            continue
        cue_pattern = r"\s+".join(re.escape(token) for token in tokens)
        if re.search(rf"\b{cue_pattern}\b(?:\s+\w+){{0,4}}\s*$", prefix):
            return True
    return False


def _suffix_mentions_presence_context(suffix: str, suffix_terms: list[str]) -> bool:
    if not suffix_terms:
        return False
    first_terms = " ".join(re.findall(r"[a-z0-9]+", suffix)[:4])
    return any(
        re.search(rf"\b{re.escape(term)}\b", first_terms)
        for term in suffix_terms
    )


def _suffix_mentions_numeric_comparison(suffix: str) -> bool:
    return bool(
        re.match(
            rf"^\s*(?:(?:is|are|was|were|has|have|contains?|with|of)\s+)?"
            rf"(?:{_OPERATOR_WORD_PATTERN})\b",
            suffix,
            re.IGNORECASE,
        )
    )


def _ingredient_phrase_context(prefix: str) -> bool:
    normalized = _normalize_schema_key(prefix)
    return bool(
        re.search(
            r"\bingredient\s+phrases?"
            r"(?:\s+(?:contains?|including|includes?|mention(?:s|ed)?|with|has|have))?$",
            normalized,
        )
    )


def _dynamic_nutrient_presence_matches(text: str) -> list[dict]:
    config = _schema_nutrition_presence_config()
    if not config:
        return []

    found = []
    text_lower = text.lower()
    used_positions: set[int] = set()
    saw_presence = False
    cues = config.get("cues", [])
    broad_cues = config.get("broad_cues", cues)
    suffix_terms = config.get("suffix_terms", [])
    negation_cues = config.get("negation_cues", [])

    candidates = []
    for alias, nutrient, pattern, requires_broad_cue in _dynamic_nutrient_presence_patterns():
        for match in pattern.finditer(text_lower):
            candidates.append((match.start(), match.end(), len(alias), nutrient, requires_broad_cue))

    candidates.sort(key=lambda item: (item[0], -item[2]))
    for start, end, _alias_len, nutrient, requires_broad_cue in candidates:
        positions = set(range(start, end))
        if positions & used_positions:
            continue
        if _has_recent_negation(text_lower, start, negation_cues):
            continue

        prefix = text_lower[max(0, start - 64):start]
        suffix = text_lower[end:end + 40]
        if _ingredient_phrase_context(prefix):
            continue
        if _suffix_mentions_numeric_comparison(suffix):
            continue

        connector_context = saw_presence and re.search(r"(?:\band\b|\bor\b|,)\s*$", prefix)
        if requires_broad_cue:
            if not (_cue_in_prefix(prefix, broad_cues) or connector_context):
                continue
        elif not (
            _cue_in_prefix(prefix, cues)
            or connector_context
            or _suffix_mentions_presence_context(suffix, suffix_terms)
        ):
            continue

        used_positions |= positions
        saw_presence = True
        found.append({
            "span": (start, end),
            "nutrient": nutrient,
            "operator": config.get("operator", ">"),
            "value": float(config.get("value", 0)),
        })

    found.sort(key=lambda item: item["span"])
    return found


def _has_recent_negation(text: str, start: int, negation_cues: list) -> bool:
    prefix = text[max(0, start - 24):start]
    return any(
        re.search(rf"\b{re.escape(str(cue).lower())}\b\s*$", prefix)
        for cue in negation_cues
        if str(cue).strip()
    )


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
    used_positions: set[int] = set()

    dynamic_numeric_matches = _dynamic_nutrition_matches(text)
    dynamic_numeric_matches.extend(
        _dynamic_nutrition_continuation_matches(text, dynamic_numeric_matches)
    )

    for item in dynamic_numeric_matches:
        nutrient = item["nutrient"]
        operator = item["operator"]
        value = item["value"]
        semantic_key = (nutrient, operator, value, item["span"])
        if semantic_key in seen_semantic:
            continue
        seen_semantic.add(semantic_key)
        seen_spans.add(item["span"])
        used_positions.update(range(item["span"][0], item["span"][1]))
        results.append({
            "nutrient": nutrient,
            "operator": operator,
            "value": value,
        })

    for pattern in _NUTRITION_PATTERNS:
        for m in pattern.finditer(text.lower()):
            if not _valid_nutrition_match(pattern, m):
                continue
            if set(range(m.start(), m.end())) & used_positions:
                continue
            if m.span() in seen_spans:
                continue
            seen_spans.add(m.span())
            nutrient = _normalize_nutrient(m.group("nutrient"))
            if not nutrient:
                continue
            op_text = (
                re.sub(r"\s+", " ", m.group("op").lower()).strip()
                if m.group("op")
                else None
            )
            operator = _OP_MAP.get(op_text, "=")
            value = float(m.group("value"))
            semantic_key = (nutrient, operator, value, m.span())
            if semantic_key in seen_semantic:
                continue
            seen_semantic.add(semantic_key)
            results.append({
                "nutrient": nutrient,
                "operator": operator,
                "value":    value,
            })

    for item in _dynamic_nutrient_presence_matches(text):
        if set(range(item["span"][0], item["span"][1])) & used_positions:
            continue
        semantic_key = (item["nutrient"], item["operator"], item["value"], item["span"])
        if semantic_key in seen_semantic:
            continue
        seen_semantic.add(semantic_key)
        used_positions.update(range(item["span"][0], item["span"][1]))
        results.append({
            "nutrient": item["nutrient"],
            "operator": item["operator"],
            "value": item["value"],
        })

    for item in _dynamic_nutrition_group_matches(text):
        if set(range(item["span"][0], item["span"][1])) & used_positions:
            continue
        semantic_key = (item["nutrient_group"], item["operator"], item["value"], item["span"])
        if semantic_key in seen_semantic:
            continue
        seen_semantic.add(semantic_key)
        used_positions.update(range(item["span"][0], item["span"][1]))
        results.append({
            "nutrient_group": item["nutrient_group"],
            "operator": item["operator"],
            "value": item["value"],
        })
    return results


# ---------------------------------------------------------
# SPAN MASKING
#
# The ingredient extractor must not see text that has already
# been claimed by the nutrition filter or dietary style detector.
#
# Without this:
#   "less than 555 calor" -> nutrition filter takes it,
#   BUT "calor" still reaches ingredient extractor -> fuzzy
#   matches "calorie mayonnaise" (a real 20k DB entry).
#
#   "vegan" -> dietary detector takes it,
#   BUT "vegan" still reaches ingredient bigram pass ->
#   matches "vegan nacho cheese" (a real 20k DB entry).
#
# Fix: replace matched spans with spaces (same length, so
# all other character offsets remain valid) before calling
# extract_ingredients().
# ---------------------------------------------------------

def _build_dietary_regex():
    """Build a single regex that matches any dietary style term."""
    parts = []
    for term in sorted(DIETARY_STYLE_LOOKUP.keys(), key=len, reverse=True):
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
    for item in _dynamic_nutrition_matches(text):
        start, end = item["span"]
        for i in range(start, end):
            if i < len(chars):
                chars[i] = " "
    for item in _dynamic_nutrition_continuation_matches(text):
        start, end = item["span"]
        for i in range(start, end):
            if i < len(chars):
                chars[i] = " "
    for item in _dynamic_nutrition_group_matches(text):
        start, end = item["span"]
        for i in range(start, end):
            if i < len(chars):
                chars[i] = " "
    for item in _dynamic_nutrient_presence_matches(text):
        start, end = item["span"]
        for i in range(start, end):
            if i < len(chars):
                chars[i] = " "
    for pattern in [*_NUTRITION_PATTERNS, _DIETARY_REGEX]:
        for m in pattern.finditer(text.lower()):
            if pattern in _NUTRITION_PATTERNS and not _valid_nutrition_match(pattern, m):
                continue
            if pattern in _NUTRITION_PATTERNS and not _normalize_nutrient(m.group("nutrient")):
                continue
            for i in range(m.start(), m.end()):
                if i < len(chars):
                    chars[i] = " "
    for start, end in extra_spans or []:
        for i in range(start, end + 1):
            if i < len(chars):
                chars[i] = " "
    return "".join(chars)


# ---------------------------------------------------------
# ACTION / PROCESS DETECTION
# ---------------------------------------------------------

_ACTION_TERMS = sorted(DB_ACTIONS.keys(), key=len, reverse=True)


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

    for token_m in re.finditer(r"\b[\w-]+\b", text_lower):
        token = token_m.group()
        if token in DB_ACTIONS:
            continue
        for base in _action_base_candidates(token):
            if base in DB_ACTIONS:
                positions = set(range(token_m.start(), token_m.end()))
                if positions & used_positions:
                    break
                used_positions |= positions
                found.append({
                    "action": base,
                    "type":   DB_ACTIONS[base]["type"],
                    "span":   (token_m.start(), token_m.end() - 1),
                })
                break

    return found


def _action_base_candidates(token: str) -> list[str]:
    candidates: list[str] = []

    def add(value: str) -> None:
        if value and value not in candidates:
            candidates.append(value)

    add(token)
    if token.endswith("ing") and len(token) > 4:
        root = token[:-3]
        add(root)
        add(root + "e")
        if len(root) > 2 and root[-1] == root[-2]:
            add(root[:-1])
    if token.endswith("ied") and len(token) > 4:
        add(token[:-3] + "y")
    if token.endswith("ed") and len(token) > 3:
        root = token[:-2]
        add(root)
        add(root + "e")
        if len(root) > 2 and root[-1] == root[-2]:
            add(root[:-1])
    return candidates


def _filter_actions_claimed_by_sql_ops(actions: list[dict], sql_operators: list[dict]) -> list[dict]:
    sql_spans = [
        tuple(item.get("span") or ())
        for item in sql_operators
        if str(item.get("canonical") or "").startswith(("LIMIT", "ORDER BY"))
    ]
    sql_spans = [
        span for span in sql_spans
        if len(span) == 2 and span[0] >= 0 and span[1] >= 0
    ]
    if not sql_spans:
        return actions

    filtered = []
    for action in actions:
        span = tuple(action.get("span") or ())
        if len(span) == 2 and any(_span_overlaps(span, sql_span) for sql_span in sql_spans):
            continue
        filtered.append(action)
    return filtered


def _span_overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


# ---------------------------------------------------------
# CATEGORY DETECTION
# ---------------------------------------------------------

_CUISINE_CATEGORY_CACHE: dict[str, str] | None = None


def _detect_categories(text: str) -> list:
    text_lower = text.lower()
    found      = []
    used       = set()
    category_synonyms = _category_synonyms()

    for term in sorted(category_synonyms.keys(), key=len, reverse=True):
        if _is_nutrition_term(term):
            continue
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text_lower):
            positions = set(range(m.start(), m.end()))
            if positions & used:
                continue
            used |= positions
            found.append({"term": term, "category": category_synonyms[term]})

    return found


def _category_claimed_spans(text: str, categories: list[dict]) -> list[tuple[int, int]]:
    text_lower = text.lower()
    spans: list[tuple[int, int]] = []
    used: set[int] = set()
    for item in categories:
        term = str(item.get("term") or "").strip().lower()
        if not term:
            continue
        if len(re.findall(r"[a-z0-9]+", term)) < 2:
            continue
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        for match in pattern.finditer(text_lower):
            positions = set(range(match.start(), match.end()))
            if positions & used:
                continue
            used |= positions
            spans.append((match.start(), match.end() - 1))
    return spans


def _category_synonyms() -> dict[str, str]:
    synonyms = dict(CATEGORY_SYNONYMS)
    synonyms.update(_load_cuisine_category_synonyms())
    return synonyms


def _load_cuisine_category_synonyms() -> dict[str, str]:
    global _CUISINE_CATEGORY_CACHE
    if _CUISINE_CATEGORY_CACHE is not None:
        return _CUISINE_CATEGORY_CACHE

    try:
        from speech_to_sql.common.config import load_recipe_db_config

        payload = load_recipe_db_config()
    except (OSError, json.JSONDecodeError):
        _CUISINE_CATEGORY_CACHE = {}
        return _CUISINE_CATEGORY_CACHE

    aliases = payload.get("cuisine_aliases", {})
    cuisine_terms = {
        str(term).lower().strip(): "cuisine"
        for term in aliases
        if str(term).strip()
    }
    flavor_terms = {
        str(term).lower().strip(): "ingredient_flavor"
        for term in payload.get("ingredient_flavor_aliases", {})
        if str(term).strip()
    }
    _CUISINE_CATEGORY_CACHE = {**cuisine_terms, **flavor_terms}
    return _CUISINE_CATEGORY_CACHE


def _is_nutrition_term(term: str) -> bool:
    return _normalize_nutrient(term) is not None


# ---------------------------------------------------------
# DIETARY STYLE DETECTION
# ---------------------------------------------------------

_STYLE_TERMS = sorted(DIETARY_STYLE_LOOKUP.keys(), key=len, reverse=True)


def _is_known_ingredient_term(text: str) -> bool:
    term = text.lower().strip()
    canonical = canonicalize_ingredient(term)
    return term in DB_INGREDIENTS or canonical in DB_INGREDIENTS


def _detect_dietary_style_matches(text: str) -> list:
    text_lower = text.lower()
    found      = []
    used       = set()

    for term in _STYLE_TERMS:
        if _is_nutrition_qualifier(term):
            continue
        p   = re.escape(term).replace(r"\ ", r"[\s-]").replace("\\-", "[\\s-]")
        pat = re.compile(r"\b" + p + r"\b", re.IGNORECASE)
        for m in pat.finditer(text_lower):
            positions = set(range(m.start(), m.end()))
            if positions & used:
                continue
            used |= positions
            found.append({
                "style": DIETARY_STYLE_LOOKUP[term],
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
            if width == 1 and phrase in {"free"}:
                continue
            if width == 1 and _is_known_ingredient_term(phrase):
                continue
            if _is_nutrition_qualifier(phrase):
                continue
            match = process.extractOne(
                phrase,
                _STYLE_TERMS,
                scorer=fuzz.WRatio,
                score_cutoff=88,
            )
            if not match:
                continue
            if _is_nutrition_qualifier(match[0]):
                continue
            used |= positions
            found.append({
                "style": DIETARY_STYLE_LOOKUP[match[0]],
                "span": (start, end),
            })

    return found


def _detect_dietary_styles(text: str) -> list:
    return list(dict.fromkeys(
        match["style"] for match in _detect_dietary_style_matches(text)
    ))


@lru_cache(maxsize=1)
def _nutrition_qualifier_terms() -> set[str]:
    try:
        from speech_to_sql.common.config import load_recipe_db_config

        payload = load_recipe_db_config()
    except (OSError, json.JSONDecodeError):
        return set()

    terms: set[str] = set()
    for rule in payload.get("intent_enrichment", {}).get("nutrition_qualifiers", []):
        if not isinstance(rule, dict):
            continue
        for term in rule.get("terms", []):
            normalized = _normalize_schema_key(str(term))
            if normalized:
                terms.add(normalized)
    return terms


def _is_nutrition_qualifier(term: str) -> bool:
    return _normalize_schema_key(term) in _nutrition_qualifier_terms()


# ---------------------------------------------------------
# MAIN TAG FUNCTION
# ---------------------------------------------------------

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

    The SQL execution path should build QueryIR from this intent and then use
    the Module 4 SQL builder. The legacy SQL plan fields are retained for older
    demos and debugging.
    """
    original_text = text
    normalization = normalize_noisy_text(text)
    text = normalization["text"]

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
    action_matches = _filter_actions_claimed_by_sql_ops(detected["actions"], sql_operators)
    categories = detected["categories"]
    utensil_matches = detected["utensils"]

    # Mask nutrition, dietary, and utensil spans so ingredient extraction stays
    # independent from the SQL operator layer while avoiding known false hits.
    claimed_spans = [m["span"] for m in utensil_matches]
    claimed_spans.extend(_category_claimed_spans(text, categories))
    claimed_spans.extend(m["span"] for m in dietary_matches)
    claimed_spans.extend(
        m["span"]
        for m in action_matches
        if not _is_known_ingredient_term(str(m.get("action") or ""))
    )
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
        "_original_query":   original_text,
        "_normalized_query": text,
        "_asr_corrections":  normalization["corrections"],
    }
