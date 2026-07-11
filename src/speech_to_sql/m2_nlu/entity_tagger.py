import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from rapidfuzz import fuzz, process

from speech_to_sql.m2_nlu.extractor import (
    extract_actions,
    extract_ingredients,
    extract_strong_ingredient_spans,
    extract_utensils,
)
from speech_to_sql.m2_nlu.adaptive_lexicon import normalize_noisy_text
from speech_to_sql.m2_nlu.data.db_loader import DB_ACTIONS, DB_INGREDIENTS, DB_UTENSILS
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
    "veg": "vegetarian",
    "veggie": "vegetarian",
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


_BASE_OP_MAP = {
    "less than or equal": "<=",
    "greater than or equal": ">=",
    "more than or equal": ">=",
    "under": "<",
    "below": "<",
    "less than": "<",
    "fewer than": "<",
    "at most": "<=",
    "fewer than or equal": "<=",
    "over": ">",
    "above": ">",
    "more than": ">",
    "greater than": ">",
    "higher than": ">",
    "larger than": ">",
    "at least": ">=",
    "around": "~",
    "about": "~",
    "approximately": "~",
}
_NEGATED_OPERATOR_PREFIX_RE = re.compile(r"^(?:not|no)\s+(.+)$", re.IGNORECASE)
_NEGATED_OPERATOR_MAP = {
    "<": ">=",
    "<=": ">",
    ">": "<=",
    ">=": "<",
}


def _operator_pattern(terms: list[str]) -> str:
    parts = []
    for term in sorted(terms, key=len, reverse=True):
        tokens = re.findall(r"[a-z0-9]+", term.lower())
        if tokens:
            parts.append(r"\s+".join(re.escape(token) for token in tokens))
    return "|".join(parts)


_BASE_OPERATOR_WORD_PATTERN = _operator_pattern(list(_BASE_OP_MAP))
_OPERATOR_WORD_PATTERN = rf"(?:(?:not|no)\s+)?(?:{_BASE_OPERATOR_WORD_PATTERN})"
_NUTRITION_UNIT_PATTERN = (
    r"milligrams|milligram|grams|gram|gms|gm|mg|g|"
    r"kcal|calories|calorie|cal"
)
_NUTRITION_SEPARATOR_PATTERN = r"[\s,;:()\[\]{}-]*"
_NUTRITION_REQUIRED_SEPARATOR_PATTERN = r"[\s,;:()\[\]{}-]+"
_NUTRIENT_ALIAS_SEPARATOR_PATTERN = r"[\s,;:()+\[\]{}-]+"
_NUTRITION_RELATION_PATTERN = (
    rf"(?:(?:se|is|are|was|were|has|have|contains?|with|of)"
    rf"{_NUTRITION_SEPARATOR_PATTERN})?"
)
_NUTRITION_OF_PATTERN = rf"(?:of\b{_NUTRITION_SEPARATOR_PATTERN})?"
_NEGATED_NUTRITION_RELATION_RE = re.compile(
    r"\b(?:(?:do|does|did|should|must|can|could|would|is|are|was|were)\s+)?"
    r"(?:not|dont|doesnt|didnt|no|without)\b"
    r"(?:\s+(?:be|being|have|has|having|contain|contains|containing|"
    r"include|includes|including|with)){0,3}$",
    re.IGNORECASE,
)


def _operator_from_text(op_text: str | None, default: str = "=") -> str:
    normalized = re.sub(r"\s+", " ", (op_text or "").lower()).strip()
    if not normalized:
        return default

    negated = _NEGATED_OPERATOR_PREFIX_RE.match(normalized)
    if negated:
        base = negated.group(1).strip()
        operator = _BASE_OP_MAP.get(base)
        if operator in _NEGATED_OPERATOR_MAP:
            return _NEGATED_OPERATOR_MAP[operator]
        return default
    return _BASE_OP_MAP.get(normalized, default)


def _is_negated_operator_text(op_text: str | None) -> bool:
    normalized = re.sub(r"\s+", " ", (op_text or "").lower()).strip()
    return bool(_NEGATED_OPERATOR_PREFIX_RE.match(normalized))


def _comparison_scope_is_negated(text: str, start: int) -> bool:
    prefix = (
        text[:start]
        .lower()
        .replace("don't", "dont")
        .replace("doesn't", "doesnt")
        .replace("didn't", "didnt")
    )
    local_clause = re.split(r"\b(?:but|and|or|then)\b|[,;]", prefix)[-1]
    local_clause = re.sub(r"\s+", " ", local_clause).strip()
    return bool(_NEGATED_NUTRITION_RELATION_RE.search(local_clause))


def _operator_from_match(
    text: str,
    match: re.Match,
    *,
    offset: int = 0,
    default: str = "=",
) -> str:
    op_text = match.group("op")
    operator = _operator_from_text(op_text, default=default)
    if (
        op_text
        and not _is_negated_operator_text(op_text)
        and operator in _NEGATED_OPERATOR_MAP
        and _comparison_scope_is_negated(text, offset + match.start())
    ):
        return _NEGATED_OPERATOR_MAP[operator]
    return operator


_NUTRITION_PATTERN = re.compile(
    rf"(?P<op>{_OPERATOR_WORD_PATTERN})?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    rf"(?P<unit>{_NUTRITION_UNIT_PATTERN})?"
    rf"{_NUTRITION_SEPARATOR_PATTERN}"
    rf"{_NUTRITION_OF_PATTERN}"
    r"(?P<nutrient>[a-z][a-z-]*)",
    re.IGNORECASE,
)

_NUTRITION_NUTRIENT_FIRST_PATTERN = re.compile(
    rf"(?P<nutrient>[a-z][a-z-]*){_NUTRITION_SEPARATOR_PATTERN}"
    rf"{_NUTRITION_RELATION_PATTERN}"
    rf"(?P<op>{_OPERATOR_WORD_PATTERN})?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    rf"(?P<unit>{_NUTRITION_UNIT_PATTERN})?",
    re.IGNORECASE,
)

_NUTRITION_PATTERNS = [_NUTRITION_PATTERN, _NUTRITION_NUTRIENT_FIRST_PATTERN]
_CALORIE_TERMS = {"calorie", "calories", "cal", "kcal"}
_NUTRIENT_AFTER_PATTERN = re.compile(
    r"^\s*([a-z][a-z-]*)\b",
    re.IGNORECASE,
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
        alias_pattern = r"\b" + _NUTRIENT_ALIAS_SEPARATOR_PATTERN.join(
            re.escape(token) for token in tokens
        ) + r"\b"
        nutrient_first = re.compile(
            rf"(?P<nutrient>{alias_pattern})"
            rf"{_NUTRITION_SEPARATOR_PATTERN}"
            rf"{_NUTRITION_RELATION_PATTERN}"
            rf"(?P<op>{_OPERATOR_WORD_PATTERN})\s+"
            rf"(?P<value>\d+(?:\.\d+)?)\s*"
            rf"(?P<unit>{_NUTRITION_UNIT_PATTERN})?",
            re.IGNORECASE,
        )
        value_first = re.compile(
            rf"(?P<op>{_OPERATOR_WORD_PATTERN})\s+"
            rf"(?P<value>\d+(?:\.\d+)?)\s*"
            rf"(?P<unit>{_NUTRITION_UNIT_PATTERN})?"
            rf"{_NUTRITION_REQUIRED_SEPARATOR_PATTERN}"
            rf"{_NUTRITION_OF_PATTERN}"
            rf"(?P<nutrient>{alias_pattern})",
            re.IGNORECASE,
        )
        patterns.append((alias, nutrient, value_first, nutrient_first))
    return tuple(patterns)


def _dynamic_nutrition_matches(text: str) -> list[dict]:
    found = []
    candidates = []
    used_positions: set[int] = set()
    for _alias, nutrient, value_first, nutrient_first in _dynamic_nutrient_patterns():
        for pattern_order, pattern in enumerate((nutrient_first, value_first)):
            for match in pattern.finditer(text.lower()):
                candidates.append((match.start(), pattern_order, match.end(), nutrient, match))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    for _start, _pattern_order, _end, nutrient, match in candidates:
        positions = set(range(match.start(), match.end()))
        if positions & used_positions:
            continue
        operator = _operator_from_match(text, match)
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


def _dynamic_nutrition_between_matches(text: str) -> list[dict]:
    found = []
    text_lower = text.lower()
    used_positions: set[int] = set()
    for alias, nutrient, *_patterns in _dynamic_nutrient_patterns():
        tokens = re.findall(r"[a-z0-9]+", alias.lower())
        if not tokens:
            continue
        alias_pattern = r"\b" + _NUTRIENT_ALIAS_SEPARATOR_PATTERN.join(
            re.escape(token) for token in tokens
        ) + r"\b"
        patterns = (
            re.compile(
                rf"(?P<nutrient>{alias_pattern})"
                rf"{_NUTRITION_SEPARATOR_PATTERN}"
                rf"{_NUTRITION_RELATION_PATTERN}"
                rf"between\s+(?P<min>\d+(?:\.\d+)?)\s*"
                rf"(?P<unit1>{_NUTRITION_UNIT_PATTERN})?"
                rf"{_NUTRITION_REQUIRED_SEPARATOR_PATTERN}"
                rf"(?:and|to)\s+(?P<max>\d+(?:\.\d+)?)\s*"
                rf"(?P<unit2>{_NUTRITION_UNIT_PATTERN})?",
                re.IGNORECASE,
            ),
            re.compile(
                rf"between\s+(?P<min>\d+(?:\.\d+)?)\s*"
                rf"(?P<unit1>{_NUTRITION_UNIT_PATTERN})?"
                rf"{_NUTRITION_REQUIRED_SEPARATOR_PATTERN}"
                rf"(?:and|to)\s+(?P<max>\d+(?:\.\d+)?)\s*"
                rf"(?P<unit2>{_NUTRITION_UNIT_PATTERN})?"
                rf"{_NUTRITION_REQUIRED_SEPARATOR_PATTERN}"
                rf"{_NUTRITION_OF_PATTERN}(?P<nutrient>{alias_pattern})",
                re.IGNORECASE,
            ),
        )
        for pattern in patterns:
            for match in pattern.finditer(text_lower):
                positions = set(range(match.start(), match.end()))
                if positions & used_positions:
                    continue
                try:
                    low = float(match.group("min"))
                    high = float(match.group("max"))
                except (TypeError, ValueError):
                    continue
                if low > high:
                    low, high = high, low
                used_positions |= positions
                found.append({
                    "span": match.span(),
                    "nutrient": nutrient,
                    "operator": "BETWEEN",
                    "values": (low, high),
                })
    found.sort(key=lambda item: item["span"])
    return found


_NUTRITION_CONTINUATION_PATTERN = re.compile(
    rf"(?:^|\b(?:but|and|or)\b|{_NUTRITION_REQUIRED_SEPARATOR_PATTERN})"
    rf"{_NUTRITION_SEPARATOR_PATTERN}"
    rf"(?:(?:also|still|should|must|be|is|are|to|remain|have|has)\s+){{0,4}}"
    rf"(?P<op>{_OPERATOR_WORD_PATTERN})\s+"
    rf"(?P<value>\d+(?:\.\d+)?)\s*"
    rf"(?P<unit>{_NUTRITION_UNIT_PATTERN})?",
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


def _looks_like_explicit_nutrient_after_value(text: str, end: int) -> bool:
    suffix = text[end:end + 48]
    tokens = re.findall(r"[a-z][a-z-]*", suffix.lower())[:4]
    for token in tokens:
        if token in {"and", "or", "but", "with", "of"}:
            continue
        return _normalize_nutrient(token) is not None
    return False


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
            if _looks_like_explicit_nutrient_after_value(text_lower, end):
                continue
            try:
                value = float(match.group("value"))
            except (TypeError, ValueError):
                continue
            found.append({
                "span": (scope_start + match.start(), end),
                "nutrient": base["nutrient"],
                "operator": _operator_from_match(text_lower, match, offset=scope_start),
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
        alias_pattern = r"\b" + _NUTRITION_SEPARATOR_PATTERN.join(
            re.escape(token) for token in tokens
        ) + r"\b"
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
            rf"^{_NUTRITION_SEPARATOR_PATTERN}{_NUTRITION_RELATION_PATTERN}"
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
    if not (match.group("op") or match.group("unit")):
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

    dynamic_between_matches = _dynamic_nutrition_between_matches(text)
    dynamic_numeric_matches = _dynamic_nutrition_matches(text)
    dynamic_numeric_matches.extend(
        _dynamic_nutrition_continuation_matches(text, dynamic_numeric_matches)
    )

    for item in dynamic_between_matches:
        nutrient = item["nutrient"]
        values = tuple(item["values"])
        semantic_key = (nutrient, item["operator"], values, item["span"])
        if semantic_key in seen_semantic:
            continue
        seen_semantic.add(semantic_key)
        seen_spans.add(item["span"])
        used_positions.update(range(item["span"][0], item["span"][1]))
        results.append({
            "nutrient": nutrient,
            "operator": item["operator"],
            "values": values,
        })

    for item in dynamic_numeric_matches:
        if set(range(item["span"][0], item["span"][1])) & used_positions:
            continue
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
            operator = _operator_from_match(text, m)
            value = float(m.group("value"))
            semantic_key = (nutrient, operator, value, m.span())
            if semantic_key in seen_semantic:
                continue
            seen_semantic.add(semantic_key)
            used_positions.update(range(m.start(), m.end()))
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
    for start, end in _nutrition_qualifier_match_spans(text):
        for i in range(start, end):
            if i < len(chars):
                chars[i] = " "
    for item in _dynamic_nutrition_matches(text):
        start, end = item["span"]
        for i in range(start, end):
            if i < len(chars):
                chars[i] = " "
    for item in _dynamic_nutrition_between_matches(text):
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

_NON_QUERY_ACTION_TERMS = {
    "check",
    "clean",
    "dry",
    "note",
    "repeat",
    "smoke",
    "test",
    "take",
}


def _detect_actions(text: str) -> list:
    return extract_actions(text, excluded_terms=_NON_QUERY_ACTION_TERMS)


def _measurement_claimed_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    dynamic_matches = _dynamic_nutrition_matches(text)
    dynamic_matches.extend(_dynamic_nutrition_between_matches(text))
    dynamic_matches.extend(_dynamic_nutrition_continuation_matches(text, dynamic_matches))
    dynamic_matches.extend(_dynamic_nutrition_group_matches(text))
    dynamic_matches.extend(_dynamic_nutrient_presence_matches(text))
    spans.extend(tuple(item["span"]) for item in dynamic_matches if item.get("span"))

    for pattern in _NUTRITION_PATTERNS:
        for match in pattern.finditer(text.lower()):
            if not _valid_nutrition_match(pattern, match):
                continue
            if not _normalize_nutrient(match.group("nutrient")):
                continue
            spans.append(match.span())
    return [
        (start, end - 1)
        for start, end in spans
        if end > start
    ]


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


def _sql_operator_claimed_spans(sql_operators: list[dict]) -> list[tuple[int, int]]:
    """Return visible operator spans that should not be reused as ingredients."""
    spans: list[tuple[int, int]] = []
    for item in sql_operators:
        span = tuple(item.get("span") or ())
        if len(span) != 2:
            continue
        start, end = span
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end < 0 or end < start:
            continue
        spans.append((start, end))
    return spans


def _sort_field_claimed_spans(text: str) -> list[tuple[int, int]]:
    try:
        from speech_to_sql.common.config import load_recipe_db_config

        config = load_recipe_db_config()
    except (OSError, json.JSONDecodeError):
        return []

    text_lower = text.lower()
    cue_terms = list(config.get("sort_cues", []))
    cue_terms.extend(config.get("time_fields", {}).get("quick_sort_terms", []))
    cue_positions = [
        match.start()
        for term in cue_terms
        for match in _literal_term_matches(text_lower, str(term))
    ]
    if not cue_positions:
        return []

    candidates: list[tuple[int, int, int, int, str, tuple[int, int]]] = []
    for field, terms in (config.get("sort_terms") or {}).items():
        for term in terms:
            for match in _literal_term_matches(text_lower, str(term)):
                distance = min(abs(match.start() - cue) for cue in cue_positions)
                candidates.append((
                    distance // 16,
                    -len(str(term).split()),
                    -len(str(term)),
                    distance,
                    str(field),
                    match.span(),
                ))
    if not candidates:
        return []

    candidates.sort()
    best_field = candidates[0][4]
    return [
        span for *_score, field, span in candidates
        if field == best_field and _sort_span_is_close_to_cue(span, cue_positions)
    ]


def _literal_term_matches(text: str, term: str) -> list[re.Match]:
    tokens = re.findall(r"[a-z0-9]+", term.lower())
    if not tokens:
        return []
    pattern = r"\b" + r"[\s-]+".join(re.escape(token) for token in tokens) + r"\b"
    return list(re.finditer(pattern, text, re.IGNORECASE))


def _sort_span_is_close_to_cue(span: tuple[int, int], cue_positions: list[int]) -> bool:
    return min(
        min(abs(span[0] - cue), abs(span[1] - cue))
        for cue in cue_positions
    ) <= 48


def _filter_actions_claimed_by_measurements(actions: list[dict], text: str) -> list[dict]:
    measurement_spans = _measurement_claimed_spans(text)
    if not measurement_spans:
        return actions

    filtered = []
    for action in actions:
        span = tuple(action.get("span") or ())
        if len(span) == 2 and any(_span_overlaps(span, claimed) for claimed in measurement_spans):
            continue
        filtered.append(action)
    return filtered


def _span_overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def _without_ingredient_collisions(
    matches: list[dict],
    ingredient_spans: list[dict],
    entity_kind: str,
) -> list[dict]:
    """Resolve exact overlaps by their CSV frequencies before span masking."""
    if not ingredient_spans:
        return matches

    def entity_frequency(match: dict) -> int:
        if entity_kind == "utensil":
            return int(DB_UTENSILS.get(str(match.get("canonical") or ""), 0))
        action = str(match.get("action") or "")
        return int(DB_ACTIONS.get(action, {}).get("frequency", 0))

    accepted = []
    for match in matches:
        layer = str(match.get("layer") or match.get("_layer") or "")
        overlaps = [
            ingredient for ingredient in ingredient_spans
            if _span_overlaps(tuple(match["span"]), tuple(ingredient["span"]))
        ]
        if not overlaps:
            accepted.append(match)
            continue
        if layer not in {"exact_aho", "exact_direct", "exact_morphology"}:
            continue
        if all(entity_frequency(match) > int(ingredient.get("frequency", 0)) for ingredient in overlaps):
            accepted.append(match)
    return accepted


_ACTION_CONTEXT_RE = re.compile(
    r"\b(?:instruction|instructions|step|steps|process|method|procedure|procedures)\b.{0,28}$|"
    r"\b(?:mention|mentions|mentioned|says?|include|includes|including)\s+$",
    re.IGNORECASE,
)
_UTENSIL_CONTEXT_RE = re.compile(
    r"\b(?:made|cooked|prepared|baked|fried|served)\s+(?:in|with|using)\s+(?:a|an|the)?\s*$|"
    r"\b(?:use|uses|using|need|needs|require|requires|with|in)\s+(?:a|an|the)?\s*$|"
    r"\b(?:utensil|utensils|tool|tools|equipment)\s+(?:is|are|as|like)?\s*$",
    re.IGNORECASE,
)


def _span_context(text: str, span: tuple[int, int], window: int = 56) -> tuple[str, str]:
    start, end = span
    return (
        text[max(0, start - window):start].lower(),
        text[end + 1:min(len(text), end + 1 + window)].lower(),
    )


def _has_action_context(text: str, span: tuple[int, int]) -> bool:
    left, right = _span_context(text, span)
    return bool(
        _ACTION_CONTEXT_RE.search(left)
        or re.match(r"\s+(?:in|inside)?\s*(?:the\s+)?(?:instruction|instructions|step|steps)\b", right)
    )


def _has_utensil_context(text: str, span: tuple[int, int]) -> bool:
    left, _right = _span_context(text, span)
    return bool(_UTENSIL_CONTEXT_RE.search(left))


def _resolve_utensil_action_collisions(
    text: str,
    utensils: list[dict],
    actions: list[dict],
) -> tuple[list[dict], list[dict]]:
    if not utensils or not actions:
        return utensils, actions

    drop_utensils: set[int] = set()
    drop_actions: set[int] = set()
    for utensil_index, utensil in enumerate(utensils):
        utensil_span = tuple(utensil.get("span") or ())
        if len(utensil_span) != 2:
            continue
        utensil_confidence = float(utensil.get("confidence", 1.0))
        utensil_layer = str(utensil.get("layer") or "")
        utensil_exact = utensil_layer.startswith("exact")
        utensil_context = _has_utensil_context(text, utensil_span)
        action_context = _has_action_context(text, utensil_span)

        for action_index, action in enumerate(actions):
            action_span = tuple(action.get("span") or ())
            if len(action_span) != 2 or not _span_overlaps(utensil_span, action_span):
                continue
            action_confidence = float(action.get("confidence", 1.0))
            action_layer = str(action.get("_layer") or "")
            action_exact = action_layer.startswith("exact")

            if action_context and not utensil_context:
                drop_utensils.add(utensil_index)
                continue
            if utensil_context and not action_context:
                drop_actions.add(action_index)
                continue
            if utensil_exact and not action_exact:
                drop_actions.add(action_index)
                continue
            if action_exact and not utensil_exact:
                drop_utensils.add(utensil_index)
                continue
            if action_confidence > utensil_confidence + 0.10:
                drop_utensils.add(utensil_index)
            elif utensil_confidence > action_confidence + 0.10:
                drop_actions.add(action_index)

    return (
        [item for index, item in enumerate(utensils) if index not in drop_utensils],
        [item for index, item in enumerate(actions) if index not in drop_actions],
    )


def _public_action_match(match: dict) -> dict:
    return {
        "action": match["action"],
        "type": match["type"],
        "span": match["span"],
        "confidence": match.get("confidence", 1.0),
    }


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
_DIETARY_NEGATION_PREFIX_RE = re.compile(
    r"(?:^|\s)(?:"
    r"(?:do|does|did|should|must|can|could|would|is|are|was|were)?\s*not"
    r"(?:\s+(?:be|being|equal\s+to|equal|equals|contain|contains|containing|include|includes|including|have|has|use|using))?"
    r"|no|without|exclude|excluding|avoid|minus|non"
    r")\s*(?:a|an|the|any)?\s*$",
    re.IGNORECASE,
)


def _is_known_ingredient_term(text: str) -> bool:
    term = text.lower().strip()
    canonical = canonicalize_ingredient(term)
    return term in DB_INGREDIENTS or canonical in DB_INGREDIENTS


def _is_negated_dietary_match(text_lower: str, span: tuple[int, int]) -> bool:
    start, _ = span
    prefix = text_lower[max(0, start - 96):start]
    local_clause = re.split(r"\b(?:but|and|or|then)\b|[,;]", prefix)[-1]
    local_clause = re.sub(r"\s+", " ", local_clause).strip()
    return bool(_DIETARY_NEGATION_PREFIX_RE.search(local_clause))


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
                "negated": _is_negated_dietary_match(text_lower, (m.start(), m.end() - 1)),
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
                "negated": _is_negated_dietary_match(text_lower, (start, end)),
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


@lru_cache(maxsize=1)
def _nutrition_qualifier_patterns() -> tuple[re.Pattern, ...]:
    patterns = []
    for term in sorted(_nutrition_qualifier_terms(), key=len, reverse=True):
        tokens = re.findall(r"[a-z0-9]+", term.lower())
        if not tokens:
            continue
        pattern = r"\b" + r"[\s-]+".join(re.escape(token) for token in tokens) + r"\b"
        patterns.append(re.compile(pattern, re.IGNORECASE))
    return tuple(patterns)


def _nutrition_qualifier_match_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    used: set[int] = set()
    for pattern in _nutrition_qualifier_patterns():
        for match in pattern.finditer(text.lower()):
            positions = set(range(match.start(), match.end()))
            if positions & used:
                continue
            used |= positions
            spans.append(match.span())
    return spans


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
            "excluded_dietary_styles": list[str] - negated dietary styles
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
    dietary_styles = list(dict.fromkeys(
        m["style"] for m in dietary_matches if not m.get("negated")
    ))
    excluded_dietary_styles = list(dict.fromkeys(
        m["style"] for m in dietary_matches if m.get("negated")
    ))
    nutrition_filters = detected["nutrition"]
    sql_operators = detected["sql_ops"]
    strong_ingredient_spans = extract_strong_ingredient_spans(text)
    action_matches = _filter_actions_claimed_by_sql_ops(detected["actions"], sql_operators)
    action_matches = _filter_actions_claimed_by_measurements(action_matches, text)
    action_matches = _without_ingredient_collisions(
        action_matches,
        strong_ingredient_spans,
        "action",
    )
    categories = detected["categories"]
    utensil_matches = _without_ingredient_collisions(
        detected["utensils"],
        strong_ingredient_spans,
        "utensil",
    )
    utensil_matches, action_matches = _resolve_utensil_action_collisions(
        text,
        utensil_matches,
        action_matches,
    )

    # Mask already-claimed spans so fuzzy ingredient matching cannot reinterpret
    # query operators, nutrition terms, dietary labels, tools, or actions as food.
    claimed_spans = [m["span"] for m in utensil_matches]
    claimed_spans.extend(_sql_operator_claimed_spans(sql_operators))
    claimed_spans.extend(_sort_field_claimed_spans(text))
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
        "actions":           [_public_action_match(match) for match in action_matches],
        "dietary_styles":    dietary_styles,
        "excluded_dietary_styles": excluded_dietary_styles,
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
