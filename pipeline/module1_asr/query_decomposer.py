import re
from rapidfuzz import fuzz, process


OR_CUES = {
    "or", "ya",
}

OR_SCOPE_PATTERNS = [
    re.compile(r"\b(?:either|any of|one of|any one of|choose between)\b"),
]

EXCLUDE_PATTERNS = [
    re.compile(r"\b(?:without|except|exclude|excluding|avoid|minus|no)\b"),
    re.compile(r"\bfree\s+of\b"),
]

INCLUDE_RESET = re.compile(
    r"\b(?:with|containing|contains|contain|including|include|using|use|has|have)\b"
)

_NEGATION_TERMS = {"not", "without", "no"}
_RELATION_TERMS = [
    "contain", "contains", "containing",
    "include", "includes", "including",
    "use", "using", "has", "have",
]


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _is_or_scope(text: str, first_start: int) -> bool:
    prefix = text[:first_start].lower()
    return any(pattern.search(prefix) for pattern in OR_SCOPE_PATTERNS)


def _has_exclude_cue(segment: str) -> bool:
    return any(pattern.search(segment) for pattern in EXCLUDE_PATTERNS)


def _has_negated_relation(segment: str) -> bool:
    tokens = re.findall(r"[a-z]+", segment.lower())
    for i, token in enumerate(tokens):
        if token not in _NEGATION_TERMS:
            continue
        after = tokens[i + 1:]
        if not after:
            return True
        phrase = " ".join(after[-2:])
        candidates = after + ([phrase] if phrase else [])
        for candidate in candidates:
            if process.extractOne(
                candidate,
                _RELATION_TERMS,
                scorer=fuzz.WRatio,
                score_cutoff=78,
            ):
                return True
    return False


def _connector_between(text: str, left_end: int, right_start: int) -> str:
    segment = _normalize_space(text[left_end + 1:right_start])
    tokens = set(re.findall(r"[a-z]+", segment))
    if tokens & OR_CUES:
        return "OR"
    return "AND"


def _is_excluded(text: str, match: dict, prior_excluded: bool) -> bool:
    start, _ = match["span"]
    before = text[:start].lower()
    window = before[-48:]

    if _has_exclude_cue(window) or _has_negated_relation(window):
        return True

    if prior_excluded:
        last_piece = before.rsplit(",", 1)[-1]
        if INCLUDE_RESET.search(last_piece):
            return False
        return True

    return False


def _dedupe_matches(matches: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for match in sorted(matches, key=lambda item: item["span"][0]):
        canonical = match["canonical"]
        if canonical in seen:
            continue
        seen.add(canonical)
        result.append(match)
    return result


def _remove_redundant_composites(matches: list[dict]) -> list[dict]:
    result = []
    for match in matches:
        start, end = match["span"]
        children = [
            other for other in matches
            if other is not match
            and other["canonical"] != match["canonical"]
            and other["span"][0] >= start
            and other["span"][1] <= end
        ]
        if len(children) >= 2:
            continue
        result.append(match)
    return result


def decompose_ingredient_logic(text: str, ingredient_matches: list[dict]) -> dict:
    """
    Convert extracted ingredient spans into SQL-ready boolean intent.

    Defaults are intentionally conservative:
    - multiple included ingredients mean AND, including implicit lists like
      "tomato onion garlic";
    - "or", "either", and "any of" switch includes to OR;
    - "without/no/exclude" moves ingredients to exclude.
    """
    useful_matches = _remove_redundant_composites(_dedupe_matches(ingredient_matches))

    include_matches = []
    exclude_matches = []
    excluding = False

    for match in useful_matches:
        excluded = _is_excluded(text, match, excluding)
        if excluded:
            exclude_matches.append(match)
            excluding = True
        else:
            include_matches.append(match)
            excluding = False

    include = [match["canonical"] for match in include_matches]
    exclude = [match["canonical"] for match in exclude_matches]

    connectors = []
    for left, right in zip(include_matches, include_matches[1:]):
        connectors.append({
            "left": left["canonical"],
            "right": right["canonical"],
            "operator": _connector_between(text, left["span"][1], right["span"][0]),
        })

    include_operator = "AND"
    if len(include_matches) > 1:
        if _is_or_scope(text, include_matches[0]["span"][0]):
            include_operator = "OR"
            for connector in connectors:
                connector["operator"] = "OR"
        else:
            operators = {connector["operator"] for connector in connectors}
            include_operator = operators.pop() if len(operators) == 1 else "MIXED"

    return {
        "include": include,
        "exclude": exclude,
        "include_operator": include_operator,
        "exclude_operator": "AND",
        "connectors": connectors,
    }
