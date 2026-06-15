import atexit
import contextlib
import os
import re
from collections import defaultdict
from pathlib import Path

from metaphone import doublemetaphone
from Levenshtein import distance as levenshtein_distance
from rapidfuzz import process, fuzz

from speech_to_sql.m2_nlu.automaton import INGREDIENT_AUTOMATON, UTENSIL_AUTOMATON
from speech_to_sql.m2_nlu.data.db_loader import DB_INGREDIENTS, DB_UTENSILS
from speech_to_sql.m2_nlu.data.ingredient_synonyms import SYNONYMS, canonicalize_ingredient

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

WORD_BOUNDARIES = set(" ,.()\n\t!?;:'\"")

UNIGRAM_THRESHOLD  = 85
BIGRAM_THRESHOLD   = 86
TRIGRAM_THRESHOLD  = 82

GREY_ZONE_LOW   = 78
GREY_ZONE_RATIO = 0.35

METAPHONE_FUZZY_THRESHOLD = 70

VALID_POS = {"NOUN", "PROPN", "ADJ", "X"}

# Semantic embeddings are useful for exploratory matching, but they are too
# expensive and too broad for the default entity extraction hot path.
ENABLE_SEMANTIC_FALLBACK = False

# Lower number = higher priority = wins when spans overlap.
LAYER_PRIORITY = {
    "exact_aho":              0,
    "exact_direct":           1,
    "exact_morphology":       2,
    "metaphone_fuzzy":        2,
    "metaphone_edit":         3,
    "neighbor_fuzzy":         4,
    "semantic":               5,
    "metaphone_fuzzy_rescue": 6,
    "metaphone_edit_rescue":  7,
    "neighbor_fuzzy_rescue":  8,
    "semantic_rescue":        9,
    "exact_direct_rescue":    10,
    "exact_morphology_rescue": 10,
}

COMMON_WORDS_BLACKLIST = {
    # -- nutrition & measurement words ----------------------
    # These are caught by _detect_nutrition_filters() already.
    # Letting them into ingredient matching causes false positives
    # like "caloriiiii" -> "tuna" via phonetic rescue pass.
    "calorie", "calories", "cal", "kcal",
    "protein", "proteins",
    "fat", "fats",
    "carb", "carbs", "carbohydrate", "carbohydrates",
    "fiber", "fibre", "sodium",
    "gram", "grams", "milligram", "milligrams",
    "under", "below", "above", "least", "most",
    # -- general --------------------------------------------
    "food", "good", "cook", "heat", "bake", "boil", "fry",
    "dish", "meal", "prep", "step", "stir", "pour", "chop",
    "give","tell", "show", "add", "put", "cut", "slice", "serve",
    "contain", "include", "use", "need", "want", "like", "make", "take",
    "items", "ingredients", "utensils", "tools", "equipment",
    "food","those", "these", "any", "all", "some", "many", "few", "several",
    "me","and", "or", "but", "if", "then", "when", "while", "before", "after",
    "dice", "mince", "mix",  "fold", "cool", "rest", "toss",
    "make", "take", "just", "some", "this", "that", "with",
    "from", "have", "will", "need", "want", "like", "used",
    "then", "when", "also", "into", "onto", "over", "done",
    "well", "very", "more", "less", "much", "many",
    "each", "both", "such", "most", "same", "next", "last",
    "first", "second", "third","the","a", "an", "in", "on", "at", "by", "for", "of", "to",
    "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "half", "quarter",
    "a", "an", "the", "this", "that", "these", "those",
    "singh",
    "cups", "tbsp", "tsp",  "gram", "grams", "ounce", "pound",
    "piece", "pieces", "slice", "slices", "bunch", "pinch",
}

# Maps a first-character to phonetically similar characters.
# Used by neighbor_candidates() when metaphone yields no pool.
PHONETIC_NEIGHBORS = {
    "b": ["b", "m", "p"],
    "m": ["m", "b", "p", "n"],
    "p": ["p", "b", "m"],
    "n": ["n", "m"],
    "g": ["g", "k", "h", "j"],
    "j": ["j", "z", "y", "g"],
    "k": ["k", "g", "h", "c"],
    "h": ["h", "g", "k", "j"],
    "s": ["s", "z", "c"],
    "z": ["z", "s", "j"],
    "d": ["d", "t"],
    "t": ["t", "d"],
    "f": ["f", "v", "p"],
    "v": ["v", "w", "b"],
    "w": ["w", "v"],
    "r": ["r", "l"],
    "l": ["l", "r"],
    "c": ["c", "k", "s", "q"],
    "a": ["a", "e", "h"],
    "e": ["e", "a", "i", "h"],
    "o": ["o", "u", "h"],
    "i": ["i", "e", "y"],
    "u": ["u", "o", "i"],
    "y": ["y", "i", "j"],
    "x": ["x", "z", "s"],
    "q": ["q", "k"],
}


# ---------------------------------------------------------
# MODELS & INDEXES
# ---------------------------------------------------------

_nlp = None
_g2p_instance = None
_SPACY_STOPS = None
_BG_STOPS = None

# -- g2p model (loaded once at import time) ----------------
# G2p() is expensive to initialise (~1-2 s). Loading it here
# means the cost is paid once, not per query.
def _get_nlp():
    """Lazy-load spaCy only when ingredient fuzzy extraction needs POS tags."""
    global _nlp, _SPACY_STOPS, _BG_STOPS
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        _SPACY_STOPS = _nlp.Defaults.stop_words
        _BG_STOPS = _nlp.Defaults.stop_words | COMMON_WORDS_BLACKLIST
    return _nlp


def _get_spacy_stops() -> set:
    _get_nlp()
    return _SPACY_STOPS or set()


def _get_bigram_stops() -> set:
    _get_nlp()
    return _BG_STOPS or COMMON_WORDS_BLACKLIST


def _get_g2p():
    """Lazy-load g2p only when phoneme re-ranking is actually reached."""
    global _g2p_instance
    if _g2p_instance is None:
        from g2p_en import G2p

        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                _g2p_instance = G2p()
    return _g2p_instance

ALL_KNOWN_TERMS = list(set(
    [x.lower().strip() for x in DB_INGREDIENTS] +
    [x.lower().strip() for x in SYNONYMS.keys()]
))
ALL_KNOWN_TERMS_SET = set(ALL_KNOWN_TERMS)

DB_INGREDIENTS_LOWER = set(x.lower().strip() for x in DB_INGREDIENTS)

# -- Stop word sets (computed once at module level) --------
# Computing these inside extract_ingredients() on every call
# created a new set object per query. Moved here so they are
# built exactly once at import time.
PREFIX_INDEX: dict = defaultdict(list)
for _term in ALL_KNOWN_TERMS:
    if _term:
        PREFIX_INDEX[_term[0]].append(_term)

METAPHONE_INDEX: dict = defaultdict(list)
for _term in ALL_KNOWN_TERMS:
    if not _term:
        continue
    _primary, _secondary = doublemetaphone(_term)
    if _primary:
        METAPHONE_INDEX[_primary].append(_term)
    if _secondary:
        METAPHONE_INDEX[_secondary].append(_term)


# ---------------------------------------------------------
# ASR CORRECTIONS DICTIONARY
# ---------------------------------------------------------

_RECIPE_CONTEXT_WORDS = {
    "recipe", "cook", "add", "use", "need", "tablespoon", "teaspoon",
    "cup", "cups", "mix", "stir", "ingredient", "pinch", "dash",
    "fresh", "dried", "ground", "whole", "chopped", "minced",
    "bake", "roast", "fry", "boil", "simmer", "grill",
}

_RISKY_CORRECTIONS: dict = {
    r"\bflower\b":  "flour",
    r"\btime\b":    "thyme",
    r"\bsalary\b":  "celery",
    r"\bleak\b":    "leek",
    r"\bplumb\b":   "plum",
}

# -- CHANGE: Added phonetically distant ASR variants -------
#
# WHY: Some ASR transcriptions are too far from the canonical
# form for any phonetic algorithm (metaphone, g2p) to bridge.
# "kepsikam" and "kepsecum" share no metaphone code with
# "capsicum", and g2p distance is too large to pass the
# consonant-skeleton threshold. These are unambiguous - no
# English word resembles them - so a direct safe correction
# is the right fix.
#
# Add more entries here as new ASR failure patterns are found.
# ----------------------------------------------------------
_SAFE_CORRECTIONS: dict = {
    r"\bchick\b": "chicken",
    r"\bkepsikam\b": "capsicum",
    r"\bkepsecum\b": "capsicum",
    r"\bkapsikam\b": "capsicum",
    r"\bkapsicum\b": "capsicum",
}


def asr_normalize(text: str) -> str:
    result = text

    for pattern, replacement in _SAFE_CORRECTIONS.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    tokens = set(result.lower().split())
    if tokens & _RECIPE_CONTEXT_WORDS:
        for pattern, replacement in _RISKY_CORRECTIONS.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


# ---------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text


# ---------------------------------------------------------
# POS FILTERING
# ---------------------------------------------------------

def get_pos_filtered_tokens(text: str) -> list:
    doc = _get_nlp()(text)
    return [
        {
            "text":  token.text.lower(),
            "start": token.idx,
            "end":   token.idx + len(token.text) - 1,
        }
        for token in doc
        if token.pos_ in VALID_POS
    ]


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def valid_candidate(token: str) -> bool:
    token = token.strip()
    if len(token) < 3:
        return False
    compact = token.replace(" ", "").replace("-", "")
    if not compact.isalpha():
        return False
    if sum(c in "aeiou" for c in compact) == 0:
        return False
    return True


# ---------------------------------------------------------
# OVERLAP RESOLUTION
# ---------------------------------------------------------

def _is_blocked_ingredient(text: str, canonical: str) -> bool:
    text = text.lower().strip()
    canonical = canonical.lower().strip()
    spacy_stops = _get_spacy_stops()
    return (
        text.startswith("no ") or
        canonical.startswith("no ") or
        text in COMMON_WORDS_BLACKLIST or
        canonical in COMMON_WORDS_BLACKLIST or
        text in spacy_stops or
        canonical in spacy_stops
    )


def add_exact_component_matches(exact_matches: list, raw_matches: list) -> list:
    """
    Keep atomic ingredients nested inside DB composite phrases.
    RecipeDB contains product/dish terms like "chicken soup" and
    "garlic butter"; retrieval still needs "chicken", "garlic",
    and "butter" as independent entities.
    """
    enriched = list(exact_matches)
    seen = {(m["span"], m["canonical"]) for m in enriched}

    for parent in exact_matches:
        p_start, p_end = parent["span"]
        child_candidates = [
            raw for raw in raw_matches
            if raw is not parent
            and raw["span"] != parent["span"]
            and raw["span"][0] >= p_start
            and raw["span"][1] <= p_end
            and raw["canonical"] != parent["canonical"]
            and len(raw["text"].split()) == 1
            and not _is_blocked_ingredient(raw["text"], raw["canonical"])
        ]
        if not _should_decompose_exact_match(parent, child_candidates):
            continue

        for raw in child_candidates:
            key = (raw["span"], raw["canonical"])
            if key in seen:
                continue
            component = dict(raw)
            component["layer"] = "exact_component"
            enriched.append(component)
            seen.add(key)

    return enriched


def _should_decompose_exact_match(parent: dict, children: list[dict]) -> bool:
    if not children:
        return False

    parent_meta = DB_INGREDIENTS.get(parent["canonical"])
    if not parent_meta:
        return False

    if parent_meta["category"].lower() == "dish":
        return True

    if parent_meta["frequency"] <= 9:
        return True

    if len(children) < 2:
        return False

    return any(
        (DB_INGREDIENTS.get(child["canonical"], {}).get("category", "").lower() == "dish")
        for child in children
    )


def resolve_overlaps(matches: list) -> list:
    def _sort_key(x):
        score = x.get("score", 0)
        if isinstance(score, float) and score <= 1.0:
            score = score * 100
        return (
            x["span"][1] - x["span"][0],
            -LAYER_PRIORITY.get(x.get("layer", "semantic"), 99),
            score,
        )

    matches.sort(key=_sort_key, reverse=True)
    accepted = []
    used = set()
    for m in matches:
        start, end = m["span"]
        positions = set(range(start, end + 1))
        if not positions & used:
            accepted.append(m)
            used |= positions
    return accepted


# ---------------------------------------------------------
# DEDUPLICATION
# ---------------------------------------------------------

def deduplicate(matches: list) -> list:
    seen  = set()
    final = []
    for m in matches:
        if m["canonical"] not in seen:
            final.append(m)
            seen.add(m["canonical"])
    return final


# ---------------------------------------------------------
# EDIT DISTANCE CONFIRMATION
# ---------------------------------------------------------

def passes_edit_distance(query: str, matched_term: str) -> bool:
    edit_dist = levenshtein_distance(query, matched_term)
    word_len  = max(len(query), len(matched_term))
    max_edits = 2 if word_len <= 6 else 3
    return (
        edit_dist <= max_edits and
        (edit_dist / word_len) <= GREY_ZONE_RATIO
    )


def _edit_ratio(query: str, matched_term: str) -> float:
    return levenshtein_distance(query, matched_term) / max(len(query), len(matched_term), 1)


def _is_plausible_fuzzy_match(query: str, matched_term: str, score: float, threshold: int) -> bool:
    if passes_edit_distance(query, matched_term):
        return True
    if score < threshold:
        return False
    # High WRatio can over-reward short phonetic collisions such as
    # "leechee" vs "leek"; edit ratio keeps the candidate structurally close.
    max_ratio = 0.42 if max(len(query), len(matched_term)) <= 8 else 0.36
    return _edit_ratio(query, matched_term) <= max_ratio


def _best_fuzzy_candidate(query: str, candidates: list, scorer, threshold: int):
    results = process.extract(
        query,
        candidates,
        scorer=scorer,
        score_cutoff=GREY_ZONE_LOW,
        limit=12,
    )
    plausible = [
        (term, score, index)
        for term, score, index in results
        if _is_plausible_fuzzy_match(query, term, score, threshold)
    ]
    if not plausible:
        return None
    return min(
        plausible,
        key=lambda item: (
            _edit_ratio(query, item[0]),
            -fuzz.ratio(query, item[0]),
            -item[1],
            abs(len(query) - len(item[0])),
        ),
    )


# ---------------------------------------------------------
# G2P PHONEME HELPERS
#
# WHY THESE EXIST:
#   g2p-en converts a word to its ARPAbet phoneme sequence,
#   e.g. "quinoa" -> ["K", "IY1", "N", "W", "AA2"].
#   We strip stress digits (IY1 -> IY) so that only the
#   phoneme identity matters, not emphasis.
#
#   This is used ONLY inside _metaphone_rerank as a second-pass
#   scorer on the (already small) metaphone candidate bucket.
#   Because the bucket is small (typically 2-15 terms), the
#   per-call cost of g2p is negligible - unlike running it
#   across the entire vocabulary, which would be O(N) per query.
#
#   PHONEME_CACHE avoids recomputing g2p for the same word
#   twice. DB terms are pre-warmed at import time so the
#   first real query never waits.
# ---------------------------------------------------------

PHONEME_CACHE: dict[str, str] = {}


def _normalize_phonemes(phonemes: list) -> list:
    """Strip stress digits and keep only alphabetic ARPAbet tokens."""
    cleaned = []
    for p in phonemes:
        p = re.sub(r"\d", "", p)   # "IY1" -> "IY"
        if p.isalpha():
            cleaned.append(p)
    return cleaned


def word_to_phoneme_key(word: str) -> str:
    """
    Convert a word to a space-joined, stress-free phoneme string.
    Results are cached so each unique surface form is processed
    by g2p at most once per process lifetime.

    Examples:
        "quinoa"  -> "K IY N W AA"
        "keenwah" -> "K IY N W AA"   ← identical -> ed = 0  Y
        "ginger"  -> "JH IH N JH ER"
        "jinjer"  -> "JH IH N JH ER" ← identical -> ed = 0  Y
        "injera"  -> "IH N JH EH R AH" ← distance 4        ✗
    """
    if word in PHONEME_CACHE:
        return PHONEME_CACHE[word]
    phonemes = _normalize_phonemes(_get_g2p()(word))
    key = " ".join(phonemes)
    PHONEME_CACHE[word] = key
    return key


# -- Phoneme disk cache ------------------------------------
# g2p pre-warming ~5000 terms at import takes 20-40s on CPU
# because g2p_en loads a neural model. We persist the cache
# to disk so the cost is paid ONCE, not every restart.
_PHONEME_CACHE_FILE = Path(__file__).parent / "data" / "_cache" / "phoneme_cache.json"

def _load_phoneme_cache():
    """Load phoneme cache from disk if it exists."""
    if _PHONEME_CACHE_FILE.exists():
        try:
            import json
            with open(_PHONEME_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_phoneme_cache():
    """Persist phoneme cache to disk."""
    import json
    _PHONEME_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_PHONEME_CACHE_FILE, "w") as f:
        json.dump(PHONEME_CACHE, f)

# Load from disk on startup. New entries are computed lazily inside
# word_to_phoneme_key(); pre-warming the full DB makes demo startup hang.
PHONEME_CACHE.update(_load_phoneme_cache())
atexit.register(_save_phoneme_cache)


# ---------------------------------------------------------
# STAGE 1 - METAPHONE RETRIEVAL
# ---------------------------------------------------------

def metaphone_candidates(query: str) -> list:
    pool = set()
    primary, secondary = doublemetaphone(query)
    if primary:
        pool.update(METAPHONE_INDEX.get(primary, []))
    if secondary:
        pool.update(METAPHONE_INDEX.get(secondary, []))
    return list(pool)


# ---------------------------------------------------------
# STAGE 2 - PHONETIC NEIGHBOR RETRIEVAL
# Only reached when metaphone produces no candidates.
# ---------------------------------------------------------

def neighbor_candidates(query: str) -> list:
    pool = set()

    _TWO_CHAR_PHONEME_MAP = {
        "ch": ["c", "s", "k"],
        "sh": ["s", "c", "ch"],
        "th": ["t", "d"],
        "kn": ["n"],
        "wh": ["w"],
        "ph": ["f"],
    }
    prefix2 = query[:2].lower() if len(query) >= 2 else ""
    if prefix2 in _TWO_CHAR_PHONEME_MAP:
        for mapped_char in _TWO_CHAR_PHONEME_MAP[prefix2]:
            pool.update(PREFIX_INDEX.get(mapped_char, []))

    first = query[0]
    for key in PHONETIC_NEIGHBORS.get(first, [first]):
        pool.update(PREFIX_INDEX.get(key, []))

    return list(pool)


# ---------------------------------------------------------
# ASR NORMALIZATION HELPERS  (vowel / double-consonant)
# ---------------------------------------------------------

def _vowel_normalize(word: str) -> str:
    return re.sub(r"[aeiou]", "A", word.lower())


def _double_consonant_normalize(word: str) -> str:
    return re.sub(r"(.)\1+", r"\1", word)


def _asr_form(word: str) -> str:
    return _vowel_normalize(_double_consonant_normalize(word.lower()))


def _last_token_variants(term: str) -> list[str]:
    parts = term.split()
    if not parts:
        return []

    token = parts[-1]
    variants = []

    def add_last(value: str) -> None:
        if not value or value == token:
            return
        candidate = " ".join([*parts[:-1], value])
        if candidate not in variants:
            variants.append(candidate)

    if token.endswith("ies") and len(token) > 4:
        add_last(token[:-3] + "y")
    if re.search(r"(oes|ches|shes|xes|zes|ses)$", token) and len(token) > 4:
        add_last(token[:-2])
    if token.endswith("ves") and len(token) > 4:
        add_last(token[:-3] + "f")
        add_last(token[:-3] + "fe")
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        add_last(token[:-1])

    return variants


def _known_morphology_match(query: str) -> str | None:
    for candidate in _last_token_variants(query):
        if candidate in ALL_KNOWN_TERMS_SET:
            return candidate
    return None


# ---------------------------------------------------------
# FUZZY SEARCH (strict pass + grey-zone pass + ASR rescue)
# ---------------------------------------------------------

def fuzzy_search(
    query:      str,
    candidates: list,
    threshold:  int,
    mode:       str = "unigram",
):
    if not candidates:
        return None

    scorer = fuzz.WRatio if mode == "unigram" else fuzz.token_set_ratio

    result = _best_fuzzy_candidate(query, candidates, scorer, threshold)
    if result:
        return result

    query_asr = _asr_form(query)
    best_asr_score     = 0
    best_asr_candidate = None
    for cand in candidates:
        asr_score = fuzz.ratio(query_asr, _asr_form(cand))
        if asr_score > best_asr_score:
            best_asr_score     = asr_score
            best_asr_candidate = cand
    if best_asr_candidate and best_asr_score >= GREY_ZONE_LOW:
        if passes_edit_distance(query, best_asr_candidate):
            raw_score = fuzz.WRatio(query, best_asr_candidate)
            return (best_asr_candidate, raw_score, 0)

    return None


# ---------------------------------------------------------
# METAPHONE BEST - raw edit distance last resort
# ---------------------------------------------------------

def metaphone_best(query: str, candidates: list) -> str | None:
    if not candidates:
        return None
    db_candidates = [c for c in candidates if c in DB_INGREDIENTS_LOWER]
    pool = db_candidates if db_candidates else candidates
    best = min(pool, key=lambda c: levenshtein_distance(query, c))
    edit_dist = levenshtein_distance(query, best)
    word_len  = max(len(query), len(best))
    if (edit_dist / word_len) > 0.4:
        return None
    return best


# ---------------------------------------------------------
# VOWEL-STRIPPED HELPERS  (used as tiebreaker inside rerank)
# ---------------------------------------------------------

def _strip_vowels(s: str) -> str:
    return re.sub(r"[aeiou]", "", s)


def _count_vowels(s: str) -> int:
    return sum(c in "aeiou" for c in s)


def _metaphone_rerank(query: str):
    """
    Re-rank the metaphone bucket using:
      1. g2p phoneme edit distance         (PRIMARY)
      2. Consonant skeleton edit distance  (tiebreaker)
      3. Vowel count proximity             (final tiebreaker)

    -- CHANGE --------------------------------------------
    Previously the sort order was:
      (consonant_skeleton, vowel_count, phoneme_distance)

    This caused "jinjer" -> "injera" because both share the
    same metaphone code (ANJR), and "injera" had a lower
    consonant skeleton edit distance than "ginger" (which
    has a different metaphone code JNJR and was not even
    in the bucket).

    Wait - "jinjer" and "ginger" DO land in different
    metaphone buckets. The real fix is that g2p correctly
    maps "jinjer" -> "JH IH N JH ER" which is identical to
    "ginger" -> "JH IH N JH ER" (distance 0), while
    "injera" -> "IH N JH EH R AH" (distance 4).

    By promoting g2p phoneme distance to PRIMARY, we pick
    "ginger" over "injera" even when both share a metaphone
    code with the query. The consonant skeleton becomes a
    tiebreaker only when g2p distances are equal.
    ------------------------------------------------------
    """
    p, s = doublemetaphone(query)
    candidates: list = []
    if p:
        candidates.extend(METAPHONE_INDEX.get(p, []))
    if s:
        candidates.extend(METAPHONE_INDEX.get(s, []))
    candidates = list(set(candidates))

    if not candidates:
        return None

    query_phoneme_key  = word_to_phoneme_key(query)
    query_stripped     = _strip_vowels(query)
    query_vowel_count  = _count_vowels(query)

    ranked = sorted(
        candidates,
        key=lambda c: (
            # 1. g2p phoneme distance - PRIMARY
            #    Most accurate signal: maps ASR surface forms to
            #    their true pronunciation before comparing.
            #    "jinjer" and "ginger" share phoneme key "JH IH N JH ER"
            #    -> distance 0. "injera" -> "IH N JH EH R AH" -> distance 4.
            levenshtein_distance(query_phoneme_key, word_to_phoneme_key(c)),
            # 2. Consonant skeleton - tiebreaker
            #    Reliable for OOV Whisper forms where g2p may guess wrong.
            levenshtein_distance(query_stripped, _strip_vowels(c)),
            # 3. Vowel count proximity - final tiebreaker
            abs(_count_vowels(c) - query_vowel_count),
        ),
    )

    best = ranked[0]
    best_phoneme_ed = levenshtein_distance(
        query_phoneme_key,
        word_to_phoneme_key(best),
    )
    best_consonant_ed = levenshtein_distance(query_stripped, _strip_vowels(best))
    raw_score = fuzz.WRatio(query, best)

    if best_consonant_ed > (1 if len(query) <= 4 else 2):
        return None
    phonetic_close = (
        len(query) >= 5 and
        best_phoneme_ed <= 1 and
        best_consonant_ed <= 1
    )
    if raw_score < 70 and not passes_edit_distance(query, best) and not phonetic_close:
        return None

    return (best, raw_score, 0)


# ---------------------------------------------------------
# MAIN TOKEN RESOLVER
# ---------------------------------------------------------

def resolve_token(query: str, threshold: int, mode: str = "unigram"):

    if query in COMMON_WORDS_BLACKLIST:
        return None

    # -- Stage 0 - Exact direct match ----------------------
    if query in ALL_KNOWN_TERMS_SET:
        return {
            "canonical": canonicalize_ingredient(query),
            "layer":     "exact_direct",
            "score":     100,
        }

    morphology_match = _known_morphology_match(query)
    if morphology_match:
        return {
            "canonical": canonicalize_ingredient(morphology_match),
            "layer":     "exact_morphology",
            "score":     100,
        }

    # -- Stage 1 - Metaphone bucket ------------------------
    meta_candidates = metaphone_candidates(query)

    if meta_candidates:

        # 1a - g2p phoneme re-rank (PRIMARY)
        meta_rerank = _metaphone_rerank(query)
        if meta_rerank:
            matched_term, score, _ = meta_rerank
            return {
                "canonical": canonicalize_ingredient(matched_term),
                "layer":     "metaphone_fuzzy",
                "score":     score,
            }

        # 1b - Character fuzzy fallback
        result = fuzzy_search(query, meta_candidates, METAPHONE_FUZZY_THRESHOLD, mode)
        if result:
            matched_term, score, _ = result
            return {
                "canonical": canonicalize_ingredient(matched_term),
                "layer":     "metaphone_fuzzy",
                "score":     score,
            }

        # 1c - Raw edit distance (last resort inside bucket)
        best = metaphone_best(query, meta_candidates)
        if best:
            edit_dist  = levenshtein_distance(query, best)
            word_len   = max(len(query), len(best))
            real_score = round((1 - edit_dist / word_len) * 100)
            return {
                "canonical": canonicalize_ingredient(best),
                "layer":     "metaphone_edit",
                "score":     real_score,
            }

    # -- Stage 2 - Phonetic neighbor prefix ----------------
    neighbor_pool = neighbor_candidates(query)
    result = fuzzy_search(query, neighbor_pool, threshold, mode)
    if result:
        matched_term, score, _ = result
        return {
            "canonical": canonicalize_ingredient(matched_term),
            "layer":     "neighbor_fuzzy",
            "score":     score,
        }

    # -- Stage 3 - Semantic embedding ----------------------
    if ENABLE_SEMANTIC_FALLBACK and len(query) > 5:
        from speech_to_sql.m2_nlu.semantic_matcher import semantic_match

        semantic_result = semantic_match(query, threshold=0.78)
        if semantic_result:
            return {
                "canonical": canonicalize_ingredient(semantic_result["canonical"]),
                "layer":     "semantic",
                "score":     semantic_result["score"],
            }

    return None


# ---------------------------------------------------------
# EXTRACT INGREDIENTS
# ---------------------------------------------------------

def extract_ingredients(text: str) -> list:

    text        = asr_normalize(text)
    text_lower  = normalize_text(text)
    raw_matches = []

    # -- Layer 1 - Exact Aho-Corasick ----------------------
    for end_idx, (source, original, canonical) in INGREDIENT_AUTOMATON.iter(text_lower):
        start_idx  = end_idx - len(original) + 1
        match_text = text_lower[start_idx: end_idx + 1]
        before = text_lower[start_idx - 1] if start_idx > 0 else " "
        after  = text_lower[end_idx + 1]   if end_idx + 1 < len(text_lower) else " "
        if before in WORD_BOUNDARIES and after in WORD_BOUNDARIES:
            if _is_blocked_ingredient(match_text, canonical):
                continue
            raw_matches.append({
                "text":      match_text,
                "canonical": canonical,
                "span":      (start_idx, end_idx),
                "layer":     "exact_aho",
                "source":    source,
            })

    exact_matches     = add_exact_component_matches(resolve_overlaps(raw_matches), raw_matches)
    matched_positions = set()
    for m in exact_matches:
        matched_positions |= set(range(m["span"][0], m["span"][1] + 1))

    fuzzy_matches = []

    def already_matched(s, e):
        return bool(set(range(s, e + 1)) & matched_positions)

    def append_match(d):
        nonlocal matched_positions
        fuzzy_matches.append(d)
        matched_positions |= set(range(d["span"][0], d["span"][1] + 1))

    pos_tokens = get_pos_filtered_tokens(text_lower)
    spacy_stops = _get_spacy_stops()
    bigram_stops = _get_bigram_stops()

    # -- Unigrams - two-pass --------------------------------

    # Pass 1 - POS-filtered
    for token in pos_tokens:
        s, e, clean = token["start"], token["end"], token["text"]
        if already_matched(s, e) or not valid_candidate(clean):
            continue
        result = resolve_token(clean, UNIGRAM_THRESHOLD, "unigram")
        if result:
            append_match({
                "text":      clean,
                "canonical": result["canonical"],
                "span":      (s, e),
                "layer":     result["layer"],
                "score":     result["score"],
            })

    # Pass 2 - Rescue (all unmatched whitespace tokens)
    # Only runs for words >= 5 chars that are NOT English stop words.
    # Short words (< 5 chars) almost never uniquely identify an
    # ingredient and cause too many false positives with 20k candidates.
    for word_match in re.finditer(r"\w+", text_lower):
        clean = word_match.group()
        s     = word_match.start()
        e     = word_match.end() - 1
        if already_matched(s, e) or not valid_candidate(clean):
            continue
        if clean in COMMON_WORDS_BLACKLIST:
            continue
        if clean in spacy_stops:
            continue
        result = resolve_token(clean, UNIGRAM_THRESHOLD + 5, "unigram")
        if result:
            append_match({
                "text":      clean,
                "canonical": result["canonical"],
                "span":      (s, e),
                "layer":     result["layer"] + "_rescue",
                "score":     result["score"],
            })

    # -- Bigrams -------------------------------------------
    for i in range(len(pos_tokens) - 1):
        t1, t2 = pos_tokens[i], pos_tokens[i + 1]
        if t2["start"] - t1["end"] > 2:
            continue
        s, e   = t1["start"], t2["end"]
        phrase = f"{t1['text']} {t2['text']}"
        # Reject bigrams where either token is a stop/blacklist word
        if t1["text"] in bigram_stops or t2["text"] in bigram_stops:
            continue
        if already_matched(s, e) or not valid_candidate(phrase):
            continue
        result = resolve_token(phrase, BIGRAM_THRESHOLD, "bigram")
        if result:
            append_match({
                "text":      phrase,
                "canonical": result["canonical"],
                "span":      (s, e),
                "layer":     result["layer"],
                "score":     result["score"],
            })

    # -- Trigrams ------------------------------------------
    for i in range(len(pos_tokens) - 2):
        t1, t2, t3 = pos_tokens[i], pos_tokens[i + 1], pos_tokens[i + 2]
        if t2["start"] - t1["end"] > 2 or t3["start"] - t2["end"] > 2:
            continue
        s, e   = t1["start"], t3["end"]
        phrase = f"{t1['text']} {t2['text']} {t3['text']}"
        # Reject trigrams where any token is a stop/blacklist word
        if any(t["text"] in bigram_stops for t in [t1, t2, t3]):
            continue
        if already_matched(s, e) or not valid_candidate(phrase):
            continue
        result = resolve_token(phrase, TRIGRAM_THRESHOLD, "trigram")
        if result:
            append_match({
                "text":      phrase,
                "canonical": result["canonical"],
                "span":      (s, e),
                "layer":     result["layer"],
                "score":     result["score"],
            })

    all_matches = deduplicate(exact_matches + fuzzy_matches)
    return sorted(all_matches, key=lambda x: x["span"][0])


# ---------------------------------------------------------
# EXTRACT UTENSILS
# ---------------------------------------------------------

_QUANTITY_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "half", "quarter", "a", "an",
}
_QUANTITY_PATTERN = re.compile(r"\b(\d+[\./]?\d*)\s+cup", re.IGNORECASE)


def extract_utensils(text: str) -> list:

    text_lower  = normalize_text(text)
    raw_matches = []

    for end_idx, (source, original, canonical) in UTENSIL_AUTOMATON.iter(text_lower):
        start_idx  = end_idx - len(original) + 1
        match_text = text_lower[start_idx: end_idx + 1]
        before = text_lower[start_idx - 1] if start_idx > 0 else " "
        after  = text_lower[end_idx + 1]   if end_idx + 1 < len(text_lower) else " "

        if before not in WORD_BOUNDARIES or after not in WORD_BOUNDARIES:
            continue

        if canonical == "cup":
            preceding = text_lower[:start_idx].strip().split()
            if preceding:
                last_word = preceding[-1].rstrip(".,")
                if last_word in _QUANTITY_WORDS or last_word.replace(".", "").isdigit():
                    continue
            if _QUANTITY_PATTERN.search(text_lower[max(0, start_idx - 10): end_idx + 5]):
                continue

        raw_matches.append({
            "text":      match_text,
            "canonical": canonical,
            "span":      (start_idx, end_idx),
            "layer":     "exact_aho",
        })

    return resolve_overlaps(raw_matches)
