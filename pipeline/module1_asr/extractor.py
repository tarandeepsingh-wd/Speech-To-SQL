import re
from collections import defaultdict

import spacy
from metaphone import doublemetaphone
from Levenshtein import distance as levenshtein_distance
from rapidfuzz import process, fuzz

from module1_asr.automaton import AUTOMATON
from module1_asr.data.db_loader import DB_INGREDIENTS
from module1_asr.data.ingredient_synonyms import SYNONYMS
from module1_asr.semantic_matcher import semantic_match


# =========================================================
# CONSTANTS
# =========================================================

WORD_BOUNDARIES = set(" ,.()\n\t!?;:'\"")

UNIGRAM_THRESHOLD = 85
BIGRAM_THRESHOLD = 86
TRIGRAM_THRESHOLD = 82

GREY_ZONE_LOW = 78
GREY_ZONE_RATIO = 0.35

METAPHONE_FUZZY_THRESHOLD = 70

VALID_POS = {"NOUN", "PROPN", "ADJ", "X"}

# =========================================================
# COMMON WORDS BLACKLIST
# words that phonetically collide with ingredients
# but are never ingredients themselves
# =========================================================

COMMON_WORDS_BLACKLIST = {
    # food-adjacent false positives
    "food", "good", "cook", "heat", "bake", "boil", "fry",
    "dish", "meal", "prep", "step", "stir", "pour", "chop",
    "dice", "mince", "mix", "fold", "cool", "rest", "toss",
    # common english words
    "make", "take", "just", "some", "this", "that", "with",
    "from", "have", "will", "need", "want", "like", "used",
    "then", "when", "also", "into", "onto", "over", "done",
    "well", "very", "more", "less", "much", "many", "some",
    "each", "both", "such", "most", "same", "next", "last",
    "first", "second", "third",
    # units / quantities that may slip through POS filter
    "cups", "tbsp", "tsp", "gram", "grams", "ounce", "pound",
    "piece", "pieces", "slice", "slices", "bunch", "pinch",
}


# =========================================================
# PHONETIC NEIGHBORS
# =========================================================

PHONETIC_NEIGHBORS = {
    "b": ["b", "m", "p"],
    "m": ["m", "b", "p", "n"],
    "p": ["p", "b", "m"],
    "n": ["n", "m"],
    "g": ["g", "k", "h", "j"],
    "j": ["j", "z", "y", "g"],
    "k": ["k", "g", "h", "c"],
    "h": ["h", "g", "k"],
    "s": ["s", "z", "c"],
    "z": ["z", "s", "j"],
    "d": ["d", "t"],
    "t": ["t", "d"],
    "f": ["f", "v", "p"],
    "v": ["v", "w", "b"],
    "w": ["w", "v"],
    "r": ["r", "l"],
    "l": ["l", "r"],
}


# =========================================================
# MODELS & INDEXES
# =========================================================

nlp = spacy.load("en_core_web_sm")

ALL_KNOWN_TERMS = list(set(
    [x.lower().strip() for x in DB_INGREDIENTS] +
    [x.lower().strip() for x in SYNONYMS.keys()]
))

DB_INGREDIENTS_LOWER = set(x.lower().strip() for x in DB_INGREDIENTS)


# =========================================================
# PREFIX INDEX
# =========================================================

PREFIX_INDEX = defaultdict(list)

for term in ALL_KNOWN_TERMS:
    if term:
        PREFIX_INDEX[term[0]].append(term)


# =========================================================
# METAPHONE INDEX
# =========================================================

METAPHONE_INDEX = defaultdict(list)

for term in ALL_KNOWN_TERMS:

    if not term:
        continue

    primary, secondary = doublemetaphone(term)

    if primary:
        METAPHONE_INDEX[primary].append(term)

    if secondary:
        METAPHONE_INDEX[secondary].append(term)


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# POS FILTERING
# =========================================================

def get_pos_filtered_tokens(text: str) -> list:

    doc = nlp(text)

    candidates = []

    for token in doc:

        if token.pos_ in VALID_POS:

            candidates.append({
                "text": token.text.lower(),
                "start": token.idx,
                "end": token.idx + len(token.text) - 1
            })

    return candidates


# =========================================================
# VALIDATION
# =========================================================

def valid_candidate(token: str) -> bool:

    token = token.strip()

    if len(token) < 3:
        return False

    compact = token.replace(" ", "")

    if not compact.isalpha():
        return False

    vowel_count = sum(c in "aeiou" for c in compact)

    if vowel_count == 0:
        return False

    return True


# =========================================================
# OVERLAP RESOLUTION
# =========================================================

def resolve_overlaps(matches: list) -> list:

    matches.sort(
        key=lambda x: (
            x["span"][1] - x["span"][0],
            x.get("score", 100)
        ),
        reverse=True
    )

    accepted = []
    used = set()

    for m in matches:

        start, end = m["span"]

        positions = set(range(start, end + 1))

        if not positions & used:
            accepted.append(m)
            used |= positions

    return accepted


# =========================================================
# DEDUPLICATION
# =========================================================

def deduplicate(matches: list) -> list:

    seen = set()
    final = []

    for m in matches:

        canonical = m["canonical"]

        if canonical not in seen:
            final.append(m)
            seen.add(canonical)

    return final


# =========================================================
# EDIT DISTANCE CONFIRMATION
# =========================================================

def passes_edit_distance(query: str, matched_term: str) -> bool:

    edit_dist = levenshtein_distance(query, matched_term)

    word_len = max(len(query), len(matched_term))

    max_edits = 2 if word_len <= 6 else 3

    return (
        edit_dist <= max_edits and
        (edit_dist / word_len) <= GREY_ZONE_RATIO
    )


# =========================================================
# STAGE 1 — METAPHONE RETRIEVAL
# =========================================================

def metaphone_candidates(query: str) -> list:

    pool = set()

    primary, secondary = doublemetaphone(query)

    if primary:
        pool.update(METAPHONE_INDEX.get(primary, []))

    if secondary:
        pool.update(METAPHONE_INDEX.get(secondary, []))

    return list(pool)


# =========================================================
# STAGE 2 — PHONETIC NEIGHBOR RETRIEVAL
# ONLY USED IF METAPHONE FAILS
# =========================================================

def neighbor_candidates(query: str) -> list:

    pool = set()

    first = query[0]

    neighbor_keys = PHONETIC_NEIGHBORS.get(first, [first])

    for key in neighbor_keys:
        pool.update(PREFIX_INDEX.get(key, []))

    return list(pool)


# =========================================================
# FUZZY SEARCH
# =========================================================

def fuzzy_search(query: str,
                 candidates: list,
                 threshold: int,
                 mode: str = "unigram"):

    if not candidates:
        return None

    scorer = (
        fuzz.WRatio
        if mode == "unigram"
        else fuzz.token_set_ratio
    )

    # -----------------------------------------------------
    # STRICT PASS
    # -----------------------------------------------------

    result = process.extractOne(
        query,
        candidates,
        scorer=scorer,
        score_cutoff=threshold
    )

    if result:
        return result

    # -----------------------------------------------------
    # GREY ZONE PASS
    # -----------------------------------------------------

    grey_result = process.extractOne(
        query,
        candidates,
        scorer=scorer,
        score_cutoff=GREY_ZONE_LOW
    )

    if grey_result:

        matched_term, score, _ = grey_result

        if passes_edit_distance(query, matched_term):
            return grey_result

    return None


# =========================================================
# METAPHONE BEST CANDIDATE
# picks DB ingredient over non-DB, uses edit distance
# to break ties when fuzzy scores are unreliable
# now includes a similarity gate to prevent bad matches
# =========================================================

def metaphone_best(query: str, candidates: list) -> str | None:

    if not candidates:
        return None

    # prefer DB ingredients over synonyms
    db_candidates = [c for c in candidates if c in DB_INGREDIENTS_LOWER]
    pool = db_candidates if db_candidates else candidates

    if len(pool) == 1:
        best = pool[0]
    else:
        # pick by lowest edit distance across full pool
        best = min(pool, key=lambda c: levenshtein_distance(query, c))

    # -------------------------------------------------------
    # SIMILARITY GATE
    # reject if the best candidate is still too different
    # from the query — prevents false positives like
    # "food" → "feta cheese"
    # -------------------------------------------------------

    edit_dist = levenshtein_distance(query, best)
    word_len = max(len(query), len(best))

    if (edit_dist / word_len) > 0.4:
        return None

    return best


# =========================================================
# MAIN TOKEN RESOLVER
# =========================================================

def resolve_token(query: str,
                  threshold: int,
                  mode: str = "unigram"):

    # =====================================================
    # BLACKLIST CHECK
    # reject common English words before any matching
    # =====================================================

    if query in COMMON_WORDS_BLACKLIST:
        return None

    # =====================================================
    # STAGE 0 — DIRECT EXACT
    # =====================================================

    if query in ALL_KNOWN_TERMS:

        canonical = SYNONYMS.get(query, query)

        return {
            "canonical": canonical,
            "layer": "exact_direct",
            "score": 100
        }

    # =====================================================
    # STAGE 1 — METAPHONE
    # metaphone narrows to phonetically similar pool
    # fuzzy disambiguates within that pool
    # edit distance is last resort if fuzzy fails
    # =====================================================

    meta_candidates = metaphone_candidates(query)

    if meta_candidates:

        # threshold is intentionally low (50) since
        # metaphone already confirmed phonetic similarity
        # fuzzy here just disambiguates, not confirms
        result = fuzzy_search(
            query,
            meta_candidates,
            METAPHONE_FUZZY_THRESHOLD,
            mode
        )

        if result:
            matched_term, score, _ = result
            canonical = SYNONYMS.get(matched_term, matched_term)
            return {
                "canonical": canonical,
                "layer": "metaphone_fuzzy",
                "score": score
            }

        # fuzzy failed — fall back to edit distance
        # similarity gate inside metaphone_best prevents
        # returning a bad match
        best = metaphone_best(query, meta_candidates)

        if best:
            canonical = SYNONYMS.get(best, best)
            # use real edit distance ratio as score
            # instead of fake 100
            edit_dist = levenshtein_distance(query, best)
            word_len = max(len(query), len(best))
            real_score = round((1 - edit_dist / word_len) * 100)
            return {
                "canonical": canonical,
                "layer": "metaphone_edit",
                "score": real_score
            }

    # =====================================================
    # STAGE 2 — PHONETIC NEIGHBOR
    # ONLY REACHED IF METAPHONE FAILED
    # =====================================================

    neighbor_pool = neighbor_candidates(query)

    result = fuzzy_search(
        query,
        neighbor_pool,
        threshold,
        mode
    )

    if result:

        matched_term, score, _ = result

        canonical = SYNONYMS.get(
            matched_term,
            matched_term
        )

        return {
            "canonical": canonical,
            "layer": "neighbor_fuzzy",
            "score": score
        }

    # =====================================================
    # STAGE 3 — SEMANTIC
    # =====================================================

    if len(query) > 4:

        semantic_result = semantic_match(
            query,
            threshold=0.78
        )

        if semantic_result:

            return {
                "canonical": semantic_result["canonical"],
                "layer": "semantic",
                "score": semantic_result["score"]
            }

    return None


# =========================================================
# MAIN EXTRACTION
# =========================================================

def extract_ingredients(text: str) -> list:

    text_lower = normalize_text(text)

    raw_matches = []

    # =====================================================
    # LAYER 1 — EXACT AHO-CORASICK
    # =====================================================

    for end_idx, (
        source,
        original,
        canonical
    ) in AUTOMATON.iter(text_lower):

        start_idx = end_idx - len(original) + 1

        match_text = text_lower[start_idx:end_idx + 1]

        before = (
            text_lower[start_idx - 1]
            if start_idx > 0
            else " "
        )

        after = (
            text_lower[end_idx + 1]
            if end_idx + 1 < len(text_lower)
            else " "
        )

        if before in WORD_BOUNDARIES and after in WORD_BOUNDARIES:

            raw_matches.append({
                "text": match_text,
                "canonical": canonical,
                "span": (start_idx, end_idx),
                "layer": "exact_aho"
            })

    exact_matches = resolve_overlaps(raw_matches)

    matched_positions = set()

    for m in exact_matches:

        start, end = m["span"]

        matched_positions |= set(range(start, end + 1))

    # =====================================================
    # HELPERS
    # =====================================================

    fuzzy_matches = []

    def already_matched(start: int, end: int):

        return bool(
            set(range(start, end + 1))
            & matched_positions
        )

    def append_match(match_dict: dict):

        nonlocal matched_positions

        fuzzy_matches.append(match_dict)

        s, e = match_dict["span"]

        matched_positions |= set(range(s, e + 1))

    # =====================================================
    # POS TOKENS
    # =====================================================

    pos_tokens = get_pos_filtered_tokens(text_lower)

    # =====================================================
    # UNIGRAMS
    # =====================================================

    for token in pos_tokens:

        start_idx = token["start"]
        end_idx = token["end"]

        clean = token["text"]

        if already_matched(start_idx, end_idx):
            continue

        if not valid_candidate(clean):
            continue

        result = resolve_token(
            clean,
            UNIGRAM_THRESHOLD,
            mode="unigram"
        )

        if result:

            append_match({
                "text": clean,
                "canonical": result["canonical"],
                "span": (start_idx, end_idx),
                "layer": result["layer"],
                "score": result["score"]
            })

    # =====================================================
    # BIGRAMS
    # =====================================================

    for i in range(len(pos_tokens) - 1):

        t1 = pos_tokens[i]
        t2 = pos_tokens[i + 1]

        if t2["start"] - t1["end"] > 2:
            continue

        s1 = t1["start"]
        e2 = t2["end"]

        phrase = f"{t1['text']} {t2['text']}"

        if already_matched(s1, e2):
            continue

        if not valid_candidate(phrase):
            continue

        result = resolve_token(
            phrase,
            BIGRAM_THRESHOLD,
            mode="bigram"
        )

        if result:

            append_match({
                "text": phrase,
                "canonical": result["canonical"],
                "span": (s1, e2),
                "layer": result["layer"],
                "score": result["score"]
            })

    # =====================================================
    # TRIGRAMS
    # =====================================================

    for i in range(len(pos_tokens) - 2):

        t1 = pos_tokens[i]
        t2 = pos_tokens[i + 1]
        t3 = pos_tokens[i + 2]

        if t2["start"] - t1["end"] > 2:
            continue

        if t3["start"] - t2["end"] > 2:
            continue

        s1 = t1["start"]
        e3 = t3["end"]

        phrase = (
            f"{t1['text']} "
            f"{t2['text']} "
            f"{t3['text']}"
        )

        if already_matched(s1, e3):
            continue

        if not valid_candidate(phrase):
            continue

        result = resolve_token(
            phrase,
            TRIGRAM_THRESHOLD,
            mode="trigram"
        )

        if result:

            append_match({
                "text": phrase,
                "canonical": result["canonical"],
                "span": (s1, e3),
                "layer": result["layer"],
                "score": result["score"]
            })

    # =====================================================
    # FINAL MERGE
    # =====================================================

    all_matches = exact_matches + fuzzy_matches

    all_matches = resolve_overlaps(all_matches)

    all_matches = deduplicate(all_matches)

    return sorted(
        all_matches,
        key=lambda x: x["span"][0]
    )