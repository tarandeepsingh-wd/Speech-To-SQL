# import re
# from collections import defaultdict

# import spacy
# from metaphone import doublemetaphone
# from Levenshtein import distance as levenshtein_distance
# from rapidfuzz import process, fuzz

# from module1_asr.automaton import AUTOMATON
# from module1_asr.data.db_loader import DB_INGREDIENTS
# from module1_asr.data.ingredient_synonyms import SYNONYMS
# from module1_asr.semantic_matcher import semantic_match


# # =========================================================
# # CONSTANTS
# # =========================================================

# WORD_BOUNDARIES = set(" ,.()\n\t!?;:'\"")

# UNIGRAM_THRESHOLD = 85
# BIGRAM_THRESHOLD = 86
# TRIGRAM_THRESHOLD = 82

# GREY_ZONE_LOW = 78
# GREY_ZONE_RATIO = 0.35

# METAPHONE_FUZZY_THRESHOLD = 70

# VALID_POS = {"NOUN", "PROPN", "ADJ", "X"}


# COMMON_WORDS_BLACKLIST = {

#     "food", "good", "cook", "heat", "bake", "boil", "fry",
#     "dish", "meal", "prep", "step", "stir", "pour", "chop",
#     "dice", "mince", "mix", "fold", "cool", "rest", "toss",
#     "make", "take", "just", "some", "this", "that", "with",
#     "from", "have", "will", "need", "want", "like", "used",
#     "then", "when", "also", "into", "onto", "over", "done",
#     "well", "very", "more", "less", "much", "many", "some",
#     "each", "both", "such", "most", "same", "next", "last",
#     "first", "second", "third",
#     "cups", "tbsp", "tsp", "gram", "grams", "ounce", "pound",
#     "piece", "pieces", "slice", "slices", "bunch", "pinch",
# }


# # =========================================================
# # PHONETIC NEIGHBORS
# # =========================================================

# PHONETIC_NEIGHBORS = {
#     "b": ["b", "m", "p"],
#     "m": ["m", "b", "p", "n"],
#     "p": ["p", "b", "m"],
#     "n": ["n", "m"],
#     "g": ["g", "k", "h", "j"],
#     "j": ["j", "z", "y", "g"],
#     "k": ["k", "g", "h", "c"],
#     "h": ["h", "g", "k"],
#     "s": ["s", "z", "c"],
#     "z": ["z", "s", "j"],
#     "d": ["d", "t"],
#     "t": ["t", "d"],
#     "f": ["f", "v", "p"],
#     "v": ["v", "w", "b"],
#     "w": ["w", "v"],
#     "r": ["r", "l"],
#     "l": ["l", "r"],
# }


# # =========================================================
# # MODELS & INDEXES
# # =========================================================

# nlp = spacy.load("en_core_web_sm")

# ALL_KNOWN_TERMS = list(set(
#     [x.lower().strip() for x in DB_INGREDIENTS] +
#     [x.lower().strip() for x in SYNONYMS.keys()]
# ))

# DB_INGREDIENTS_LOWER = set(x.lower().strip() for x in DB_INGREDIENTS)


# # =========================================================
# # PREFIX INDEX
# # =========================================================

# PREFIX_INDEX = defaultdict(list)

# for term in ALL_KNOWN_TERMS:
#     if term:
#         PREFIX_INDEX[term[0]].append(term)


# # =========================================================
# # METAPHONE INDEX
# # =========================================================

# METAPHONE_INDEX = defaultdict(list)

# for term in ALL_KNOWN_TERMS:

#     if not term:
#         continue

#     primary, secondary = doublemetaphone(term)

#     if primary:
#         METAPHONE_INDEX[primary].append(term)

#     if secondary:
#         METAPHONE_INDEX[secondary].append(term)


# # =========================================================
# # TEXT NORMALIZATION
# # =========================================================

# def normalize_text(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()


# # =========================================================
# # POS FILTERING
# # =========================================================

# def get_pos_filtered_tokens(text: str) -> list:

#     doc = nlp(text)

#     candidates = []

#     for token in doc:

#         if token.pos_ in VALID_POS:

#             candidates.append({
#                 "text": token.text.lower(),
#                 "start": token.idx,
#                 "end": token.idx + len(token.text) - 1
#             })

#     return candidates


# # =========================================================
# # VALIDATION
# # =========================================================

# def valid_candidate(token: str) -> bool:

#     token = token.strip()

#     if len(token) < 3:
#         return False

#     compact = token.replace(" ", "")

#     if not compact.isalpha():
#         return False

#     vowel_count = sum(c in "aeiou" for c in compact)

#     if vowel_count == 0:
#         return False

#     return True


# # =========================================================
# # OVERLAP RESOLUTION
# # =========================================================

# def resolve_overlaps(matches: list) -> list:

#     matches.sort(
#         key=lambda x: (
#             x["span"][1] - x["span"][0],
#             x.get("score", 100)
#         ),
#         reverse=True
#     )

#     accepted = []
#     used = set()

#     for m in matches:

#         start, end = m["span"]

#         positions = set(range(start, end + 1))

#         if not positions & used:
#             accepted.append(m)
#             used |= positions

#     return accepted


# # =========================================================
# # DEDUPLICATION
# # =========================================================

# def deduplicate(matches: list) -> list:

#     seen = set()
#     final = []

#     for m in matches:

#         canonical = m["canonical"]

#         if canonical not in seen:
#             final.append(m)
#             seen.add(canonical)

#     return final


# # =========================================================
# # EDIT DISTANCE CONFIRMATION
# # =========================================================

# def passes_edit_distance(query: str, matched_term: str) -> bool:

#     edit_dist = levenshtein_distance(query, matched_term)

#     word_len = max(len(query), len(matched_term))

#     max_edits = 2 if word_len <= 6 else 3

#     return (
#         edit_dist <= max_edits and
#         (edit_dist / word_len) <= GREY_ZONE_RATIO
#     )


# # =========================================================
# # STAGE 1 — METAPHONE RETRIEVAL
# # =========================================================

# def metaphone_candidates(query: str) -> list:

#     pool = set()

#     primary, secondary = doublemetaphone(query)

#     if primary:
#         pool.update(METAPHONE_INDEX.get(primary, []))

#     if secondary:
#         pool.update(METAPHONE_INDEX.get(secondary, []))

#     return list(pool)


# # =========================================================
# # STAGE 2 — PHONETIC NEIGHBOR RETRIEVAL
# # ONLY USED IF METAPHONE FAILS
# # =========================================================

# def neighbor_candidates(query: str) -> list:

#     pool = set()

#     first = query[0]

#     neighbor_keys = PHONETIC_NEIGHBORS.get(first, [first])

#     for key in neighbor_keys:
#         pool.update(PREFIX_INDEX.get(key, []))

#     return list(pool)


# # =========================================================
# # FUZZY SEARCH
# # =========================================================

# def fuzzy_search(query: str,
#                  candidates: list,
#                  threshold: int,
#                  mode: str = "unigram"):

#     if not candidates:
#         return None

#     scorer = (
#         fuzz.WRatio
#         if mode == "unigram"
#         else fuzz.token_set_ratio
#     )

#     # -----------------------------------------------------
#     # STRICT PASS
#     # -----------------------------------------------------

#     result = process.extractOne(
#         query,
#         candidates,
#         scorer=scorer,
#         score_cutoff=threshold
#     )

#     if result:
#         return result

#     # -----------------------------------------------------
#     # GREY ZONE PASS
#     # -----------------------------------------------------

#     grey_result = process.extractOne(
#         query,
#         candidates,
#         scorer=scorer,
#         score_cutoff=GREY_ZONE_LOW
#     )

#     if grey_result:

#         matched_term, score, _ = grey_result

#         if passes_edit_distance(query, matched_term):
#             return grey_result

#     return None


# # =========================================================
# # METAPHONE BEST CANDIDATE
# # picks DB ingredient over non-DB, uses edit distance
# # to break ties when fuzzy scores are unreliable
# # now includes a similarity gate to prevent bad matches
# # =========================================================

# def metaphone_best(query: str, candidates: list) -> str | None:

#     if not candidates:
#         return None

#     # prefer DB ingredients over synonyms
#     db_candidates = [c for c in candidates if c in DB_INGREDIENTS_LOWER]
#     pool = db_candidates if db_candidates else candidates

#     if len(pool) == 1:
#         best = pool[0]
#     else:
#         # pick by lowest edit distance across full pool
#         best = min(pool, key=lambda c: levenshtein_distance(query, c))

#     # -------------------------------------------------------
#     # SIMILARITY GATE
#     # reject if the best candidate is still too different
#     # from the query — prevents false positives like
#     # "food" → "feta cheese"
#     # -------------------------------------------------------

#     edit_dist = levenshtein_distance(query, best)
#     word_len = max(len(query), len(best))

#     if (edit_dist / word_len) > 0.4:
#         return None

#     return best


# # =========================================================
# # MAIN TOKEN RESOLVER
# # =========================================================

# def resolve_token(query: str,
#                   threshold: int,
#                   mode: str = "unigram"):

#     # =====================================================
#     # BLACKLIST CHECK
#     # reject common English words before any matching
#     # =====================================================

#     if query in COMMON_WORDS_BLACKLIST:
#         return None

#     # =====================================================
#     # STAGE 0 — DIRECT EXACT
#     # =====================================================

#     if query in ALL_KNOWN_TERMS:

#         canonical = SYNONYMS.get(query, query)

#         return {
#             "canonical": canonical,
#             "layer": "exact_direct",
#             "score": 100
#         }

#     # =====================================================
#     # STAGE 1 — METAPHONE
#     # metaphone narrows to phonetically similar pool
#     # fuzzy disambiguates within that pool
#     # edit distance is last resort if fuzzy fails
#     # =====================================================

#     meta_candidates = metaphone_candidates(query)

#     if meta_candidates:

#         # threshold is intentionally low (50) since
#         # metaphone already confirmed phonetic similarity
#         # fuzzy here just disambiguates, not confirms
#         result = fuzzy_search(
#             query,
#             meta_candidates,
#             METAPHONE_FUZZY_THRESHOLD,
#             mode
#         )

#         if result:
#             matched_term, score, _ = result
#             canonical = SYNONYMS.get(matched_term, matched_term)
#             return {
#                 "canonical": canonical,
#                 "layer": "metaphone_fuzzy",
#                 "score": score
#             }

#         # fuzzy failed — fall back to edit distance
#         # similarity gate inside metaphone_best prevents
#         # returning a bad match
#         best = metaphone_best(query, meta_candidates)

#         if best:
#             canonical = SYNONYMS.get(best, best)
#             # use real edit distance ratio as score
#             # instead of fake 100
#             edit_dist = levenshtein_distance(query, best)
#             word_len = max(len(query), len(best))
#             real_score = round((1 - edit_dist / word_len) * 100)
#             return {
#                 "canonical": canonical,
#                 "layer": "metaphone_edit",
#                 "score": real_score
#             }

#     # =====================================================
#     # STAGE 2 — PHONETIC NEIGHBOR
#     # ONLY REACHED IF METAPHONE FAILED
#     # =====================================================

#     neighbor_pool = neighbor_candidates(query)

#     result = fuzzy_search(
#         query,
#         neighbor_pool,
#         threshold,
#         mode
#     )

#     if result:

#         matched_term, score, _ = result

#         canonical = SYNONYMS.get(
#             matched_term,
#             matched_term
#         )

#         return {
#             "canonical": canonical,
#             "layer": "neighbor_fuzzy",
#             "score": score
#         }

#     # =====================================================
#     # STAGE 3 — SEMANTIC
#     # =====================================================

#     if len(query) > 4:

#         semantic_result = semantic_match(
#             query,
#             threshold=0.78
#         )

#         if semantic_result:

#             return {
#                 "canonical": semantic_result["canonical"],
#                 "layer": "semantic",
#                 "score": semantic_result["score"]
#             }

#     return None


# # =========================================================
# # MAIN EXTRACTION
# # =========================================================

# def extract_ingredients(text: str) -> list:

#     text_lower = normalize_text(text)

#     raw_matches = []

#     # =====================================================
#     # LAYER 1 — EXACT AHO-CORASICK
#     # =====================================================

#     for end_idx, (
#         source,
#         original,
#         canonical
#     ) in AUTOMATON.iter(text_lower):

#         start_idx = end_idx - len(original) + 1

#         match_text = text_lower[start_idx:end_idx + 1]

#         before = (
#             text_lower[start_idx - 1]
#             if start_idx > 0
#             else " "
#         )

#         after = (
#             text_lower[end_idx + 1]
#             if end_idx + 1 < len(text_lower)
#             else " "
#         )

#         if before in WORD_BOUNDARIES and after in WORD_BOUNDARIES:

#             raw_matches.append({
#                 "text": match_text,
#                 "canonical": canonical,
#                 "span": (start_idx, end_idx),
#                 "layer": "exact_aho"
#             })

#     exact_matches = resolve_overlaps(raw_matches)

#     matched_positions = set()

#     for m in exact_matches:

#         start, end = m["span"]

#         matched_positions |= set(range(start, end + 1))

#     # =====================================================
#     # HELPERS
#     # =====================================================

#     fuzzy_matches = []

#     def already_matched(start: int, end: int):

#         return bool(
#             set(range(start, end + 1))
#             & matched_positions
#         )

#     def append_match(match_dict: dict):

#         nonlocal matched_positions

#         fuzzy_matches.append(match_dict)

#         s, e = match_dict["span"]

#         matched_positions |= set(range(s, e + 1))

#     # =====================================================
#     # POS TOKENS
#     # =====================================================

#     pos_tokens = get_pos_filtered_tokens(text_lower)

#     # =====================================================
#     # UNIGRAMS
#     # =====================================================

#     for token in pos_tokens:

#         start_idx = token["start"]
#         end_idx = token["end"]

#         clean = token["text"]

#         if already_matched(start_idx, end_idx):
#             continue

#         if not valid_candidate(clean):
#             continue

#         result = resolve_token(
#             clean,
#             UNIGRAM_THRESHOLD,
#             mode="unigram"
#         )

#         if result:

#             append_match({
#                 "text": clean,
#                 "canonical": result["canonical"],
#                 "span": (start_idx, end_idx),
#                 "layer": result["layer"],
#                 "score": result["score"]
#             })

#     # =====================================================
#     # BIGRAMS
#     # =====================================================

#     for i in range(len(pos_tokens) - 1):

#         t1 = pos_tokens[i]
#         t2 = pos_tokens[i + 1]

#         if t2["start"] - t1["end"] > 2:
#             continue

#         s1 = t1["start"]
#         e2 = t2["end"]

#         phrase = f"{t1['text']} {t2['text']}"

#         if already_matched(s1, e2):
#             continue

#         if not valid_candidate(phrase):
#             continue

#         result = resolve_token(
#             phrase,
#             BIGRAM_THRESHOLD,
#             mode="bigram"
#         )

#         if result:

#             append_match({
#                 "text": phrase,
#                 "canonical": result["canonical"],
#                 "span": (s1, e2),
#                 "layer": result["layer"],
#                 "score": result["score"]
#             })

#     # =====================================================
#     # TRIGRAMS
#     # =====================================================

#     for i in range(len(pos_tokens) - 2):

#         t1 = pos_tokens[i]
#         t2 = pos_tokens[i + 1]
#         t3 = pos_tokens[i + 2]

#         if t2["start"] - t1["end"] > 2:
#             continue

#         if t3["start"] - t2["end"] > 2:
#             continue

#         s1 = t1["start"]
#         e3 = t3["end"]

#         phrase = (
#             f"{t1['text']} "
#             f"{t2['text']} "
#             f"{t3['text']}"
#         )

#         if already_matched(s1, e3):
#             continue

#         if not valid_candidate(phrase):
#             continue

#         result = resolve_token(
#             phrase,
#             TRIGRAM_THRESHOLD,
#             mode="trigram"
#         )

#         if result:

#             append_match({
#                 "text": phrase,
#                 "canonical": result["canonical"],
#                 "span": (s1, e3),
#                 "layer": result["layer"],
#                 "score": result["score"]
#             })

#     # =====================================================
#     # FINAL MERGE
#     # =====================================================

#     all_matches = exact_matches + fuzzy_matches

#     all_matches = resolve_overlaps(all_matches)

#     all_matches = deduplicate(all_matches)

#     return sorted(
#         all_matches,
#         key=lambda x: x["span"][0]
#     )

# import re
# from collections import defaultdict

# import spacy
# from metaphone import doublemetaphone
# from Levenshtein import distance as levenshtein_distance
# from rapidfuzz import process, fuzz

# from module1_asr.automaton import INGREDIENT_AUTOMATON, UTENSIL_AUTOMATON
# from module1_asr.data.db_loader import DB_INGREDIENTS, DB_UTENSILS
# from module1_asr.data.ingredient_synonyms import SYNONYMS
# from module1_asr.semantic_matcher import semantic_match


# # ─────────────────────────────────────────────────────────
# # CONSTANTS
# # ─────────────────────────────────────────────────────────

# WORD_BOUNDARIES = set(" ,.()\\n\\t!?;:'\\\"")

# UNIGRAM_THRESHOLD  = 85
# BIGRAM_THRESHOLD   = 86
# TRIGRAM_THRESHOLD  = 82

# GREY_ZONE_LOW   = 78
# GREY_ZONE_RATIO = 0.35

# METAPHONE_FUZZY_THRESHOLD = 70

# VALID_POS = {"NOUN", "PROPN", "ADJ", "X"}

# # Priority order for resolve_overlaps tiebreaking.
# # Lower number = higher priority = wins the overlap.
# LAYER_PRIORITY = {
#     "exact_aho":       0,
#     "exact_direct":    1,
#     "metaphone_fuzzy": 2,
#     "metaphone_edit":  3,
#     "neighbor_fuzzy":  4,
#     "semantic":        5,
# }

# # ─────────────────────────────────────────────────────────
# # COMMON WORDS BLACKLIST
# # Scoped to ingredient extraction only.
# # "cook", "dish", "cool", "cut" etc. are intentionally here
# # because they are NOT ingredients. They are utensils/actions
# # and are handled separately by entity_tagger.
# # ─────────────────────────────────────────────────────────

# COMMON_WORDS_BLACKLIST = {
#     "food", "good", "cook", "heat", "bake", "boil", "fry",
#     "dish", "meal", "prep", "step", "stir", "pour", "chop",
#     "dice", "mince", "mix",  "fold", "cool", "rest", "toss",
#     "make", "take", "just", "some", "this", "that", "with",
#     "from", "have", "will", "need", "want", "like", "used",
#     "then", "when", "also", "into", "onto", "over", "done",
#     "well", "very", "more", "less", "much", "many",
#     "each", "both", "such", "most", "same", "next", "last",
#     "first", "second", "third",
#     "cups", "tbsp", "tsp",  "gram", "grams", "ounce", "pound",
#     "piece", "pieces", "slice", "slices", "bunch", "pinch",
# }


# # ─────────────────────────────────────────────────────────
# # PHONETIC NEIGHBORS
# # ─────────────────────────────────────────────────────────

# PHONETIC_NEIGHBORS = {
#     "b": ["b", "m", "p"],
#     "m": ["m", "b", "p", "n"],
#     "p": ["p", "b", "m"],
#     "n": ["n", "m"],
#     "g": ["g", "k", "h", "j"],
#     "j": ["j", "z", "y", "g"],
#     "k": ["k", "g", "h", "c"],
#     "h": ["h", "g", "k"],
#     "s": ["s", "z", "c"],
#     "z": ["z", "s", "j"],
#     "d": ["d", "t"],
#     "t": ["t", "d"],
#     "f": ["f", "v", "p"],
#     "v": ["v", "w", "b"],
#     "w": ["w", "v"],
#     "r": ["r", "l"],
#     "l": ["l", "r"],
# }


# # ─────────────────────────────────────────────────────────
# # MODELS & INDEXES
# # ─────────────────────────────────────────────────────────

# nlp = spacy.load("en_core_web_sm")

# ALL_KNOWN_TERMS = list(set(
#     [x.lower().strip() for x in DB_INGREDIENTS] +
#     [x.lower().strip() for x in SYNONYMS.keys()]
# ))

# DB_INGREDIENTS_LOWER = set(x.lower().strip() for x in DB_INGREDIENTS)


# # ─────────────────────────────────────────────────────────
# # PREFIX INDEX
# # ─────────────────────────────────────────────────────────

# PREFIX_INDEX: dict = defaultdict(list)

# for _term in ALL_KNOWN_TERMS:
#     if _term:
#         PREFIX_INDEX[_term[0]].append(_term)


# # ─────────────────────────────────────────────────────────
# # METAPHONE INDEX
# # ─────────────────────────────────────────────────────────

# METAPHONE_INDEX: dict = defaultdict(list)

# for _term in ALL_KNOWN_TERMS:
#     if not _term:
#         continue
#     _primary, _secondary = doublemetaphone(_term)
#     if _primary:
#         METAPHONE_INDEX[_primary].append(_term)
#     if _secondary:
#         METAPHONE_INDEX[_secondary].append(_term)


# # ─────────────────────────────────────────────────────────
# # TEXT NORMALIZATION
# # ─────────────────────────────────────────────────────────

# def normalize_text(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()


# # ─────────────────────────────────────────────────────────
# # POS FILTERING
# # ─────────────────────────────────────────────────────────

# def get_pos_filtered_tokens(text: str) -> list:
#     doc = nlp(text)
#     return [
#         {
#             "text":  token.text.lower(),
#             "start": token.idx,
#             "end":   token.idx + len(token.text) - 1,
#         }
#         for token in doc
#         if token.pos_ in VALID_POS
#     ]


# # ─────────────────────────────────────────────────────────
# # VALIDATION
# #
# # CHANGE: was `compact.isalpha()` which rejected hyphens.
# #
# # WHY THIS IS WRONG:
# #   Hyphenated ingredient names like "all-purpose" (flour)
# #   were being rejected before any matching attempt.
# #   After normalize_text, hyphens become spaces, so
# #   "all-purpose" → "all purpose". But the POS filter runs
# #   on the NORMALIZED text, so we still see "all purpose"
# #   as two tokens. The issue is in the BIGRAM/TRIGRAM pass:
# #   if a synonym key contains a hyphen (e.g. "gluten-free"),
# #   the compact check `"glutenfree".isalpha()` was True —
# #   but "gluten-free".replace(" ","") = "gluten-free" which
# #   is NOT alpha. So the check was failing for these.
# #
# # FIX: Strip BOTH spaces AND hyphens before isalpha() check.
# # ─────────────────────────────────────────────────────────

# def valid_candidate(token: str) -> bool:
#     token = token.strip()

#     if len(token) < 3:
#         return False

#     # FIXED: also strip hyphens — "all-purpose" should pass
#     compact = token.replace(" ", "").replace("-", "")

#     if not compact.isalpha():
#         return False

#     vowel_count = sum(c in "aeiou" for c in compact)

#     if vowel_count == 0:
#         return False

#     return True


# # ─────────────────────────────────────────────────────────
# # OVERLAP RESOLUTION
# #
# # CHANGE: sort key was (span_length, score) descending.
# #
# # WHY THIS IS WRONG:
# #   For Aho-Corasick exact matches, "score" is not set, so
# #   x.get("score", 100) returns 100 for all of them. When
# #   two exact matches have the same span length (e.g. "salt"
# #   matched both as a DB ingredient and as a synonym alias
# #   pointing to something different), the tie was broken
# #   arbitrarily — Python's sort is stable but the input
# #   order is non-deterministic across automaton iterations.
# #
# # FIX: Add layer priority as a tiebreaker. exact_aho (0)
# #   always beats metaphone_fuzzy (2) etc. within the same
# #   span length. This makes the result deterministic and
# #   semantically correct (exact matches beat fuzzy ones).
# # ─────────────────────────────────────────────────────────

# def resolve_overlaps(matches: list) -> list:

#     matches.sort(
#         key=lambda x: (
#             x["span"][1] - x["span"][0],        # longer span wins
#             -LAYER_PRIORITY.get(x.get("layer", "semantic"), 99),  # FIXED: layer priority
#             x.get("score", 100),                 # higher score wins
#         ),
#         reverse=True,
#     )

#     accepted = []
#     used = set()

#     for m in matches:
#         start, end = m["span"]
#         positions = set(range(start, end + 1))

#         if not positions & used:
#             accepted.append(m)
#             used |= positions

#     return accepted


# # ─────────────────────────────────────────────────────────
# # DEDUPLICATION
# # ─────────────────────────────────────────────────────────

# def deduplicate(matches: list) -> list:
#     seen  = set()
#     final = []
#     for m in matches:
#         if m["canonical"] not in seen:
#             final.append(m)
#             seen.add(m["canonical"])
#     return final


# # ─────────────────────────────────────────────────────────
# # EDIT DISTANCE CONFIRMATION
# # ─────────────────────────────────────────────────────────

# def passes_edit_distance(query: str, matched_term: str) -> bool:
#     edit_dist = levenshtein_distance(query, matched_term)
#     word_len  = max(len(query), len(matched_term))
#     max_edits = 2 if word_len <= 6 else 3
#     return (
#         edit_dist <= max_edits and
#         (edit_dist / word_len) <= GREY_ZONE_RATIO
#     )


# # ─────────────────────────────────────────────────────────
# # STAGE 1 — METAPHONE RETRIEVAL
# # ─────────────────────────────────────────────────────────

# def metaphone_candidates(query: str) -> list:
#     pool = set()
#     primary, secondary = doublemetaphone(query)
#     if primary:
#         pool.update(METAPHONE_INDEX.get(primary, []))
#     if secondary:
#         pool.update(METAPHONE_INDEX.get(secondary, []))
#     return list(pool)


# # ─────────────────────────────────────────────────────────
# # STAGE 2 — PHONETIC NEIGHBOR RETRIEVAL (only if metaphone fails)
# # ─────────────────────────────────────────────────────────

# def neighbor_candidates(query: str) -> list:
#     pool = set()
#     first = query[0]
#     for key in PHONETIC_NEIGHBORS.get(first, [first]):
#         pool.update(PREFIX_INDEX.get(key, []))
#     return list(pool)


# # ─────────────────────────────────────────────────────────
# # FUZZY SEARCH
# # ─────────────────────────────────────────────────────────

# def fuzzy_search(
#     query:      str,
#     candidates: list,
#     threshold:  int,
#     mode:       str = "unigram",
# ):
#     if not candidates:
#         return None

#     scorer = fuzz.WRatio if mode == "unigram" else fuzz.token_set_ratio

#     result = process.extractOne(
#         query, candidates, scorer=scorer, score_cutoff=threshold
#     )
#     if result:
#         return result

#     # Grey-zone pass
#     grey = process.extractOne(
#         query, candidates, scorer=scorer, score_cutoff=GREY_ZONE_LOW
#     )
#     if grey:
#         matched_term, score, _ = grey
#         if passes_edit_distance(query, matched_term):
#             return grey

#     return None


# # ─────────────────────────────────────────────────────────
# # METAPHONE BEST CANDIDATE
# # ─────────────────────────────────────────────────────────

# def metaphone_best(query: str, candidates: list) -> str | None:
#     if not candidates:
#         return None

#     db_candidates = [c for c in candidates if c in DB_INGREDIENTS_LOWER]
#     pool = db_candidates if db_candidates else candidates

#     best = min(pool, key=lambda c: levenshtein_distance(query, c))

#     edit_dist = levenshtein_distance(query, best)
#     word_len  = max(len(query), len(best))

#     if (edit_dist / word_len) > 0.4:
#         return None

#     return best


# # ─────────────────────────────────────────────────────────
# # MAIN TOKEN RESOLVER
# # ─────────────────────────────────────────────────────────

# def resolve_token(query: str, threshold: int, mode: str = "unigram"):

#     if query in COMMON_WORDS_BLACKLIST:
#         return None

#     # Stage 0 — Direct exact match
#     if query in ALL_KNOWN_TERMS:
#         canonical = SYNONYMS.get(query, query)
#         return {"canonical": canonical, "layer": "exact_direct", "score": 100}

#     # Stage 1 — Metaphone phonetic
#     meta_candidates = metaphone_candidates(query)

#     if meta_candidates:
#         result = fuzzy_search(query, meta_candidates, METAPHONE_FUZZY_THRESHOLD, mode)
#         if result:
#             matched_term, score, _ = result
#             return {
#                 "canonical": SYNONYMS.get(matched_term, matched_term),
#                 "layer":     "metaphone_fuzzy",
#                 "score":     score,
#             }

#         best = metaphone_best(query, meta_candidates)
#         if best:
#             edit_dist  = levenshtein_distance(query, best)
#             word_len   = max(len(query), len(best))
#             real_score = round((1 - edit_dist / word_len) * 100)
#             return {
#                 "canonical": SYNONYMS.get(best, best),
#                 "layer":     "metaphone_edit",
#                 "score":     real_score,
#             }

#     # Stage 2 — Phonetic neighbor prefix
#     neighbor_pool = neighbor_candidates(query)
#     result = fuzzy_search(query, neighbor_pool, threshold, mode)
#     if result:
#         matched_term, score, _ = result
#         return {
#             "canonical": SYNONYMS.get(matched_term, matched_term),
#             "layer":     "neighbor_fuzzy",
#             "score":     score,
#         }

#     # Stage 3 — Semantic embedding
#     if len(query) > 4:
#         semantic_result = semantic_match(query, threshold=0.78)
#         if semantic_result:
#             return {
#                 "canonical": semantic_result["canonical"],
#                 "layer":     "semantic",
#                 "score":     semantic_result["score"],
#             }

#     return None


# # ─────────────────────────────────────────────────────────
# # EXTRACT INGREDIENTS (unchanged logic, updated automaton name)
# # ─────────────────────────────────────────────────────────

# def extract_ingredients(text: str) -> list:

#     text_lower = normalize_text(text)
#     raw_matches = []

#     # Layer 1 — Exact Aho-Corasick
#     for end_idx, (source, original, canonical) in INGREDIENT_AUTOMATON.iter(text_lower):
#         start_idx  = end_idx - len(original) + 1
#         match_text = text_lower[start_idx: end_idx + 1]
#         before = text_lower[start_idx - 1] if start_idx > 0 else " "
#         after  = text_lower[end_idx + 1]   if end_idx + 1 < len(text_lower) else " "

#         if before in WORD_BOUNDARIES and after in WORD_BOUNDARIES:
#             raw_matches.append({
#                 "text":      match_text,
#                 "canonical": canonical,
#                 "span":      (start_idx, end_idx),
#                 "layer":     "exact_aho",
#             })

#     exact_matches     = resolve_overlaps(raw_matches)
#     matched_positions = set()
#     for m in exact_matches:
#         matched_positions |= set(range(m["span"][0], m["span"][1] + 1))

#     fuzzy_matches = []

#     def already_matched(s, e):
#         return bool(set(range(s, e + 1)) & matched_positions)

#     def append_match(d):
#         nonlocal matched_positions
#         fuzzy_matches.append(d)
#         matched_positions |= set(range(d["span"][0], d["span"][1] + 1))

#     pos_tokens = get_pos_filtered_tokens(text_lower)

#     # Unigrams
#     for token in pos_tokens:
#         s, e, clean = token["start"], token["end"], token["text"]
#         if already_matched(s, e) or not valid_candidate(clean):
#             continue
#         result = resolve_token(clean, UNIGRAM_THRESHOLD, "unigram")
#         if result:
#             append_match({"text": clean, "canonical": result["canonical"],
#                           "span": (s, e), "layer": result["layer"], "score": result["score"]})

#     # Bigrams
#     for i in range(len(pos_tokens) - 1):
#         t1, t2 = pos_tokens[i], pos_tokens[i + 1]
#         if t2["start"] - t1["end"] > 2:
#             continue
#         s, e = t1["start"], t2["end"]
#         phrase = f"{t1['text']} {t2['text']}"
#         if already_matched(s, e) or not valid_candidate(phrase):
#             continue
#         result = resolve_token(phrase, BIGRAM_THRESHOLD, "bigram")
#         if result:
#             append_match({"text": phrase, "canonical": result["canonical"],
#                           "span": (s, e), "layer": result["layer"], "score": result["score"]})

#     # Trigrams
#     for i in range(len(pos_tokens) - 2):
#         t1, t2, t3 = pos_tokens[i], pos_tokens[i + 1], pos_tokens[i + 2]
#         if t2["start"] - t1["end"] > 2 or t3["start"] - t2["end"] > 2:
#             continue
#         s, e = t1["start"], t3["end"]
#         phrase = f"{t1['text']} {t2['text']} {t3['text']}"
#         if already_matched(s, e) or not valid_candidate(phrase):
#             continue
#         result = resolve_token(phrase, TRIGRAM_THRESHOLD, "trigram")
#         if result:
#             append_match({"text": phrase, "canonical": result["canonical"],
#                           "span": (s, e), "layer": result["layer"], "score": result["score"]})

#     all_matches = resolve_overlaps(exact_matches + fuzzy_matches)
#     all_matches = deduplicate(all_matches)
#     return sorted(all_matches, key=lambda x: x["span"][0])


# # ─────────────────────────────────────────────────────────
# # EXTRACT UTENSILS
# #
# # NEW: Separate function for utensil extraction.
# #
# # WHY SEPARATE FROM extract_ingredients:
# #   1. Utensils don't need the 4-stage fuzzy pipeline —
# #      utensil names are short, unambiguous English words
# #      (bowl, pan, skillet). Exact Aho-Corasick is enough.
# #   2. "cup" conflict: "cup" is a utensil AND a unit of
# #      measurement. We detect it as a utensil ONLY when
# #      it is NOT immediately preceded by a number or
# #      quantity word (one, two, three, half, etc.).
# #      e.g. "use a cup" → utensil; "one cup of flour" → unit
# # ─────────────────────────────────────────────────────────

# _QUANTITY_WORDS = {
#     "one", "two", "three", "four", "five", "six", "seven",
#     "eight", "nine", "ten", "half", "quarter", "a", "an",
# }
# _QUANTITY_PATTERN = re.compile(r"\b(\d+[\./]?\d*)\s+cup", re.IGNORECASE)


# def extract_utensils(text: str) -> list:

#     text_lower = normalize_text(text)
#     raw_matches = []

#     for end_idx, (source, original, canonical) in UTENSIL_AUTOMATON.iter(text_lower):
#         start_idx  = end_idx - len(original) + 1
#         match_text = text_lower[start_idx: end_idx + 1]
#         before = text_lower[start_idx - 1] if start_idx > 0 else " "
#         after  = text_lower[end_idx + 1]   if end_idx + 1 < len(text_lower) else " "

#         if before not in WORD_BOUNDARIES or after not in WORD_BOUNDARIES:
#             continue

#         # "cup" conflict: skip if preceded by a number/quantity word
#         if canonical == "cup":
#             preceding = text_lower[:start_idx].strip().split()
#             if preceding:
#                 last_word = preceding[-1].rstrip(".,")
#                 if last_word in _QUANTITY_WORDS or last_word.replace(".", "").isdigit():
#                     continue
#             if _QUANTITY_PATTERN.search(text_lower[max(0, start_idx - 10): end_idx + 5]):
#                 continue

#         raw_matches.append({
#             "text":      match_text,
#             "canonical": canonical,
#             "span":      (start_idx, end_idx),
#             "layer":     "exact_aho",
#         })

#     return resolve_overlaps(raw_matches)
# i
# import re
# from collections import defaultdict

# import spacy
# from metaphone import doublemetaphone
# from Levenshtein import distance as levenshtein_distance
# from rapidfuzz import process, fuzz

# from module1_asr.automaton import INGREDIENT_AUTOMATON, UTENSIL_AUTOMATON
# from module1_asr.data.db_loader import DB_INGREDIENTS, DB_UTENSILS
# from module1_asr.data.ingredient_synonyms import SYNONYMS
# from module1_asr.semantic_matcher import semantic_match


# # ─────────────────────────────────────────────────────────
# # CONSTANTS
# # ─────────────────────────────────────────────────────────

# WORD_BOUNDARIES = set(" ,.()\n\t!?;:'\"")

# UNIGRAM_THRESHOLD  = 85
# BIGRAM_THRESHOLD   = 86
# TRIGRAM_THRESHOLD  = 82

# GREY_ZONE_LOW   = 78
# GREY_ZONE_RATIO = 0.35

# METAPHONE_FUZZY_THRESHOLD = 70

# VALID_POS = {"NOUN", "PROPN", "ADJ", "X"}

# # Priority order for resolve_overlaps tiebreaking.
# # Lower number = higher priority = wins the overlap.
# LAYER_PRIORITY = {
#     "exact_aho":              0,
#     "exact_direct":           1,
#     "metaphone_fuzzy":        2,
#     "metaphone_edit":         3,
#     "neighbor_fuzzy":         4,
#     "semantic":               5,
#     # rescue pass variants — same type as base but ranked lower
#     # because POS confirmation is absent
#     "metaphone_fuzzy_rescue": 6,
#     "metaphone_edit_rescue":  7,
#     "neighbor_fuzzy_rescue":  8,
#     "semantic_rescue":        9,
#     "exact_direct_rescue":    10,
# }


# # ─────────────────────────────────────────────────────────
# # COMMON WORDS BLACKLIST
# # Scoped to ingredient extraction only.
# # "cook", "dish", "cool", "cut" etc. are intentionally here
# # because they are NOT ingredients. They are utensils/actions
# # and are handled separately by entity_tagger.
# # ─────────────────────────────────────────────────────────

# COMMON_WORDS_BLACKLIST = {
#     "food", "good", "cook", "heat", "bake", "boil", "fry",
#     "dish", "meal", "prep", "step", "stir", "pour", "chop",
#     "dice", "mince", "mix",  "fold", "cool", "rest", "toss",
#     "make", "take", "just", "some", "this", "that", "with",
#     "from", "have", "will", "need", "want", "like", "used",
#     "then", "when", "also", "into", "onto", "over", "done",
#     "well", "very", "more", "less", "much", "many",
#     "each", "both", "such", "most", "same", "next", "last",
#     "first", "second", "third",
#     "cups", "tbsp", "tsp",  "gram", "grams", "ounce", "pound",
#     "piece", "pieces", "slice", "slices", "bunch", "pinch",
# }


# # ─────────────────────────────────────────────────────────
# # PHONETIC NEIGHBORS
# # Maps a first-character to phonetically similar characters.
# # Used by neighbor_candidates() when metaphone produces no pool.
# # ─────────────────────────────────────────────────────────

# PHONETIC_NEIGHBORS = {
#     "b": ["b", "m", "p"],
#     "m": ["m", "b", "p", "n"],
#     "p": ["p", "b", "m"],
#     "n": ["n", "m"],
#     "g": ["g", "k", "h", "j"],
#     "j": ["j", "z", "y", "g"],
#     "k": ["k", "g", "h", "c"],
#     "h": ["h", "g", "k"],
#     "s": ["s", "z", "c"],
#     "z": ["z", "s", "j"],
#     "d": ["d", "t"],
#     "t": ["t", "d"],
#     "f": ["f", "v", "p"],
#     "v": ["v", "w", "b"],
#     "w": ["w", "v"],
#     "r": ["r", "l"],
#     "l": ["l", "r"],
# }


# # ─────────────────────────────────────────────────────────
# # MODELS & INDEXES
# # ─────────────────────────────────────────────────────────

# nlp = spacy.load("en_core_web_sm")

# ALL_KNOWN_TERMS = list(set(
#     [x.lower().strip() for x in DB_INGREDIENTS] +
#     [x.lower().strip() for x in SYNONYMS.keys()]
# ))

# DB_INGREDIENTS_LOWER = set(x.lower().strip() for x in DB_INGREDIENTS)


# # ─────────────────────────────────────────────────────────
# # PREFIX INDEX
# # ─────────────────────────────────────────────────────────

# PREFIX_INDEX: dict = defaultdict(list)

# for _term in ALL_KNOWN_TERMS:
#     if _term:
#         PREFIX_INDEX[_term[0]].append(_term)


# # ─────────────────────────────────────────────────────────
# # METAPHONE INDEX
# # ─────────────────────────────────────────────────────────

# METAPHONE_INDEX: dict = defaultdict(list)

# for _term in ALL_KNOWN_TERMS:
#     if not _term:
#         continue
#     _primary, _secondary = doublemetaphone(_term)
#     if _primary:
#         METAPHONE_INDEX[_primary].append(_term)
#     if _secondary:
#         METAPHONE_INDEX[_secondary].append(_term)


# # ─────────────────────────────────────────────────────────
# # TEXT NORMALIZATION
# # ─────────────────────────────────────────────────────────

# def normalize_text(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()


# # ─────────────────────────────────────────────────────────
# # POS FILTERING
# # ─────────────────────────────────────────────────────────

# def get_pos_filtered_tokens(text: str) -> list:
#     doc = nlp(text)
#     return [
#         {
#             "text":  token.text.lower(),
#             "start": token.idx,
#             "end":   token.idx + len(token.text) - 1,
#         }
#         for token in doc
#         if token.pos_ in VALID_POS
#     ]


# # ─────────────────────────────────────────────────────────
# # VALIDATION
# #
# # CHANGE: was `compact.isalpha()` which rejected hyphens.
# #
# # WHY THIS IS WRONG:
# #   Hyphenated ingredient names like "all-purpose" (flour)
# #   were being rejected before any matching attempt.
# #   After normalize_text, hyphens become spaces, so
# #   "all-purpose" → "all purpose". But the POS filter runs
# #   on the NORMALIZED text, so we still see "all purpose"
# #   as two tokens. The issue is in the BIGRAM/TRIGRAM pass:
# #   if a synonym key contains a hyphen (e.g. "gluten-free"),
# #   the compact check `"glutenfree".isalpha()` was True —
# #   but "gluten-free".replace(" ","") = "gluten-free" which
# #   is NOT alpha. So the check was failing for these.
# #
# # FIX: Strip BOTH spaces AND hyphens before isalpha() check.
# # ─────────────────────────────────────────────────────────

# def valid_candidate(token: str) -> bool:
#     token = token.strip()

#     if len(token) < 3:
#         return False

#     # FIXED: also strip hyphens — "all-purpose" should pass
#     compact = token.replace(" ", "").replace("-", "")

#     if not compact.isalpha():
#         return False

#     vowel_count = sum(c in "aeiou" for c in compact)

#     if vowel_count == 0:
#         return False

#     return True


# # ─────────────────────────────────────────────────────────
# # OVERLAP RESOLUTION
# #
# # CHANGE: sort key was (span_length, score) descending.
# #
# # WHY THIS IS WRONG:
# #   For Aho-Corasick exact matches, "score" is not set, so
# #   x.get("score", 100) returns 100 for all of them. When
# #   two exact matches have the same span length (e.g. "salt"
# #   matched both as a DB ingredient and as a synonym alias
# #   pointing to something different), the tie was broken
# #   arbitrarily — Python's sort is stable but the input
# #   order is non-deterministic across automaton iterations.
# #
# # FIX: Add layer priority as a tiebreaker. exact_aho (0)
# #   always beats metaphone_fuzzy (2) etc. within the same
# #   span length. This makes the result deterministic and
# #   semantically correct (exact matches beat fuzzy ones).
# # ─────────────────────────────────────────────────────────

# def resolve_overlaps(matches: list) -> list:
#     matches.sort(
#         key=lambda x: (
#             x["span"][1] - x["span"][0],                              # longer span wins
#             -LAYER_PRIORITY.get(x.get("layer", "semantic"), 99),      # FIXED: layer priority
#             x.get("score", 100),                                       # higher score wins
#         ),
#         reverse=True,
#     )

#     accepted = []
#     used = set()

#     for m in matches:
#         start, end = m["span"]
#         positions = set(range(start, end + 1))

#         if not positions & used:
#             accepted.append(m)
#             used |= positions

#     return accepted


# # ─────────────────────────────────────────────────────────
# # DEDUPLICATION
# # ─────────────────────────────────────────────────────────

# def deduplicate(matches: list) -> list:
#     seen  = set()
#     final = []
#     for m in matches:
#         if m["canonical"] not in seen:
#             final.append(m)
#             seen.add(m["canonical"])
#     return final


# # ─────────────────────────────────────────────────────────
# # EDIT DISTANCE CONFIRMATION
# # ─────────────────────────────────────────────────────────

# def passes_edit_distance(query: str, matched_term: str) -> bool:
#     edit_dist = levenshtein_distance(query, matched_term)
#     word_len  = max(len(query), len(matched_term))
#     max_edits = 2 if word_len <= 6 else 3
#     return (
#         edit_dist <= max_edits and
#         (edit_dist / word_len) <= GREY_ZONE_RATIO
#     )


# # ─────────────────────────────────────────────────────────
# # STAGE 1 — METAPHONE RETRIEVAL
# # ─────────────────────────────────────────────────────────

# def metaphone_candidates(query: str) -> list:
#     pool = set()
#     primary, secondary = doublemetaphone(query)
#     if primary:
#         pool.update(METAPHONE_INDEX.get(primary, []))
#     if secondary:
#         pool.update(METAPHONE_INDEX.get(secondary, []))
#     return list(pool)


# # ─────────────────────────────────────────────────────────
# # STAGE 2 — PHONETIC NEIGHBOR RETRIEVAL (only if metaphone fails)
# # ─────────────────────────────────────────────────────────

# def neighbor_candidates(query: str) -> list:
#     pool = set()
#     first = query[0]
#     for key in PHONETIC_NEIGHBORS.get(first, [first]):
#         pool.update(PREFIX_INDEX.get(key, []))
#     return list(pool)


# # ─────────────────────────────────────────────────────────
# # FUZZY SEARCH
# # ─────────────────────────────────────────────────────────

# def fuzzy_search(
#     query:      str,
#     candidates: list,
#     threshold:  int,
#     mode:       str = "unigram",
# ):
#     if not candidates:
#         return None

#     scorer = fuzz.WRatio if mode == "unigram" else fuzz.token_set_ratio

#     result = process.extractOne(
#         query, candidates, scorer=scorer, score_cutoff=threshold
#     )
#     if result:
#         return result

#     # Grey-zone pass
#     grey = process.extractOne(
#         query, candidates, scorer=scorer, score_cutoff=GREY_ZONE_LOW
#     )
#     if grey:
#         matched_term, score, _ = grey
#         if passes_edit_distance(query, matched_term):
#             return grey

#     return None


# # ─────────────────────────────────────────────────────────
# # METAPHONE BEST CANDIDATE
# # picks DB ingredient over non-DB, uses edit distance
# # to break ties when fuzzy scores are unreliable;
# # includes a similarity gate to prevent bad matches
# # ─────────────────────────────────────────────────────────

# def metaphone_best(query: str, candidates: list) -> str | None:
#     if not candidates:
#         return None

#     db_candidates = [c for c in candidates if c in DB_INGREDIENTS_LOWER]
#     pool = db_candidates if db_candidates else candidates

#     best = min(pool, key=lambda c: levenshtein_distance(query, c))

#     edit_dist = levenshtein_distance(query, best)
#     word_len  = max(len(query), len(best))

#     if (edit_dist / word_len) > 0.4:
#         return None

#     return best


# # ─────────────────────────────────────────────────────────
# # VOWEL-STRIPPED METAPHONE RE-RANK  (NEW)
# #
# # WHY THIS IS NEEDED:
# #   Standard fuzzy and raw edit distance are misled by
# #   vowel noise in Whisper output. Consider "leechi":
# #     metaphone bucket → [leek, lychee]
# #     WRatio: leek (77) > lychee (67)  → leek wins (wrong)
# #     raw edit: leek (3) < lychee (4)  → leek wins (wrong)
# #
# #   Stripping vowels isolates the consonant skeleton, which
# #   is far more stable across Whisper transcription errors:
# #     leechi → lch    leek → lk    lychee → lych
# #     ed(lch, lk)=2   ed(lch, lych)=1  → lychee wins ✓
# #
# # CASES FIXED (no hardcoding):
# #   leechi  → lychee  (lch vs lych/lk)
# #   jinger  → ginger  (jngr vs gngr, phonetic neighbor)
# #   sirial  → cereal  (srl vs srl, same code → ed=0)
# #   biriyani→ biryani (bryn vs bryn, same code → ed=0)
# # ─────────────────────────────────────────────────────────

# def _strip_vowels(s: str) -> str:
#     """Remove all vowels from s — leaves the consonant skeleton."""
#     return re.sub(r"[aeiou]", "", s)


# def _metaphone_rerank(query: str):
#     """
#     Re-rank metaphone bucket candidates using vowel-stripped
#     Levenshtein distance. Vowels are the noisiest part of
#     Whisper output — stripping them gives a cleaner consonant
#     skeleton comparison that outperforms raw edit distance for
#     phonetically similar but vowel-mangled words.

#     Returns:
#         (matched_term, fuzzy_score, 0) tuple (same shape as
#         rapidfuzz extractOne) if a candidate passes the gate,
#         otherwise None.

#     Gate: consonant-skeleton edit distance ≤ 2. This is tight
#     enough to reject genuinely unrelated words while permitting
#     vowel-heavy spelling variants.
#     """
#     p, s = doublemetaphone(query)
#     candidates: list = []
#     if p:
#         candidates.extend(METAPHONE_INDEX.get(p, []))
#     if s:
#         candidates.extend(METAPHONE_INDEX.get(s, []))
#     candidates = list(set(candidates))

#     if not candidates:
#         return None

#     query_stripped = _strip_vowels(query)

#     ranked = sorted(
#         candidates,
#         key=lambda c: levenshtein_distance(query_stripped, _strip_vowels(c)),
#     )

#     best            = ranked[0]
#     best_consonant_ed = levenshtein_distance(query_stripped, _strip_vowels(best))

#     # Only accept if consonant skeleton is genuinely close
#     if best_consonant_ed > 2:
#         return None

#     best_score = fuzz.WRatio(query, best)
#     return (best, best_score, 0)


# # ─────────────────────────────────────────────────────────
# # MAIN TOKEN RESOLVER
# # ─────────────────────────────────────────────────────────

# def resolve_token(query: str, threshold: int, mode: str = "unigram"):

#     if query in COMMON_WORDS_BLACKLIST:
#         return None

#     # ── Stage 0 — Direct exact match ──────────────────────
#     if query in ALL_KNOWN_TERMS:
#         canonical = SYNONYMS.get(query, query)
#         return {"canonical": canonical, "layer": "exact_direct", "score": 100}

#     # ── Stage 1 — Metaphone phonetic ──────────────────────
#     meta_candidates = metaphone_candidates(query)

#     if meta_candidates:
#         result = fuzzy_search(query, meta_candidates, METAPHONE_FUZZY_THRESHOLD, mode)
#         if result:
#             matched_term, score, _ = result
#             return {
#                 "canonical": SYNONYMS.get(matched_term, matched_term),
#                 "layer":     "metaphone_fuzzy",
#                 "score":     score,
#             }

#         # ── Stage 1.5 (NEW) — Vowel-stripped consonant skeleton re-rank ──
#         # Called when standard fuzzy fails on the metaphone bucket.
#         # Fixes cases like "leechi" → "lychee" where vowel patterns
#         # mislead character-level fuzzy and raw edit distance.
#         # Tried BEFORE metaphone_best so the vowel-aware scorer gets
#         # first pick on the same candidate pool.
#         meta_rerank = _metaphone_rerank(query)
#         if meta_rerank:
#             matched_term, score, _ = meta_rerank
#             return {
#                 "canonical": SYNONYMS.get(matched_term, matched_term),
#                 "layer":     "metaphone_fuzzy",
#                 "score":     score,
#             }

#         # ── Stage 1 fallback — Raw edit distance on metaphone pool ──
#         best = metaphone_best(query, meta_candidates)
#         if best:
#             edit_dist  = levenshtein_distance(query, best)
#             word_len   = max(len(query), len(best))
#             real_score = round((1 - edit_dist / word_len) * 100)
#             return {
#                 "canonical": SYNONYMS.get(best, best),
#                 "layer":     "metaphone_edit",
#                 "score":     real_score,
#             }

#     # ── Stage 2 — Phonetic neighbor prefix ────────────────
#     # Only reached if metaphone produced no candidates at all
#     # (e.g. very short or unusual tokens).
#     neighbor_pool = neighbor_candidates(query)
#     result = fuzzy_search(query, neighbor_pool, threshold, mode)
#     if result:
#         matched_term, score, _ = result
#         return {
#             "canonical": SYNONYMS.get(matched_term, matched_term),
#             "layer":     "neighbor_fuzzy",
#             "score":     score,
#         }

#     # ── Stage 3 — Semantic embedding ──────────────────────
#     if len(query) > 4:
#         semantic_result = semantic_match(query, threshold=0.78)
#         if semantic_result:
#             return {
#                 "canonical": semantic_result["canonical"],
#                 "layer":     "semantic",
#                 "score":     semantic_result["score"],
#             }

#     return None


# # ─────────────────────────────────────────────────────────
# # EXTRACT INGREDIENTS
# # ─────────────────────────────────────────────────────────

# def extract_ingredients(text: str) -> list:

#     text_lower  = normalize_text(text)
#     raw_matches = []

#     # ── Layer 1 — Exact Aho-Corasick ──────────────────────
#     for end_idx, (source, original, canonical) in INGREDIENT_AUTOMATON.iter(text_lower):
#         start_idx  = end_idx - len(original) + 1
#         match_text = text_lower[start_idx: end_idx + 1]
#         before = text_lower[start_idx - 1] if start_idx > 0 else " "
#         after  = text_lower[end_idx + 1]   if end_idx + 1 < len(text_lower) else " "

#         if before in WORD_BOUNDARIES and after in WORD_BOUNDARIES:
#             raw_matches.append({
#                 "text":      match_text,
#                 "canonical": canonical,
#                 "span":      (start_idx, end_idx),
#                 "layer":     "exact_aho",
#             })

#     exact_matches     = resolve_overlaps(raw_matches)
#     matched_positions = set()
#     for m in exact_matches:
#         matched_positions |= set(range(m["span"][0], m["span"][1] + 1))

#     fuzzy_matches = []

#     def already_matched(s, e):
#         return bool(set(range(s, e + 1)) & matched_positions)

#     def append_match(d):
#         nonlocal matched_positions
#         fuzzy_matches.append(d)
#         matched_positions |= set(range(d["span"][0], d["span"][1] + 1))

#     pos_tokens = get_pos_filtered_tokens(text_lower)

#     # ─────────────────────────────────────────────────────
#     # UNIGRAMS — TWO-PASS APPROACH
#     #
#     # WHY: spaCy's English model mis-tags Hindi/food words.
#     # "lehsoon", "adrak", "jeera" get tagged as VERB, NUM,
#     # or other non-NOUN tags and are silently dropped by the
#     # POS filter — meaning fuzzy matching NEVER runs on them.
#     #
#     # FIX: Pass 1 uses POS-filtered tokens (high confidence).
#     #      Pass 2 is a rescue pass — ALL whitespace tokens
#     #      that are still unmatched after Pass 1 get a fuzzy
#     #      attempt too, regardless of their spaCy POS tag.
#     #      This is what makes the system work without hardcoding
#     #      every spelling variant of every Hindi ingredient.
#     # ─────────────────────────────────────────────────────

#     # Pass 1 — POS-filtered unigrams (high confidence)
#     for token in pos_tokens:
#         s, e, clean = token["start"], token["end"], token["text"]
#         if already_matched(s, e) or not valid_candidate(clean):
#             continue
#         result = resolve_token(clean, UNIGRAM_THRESHOLD, "unigram")
#         if result:
#             append_match({
#                 "text":      clean,
#                 "canonical": result["canonical"],
#                 "span":      (s, e),
#                 "layer":     result["layer"],
#                 "score":     result["score"],
#             })

#     # Pass 2 — Rescue pass: all unmatched whitespace tokens
#     # Covers words spaCy tagged as VERB/DET/etc. (Hindi words,
#     # domain-specific terms, proper nouns in food context).
#     # Uses a slightly higher threshold to compensate for the
#     # lack of POS-based confidence.
#     for word_match in re.finditer(r"\w+", text_lower):
#         clean = word_match.group()
#         s     = word_match.start()
#         e     = word_match.end() - 1
#         if already_matched(s, e) or not valid_candidate(clean):
#             continue
#         # Skip stop-words and very common English words
#         # that POS filter would have caught anyway
#         if clean in COMMON_WORDS_BLACKLIST:
#             continue
#         result = resolve_token(clean, UNIGRAM_THRESHOLD + 5, "unigram")
#         if result:
#             append_match({
#                 "text":      clean,
#                 "canonical": result["canonical"],
#                 "span":      (s, e),
#                 "layer":     result["layer"] + "_rescue",
#                 "score":     result["score"],
#             })

#     # ── Bigrams ───────────────────────────────────────────
#     for i in range(len(pos_tokens) - 1):
#         t1, t2 = pos_tokens[i], pos_tokens[i + 1]
#         if t2["start"] - t1["end"] > 2:
#             continue
#         s, e   = t1["start"], t2["end"]
#         phrase = f"{t1['text']} {t2['text']}"
#         if already_matched(s, e) or not valid_candidate(phrase):
#             continue
#         result = resolve_token(phrase, BIGRAM_THRESHOLD, "bigram")
#         if result:
#             append_match({
#                 "text":      phrase,
#                 "canonical": result["canonical"],
#                 "span":      (s, e),
#                 "layer":     result["layer"],
#                 "score":     result["score"],
#             })

#     # ── Trigrams ──────────────────────────────────────────
#     for i in range(len(pos_tokens) - 2):
#         t1, t2, t3 = pos_tokens[i], pos_tokens[i + 1], pos_tokens[i + 2]
#         if t2["start"] - t1["end"] > 2 or t3["start"] - t2["end"] > 2:
#             continue
#         s, e   = t1["start"], t3["end"]
#         phrase = f"{t1['text']} {t2['text']} {t3['text']}"
#         if already_matched(s, e) or not valid_candidate(phrase):
#             continue
#         result = resolve_token(phrase, TRIGRAM_THRESHOLD, "trigram")
#         if result:
#             append_match({
#                 "text":      phrase,
#                 "canonical": result["canonical"],
#                 "span":      (s, e),
#                 "layer":     result["layer"],
#                 "score":     result["score"],
#             })

#     all_matches = resolve_overlaps(exact_matches + fuzzy_matches)
#     all_matches = deduplicate(all_matches)
#     return sorted(all_matches, key=lambda x: x["span"][0])


# # ─────────────────────────────────────────────────────────
# # EXTRACT UTENSILS
# #
# # NEW: Separate function for utensil extraction.
# #
# # WHY SEPARATE FROM extract_ingredients:
# #   1. Utensils don't need the 4-stage fuzzy pipeline —
# #      utensil names are short, unambiguous English words
# #      (bowl, pan, skillet). Exact Aho-Corasick is enough.
# #   2. "cup" conflict: "cup" is a utensil AND a unit of
# #      measurement. We detect it as a utensil ONLY when
# #      it is NOT immediately preceded by a number or
# #      quantity word (one, two, three, half, etc.).
# #      e.g. "use a cup" → utensil; "one cup of flour" → unit
# # ─────────────────────────────────────────────────────────

# _QUANTITY_WORDS = {
#     "one", "two", "three", "four", "five", "six", "seven",
#     "eight", "nine", "ten", "half", "quarter", "a", "an",
# }
# _QUANTITY_PATTERN = re.compile(r"\b(\d+[\./]?\d*)\s+cup", re.IGNORECASE)


# def extract_utensils(text: str) -> list:

#     text_lower  = normalize_text(text)
#     raw_matches = []

#     for end_idx, (source, original, canonical) in UTENSIL_AUTOMATON.iter(text_lower):
#         start_idx  = end_idx - len(original) + 1
#         match_text = text_lower[start_idx: end_idx + 1]
#         before = text_lower[start_idx - 1] if start_idx > 0 else " "
#         after  = text_lower[end_idx + 1]   if end_idx + 1 < len(text_lower) else " "

#         if before not in WORD_BOUNDARIES or after not in WORD_BOUNDARIES:
#             continue

#         # "cup" conflict: skip if preceded by a number/quantity word
#         if canonical == "cup":
#             preceding = text_lower[:start_idx].strip().split()
#             if preceding:
#                 last_word = preceding[-1].rstrip(".,")
#                 if last_word in _QUANTITY_WORDS or last_word.replace(".", "").isdigit():
#                     continue
#             if _QUANTITY_PATTERN.search(text_lower[max(0, start_idx - 10): end_idx + 5]):
#                 continue

#         raw_matches.append({
#             "text":      match_text,
#             "canonical": canonical,
#             "span":      (start_idx, end_idx),
#             "layer":     "exact_aho",
#         })

#     return resolve_overlaps(raw_matches)
import re
from collections import defaultdict

import spacy
from metaphone import doublemetaphone
from Levenshtein import distance as levenshtein_distance
from rapidfuzz import process, fuzz

from module1_asr.automaton import INGREDIENT_AUTOMATON, UTENSIL_AUTOMATON
from module1_asr.data.db_loader import DB_INGREDIENTS, DB_UTENSILS
from module1_asr.data.ingredient_synonyms import SYNONYMS
from module1_asr.semantic_matcher import semantic_match


# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────

WORD_BOUNDARIES = set(" ,.()\n\t!?;:'\"")

UNIGRAM_THRESHOLD  = 85
BIGRAM_THRESHOLD   = 86
TRIGRAM_THRESHOLD  = 82

GREY_ZONE_LOW   = 78
GREY_ZONE_RATIO = 0.35

METAPHONE_FUZZY_THRESHOLD = 70

VALID_POS = {"NOUN", "PROPN", "ADJ", "X"}

# Lower number = higher priority = wins when spans overlap.
LAYER_PRIORITY = {
    "exact_aho":              0,
    "exact_direct":           1,
    "metaphone_fuzzy":        2,
    "metaphone_edit":         3,
    "neighbor_fuzzy":         4,
    "semantic":               5,
    "metaphone_fuzzy_rescue": 6,
    "metaphone_edit_rescue":  7,
    "neighbor_fuzzy_rescue":  8,
    "semantic_rescue":        9,
    "exact_direct_rescue":    10,
}

COMMON_WORDS_BLACKLIST = {
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


# ─────────────────────────────────────────────────────────
# MODELS & INDEXES
# ─────────────────────────────────────────────────────────

nlp = spacy.load("en_core_web_sm")

ALL_KNOWN_TERMS = list(set(
    [x.lower().strip() for x in DB_INGREDIENTS] +
    [x.lower().strip() for x in SYNONYMS.keys()]
))

DB_INGREDIENTS_LOWER = set(x.lower().strip() for x in DB_INGREDIENTS)

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


# ─────────────────────────────────────────────────────────
# TEXT NORMALIZATION
# ─────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────
# POS FILTERING
# ─────────────────────────────────────────────────────────

def get_pos_filtered_tokens(text: str) -> list:
    doc = nlp(text)
    return [
        {
            "text":  token.text.lower(),
            "start": token.idx,
            "end":   token.idx + len(token.text) - 1,
        }
        for token in doc
        if token.pos_ in VALID_POS
    ]


# ─────────────────────────────────────────────────────────
# VALIDATION
#
# FIX: Strip both spaces AND hyphens before isalpha() so
# that hyphenated multi-word terms ("all-purpose") pass.
# ─────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────
# OVERLAP RESOLUTION
#
# FIX: Added LAYER_PRIORITY as tiebreaker so exact_aho
# always beats fuzzy layers when spans are equal length.
# ─────────────────────────────────────────────────────────

def resolve_overlaps(matches: list) -> list:
    matches.sort(
        key=lambda x: (
            x["span"][1] - x["span"][0],
            -LAYER_PRIORITY.get(x.get("layer", "semantic"), 99),
            x.get("score", 100),
        ),
        reverse=True,
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


# ─────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────

def deduplicate(matches: list) -> list:
    seen  = set()
    final = []
    for m in matches:
        if m["canonical"] not in seen:
            final.append(m)
            seen.add(m["canonical"])
    return final


# ─────────────────────────────────────────────────────────
# EDIT DISTANCE CONFIRMATION
# ─────────────────────────────────────────────────────────

def passes_edit_distance(query: str, matched_term: str) -> bool:
    edit_dist = levenshtein_distance(query, matched_term)
    word_len  = max(len(query), len(matched_term))
    max_edits = 2 if word_len <= 6 else 3
    return (
        edit_dist <= max_edits and
        (edit_dist / word_len) <= GREY_ZONE_RATIO
    )


# ─────────────────────────────────────────────────────────
# STAGE 1 — METAPHONE RETRIEVAL
# ─────────────────────────────────────────────────────────

def metaphone_candidates(query: str) -> list:
    pool = set()
    primary, secondary = doublemetaphone(query)
    if primary:
        pool.update(METAPHONE_INDEX.get(primary, []))
    if secondary:
        pool.update(METAPHONE_INDEX.get(secondary, []))
    return list(pool)


# ─────────────────────────────────────────────────────────
# STAGE 2 — PHONETIC NEIGHBOR RETRIEVAL
# Only reached when metaphone produces no candidates.
# ─────────────────────────────────────────────────────────

def neighbor_candidates(query: str) -> list:
    pool = set()
    first = query[0]
    for key in PHONETIC_NEIGHBORS.get(first, [first]):
        pool.update(PREFIX_INDEX.get(key, []))
    return list(pool)


# ─────────────────────────────────────────────────────────
# FUZZY SEARCH (strict pass + grey-zone pass)
# ─────────────────────────────────────────────────────────

def fuzzy_search(
    query:      str,
    candidates: list,
    threshold:  int,
    mode:       str = "unigram",
):
    if not candidates:
        return None

    scorer = fuzz.WRatio if mode == "unigram" else fuzz.token_set_ratio

    result = process.extractOne(
        query, candidates, scorer=scorer, score_cutoff=threshold
    )
    if result:
        return result

    # Grey-zone: lower threshold confirmed by edit distance gate
    grey = process.extractOne(
        query, candidates, scorer=scorer, score_cutoff=GREY_ZONE_LOW
    )
    if grey:
        matched_term, score, _ = grey
        if passes_edit_distance(query, matched_term):
            return grey

    return None


# ─────────────────────────────────────────────────────────
# METAPHONE BEST — raw edit distance last resort
# ─────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────
# VOWEL-STRIPPED METAPHONE RE-RANK  (PRIMARY SCORER)
#
# WHY THIS RUNS BEFORE FUZZY — not after:
#
#   Within a metaphone bucket every candidate is already
#   phonetically equivalent. WRatio then picks the wrong
#   winner by coincidental CHARACTER overlap:
#
#     "sirial" vs "sorrel" — share leading 's', WRatio ≈ 75
#     "sirial" vs "cereal" — no shared start,   WRatio ≈ 67
#     → fuzzy picks sorrel  ✗
#
#     "jingar" vs "jhinga"(shrimp) — share 'jing', WRatio high
#     "jingar" vs "ginger"         — share 'ing',  WRatio lower
#     → fuzzy picks jhinga/shrimp  ✗
#
#   Consonant skeleton has no such bias. All vowels are noise
#   from Whisper's acoustic model; stripping them leaves the
#   stable phonemic core:
#
#     sirial→srl | cereal→srl (ed=0)  sorrel→srrl (ed=1) → cereal ✓
#     meda  →md  | maida →md  (ed=0)  meat  →mt   (ed=1) → maida  ✓
#     leechi→lch | lychee→lch (ed=0)  leek  →lk   (ed=2) → lychee ✓
#     jingar→jngr| ginger→gngr(ed=1)  jhinga→jhng (ed=2) → ginger ✓
#
#   Ties broken by vowel count proximity (sirial has 3 vowels;
#   cereal has 3, sorrel has 2 → cereal wins the tie).
#
# GATE: consonant ed ≤ 2 prevents unrelated words from
#   sneaking in when the bucket happens to be large.
# ─────────────────────────────────────────────────────────

def _strip_vowels(s: str) -> str:
    return re.sub(r"[aeiou]", "", s)


def _count_vowels(s: str) -> int:
    return sum(c in "aeiou" for c in s)


def _metaphone_rerank(query: str):
    """
    Re-rank the metaphone bucket using:
      1. Vowel-stripped Levenshtein distance  (primary)
      2. Vowel count proximity to query       (tiebreaker)

    Returns (matched_term, fuzzy_score, 0) or None.
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

    query_stripped    = _strip_vowels(query)
    query_vowel_count = _count_vowels(query)

    ranked = sorted(
        candidates,
        key=lambda c: (
            levenshtein_distance(query_stripped, _strip_vowels(c)),  # primary
            abs(_count_vowels(c) - query_vowel_count),               # tiebreaker
        ),
    )

    best              = ranked[0]
    best_consonant_ed = levenshtein_distance(query_stripped, _strip_vowels(best))

    if best_consonant_ed > 2:
        return None

    return (best, fuzz.WRatio(query, best), 0)


# ─────────────────────────────────────────────────────────
# MAIN TOKEN RESOLVER
#
# Stage order:
#   0  — exact direct match
#   1a — metaphone rerank  (consonant skeleton, PRIMARY)
#   1b — metaphone fuzzy   (WRatio, fallback when gate fails)
#   1c — metaphone edit    (raw Levenshtein, last resort)
#   2  — phonetic neighbor fuzzy
#   3  — semantic embedding
# ─────────────────────────────────────────────────────────

def resolve_token(query: str, threshold: int, mode: str = "unigram"):

    if query in COMMON_WORDS_BLACKLIST:
        return None

    # ── Stage 0 — Exact direct match ──────────────────────
    if query in ALL_KNOWN_TERMS:
        return {
            "canonical": SYNONYMS.get(query, query),
            "layer":     "exact_direct",
            "score":     100,
        }

    # ── Stage 1 — Metaphone bucket ────────────────────────
    meta_candidates = metaphone_candidates(query)

    if meta_candidates:

        # 1a — Consonant skeleton re-rank (PRIMARY)
        meta_rerank = _metaphone_rerank(query)
        if meta_rerank:
            matched_term, score, _ = meta_rerank
            return {
                "canonical": SYNONYMS.get(matched_term, matched_term),
                "layer":     "metaphone_fuzzy",
                "score":     score,
            }

        # 1b — Character fuzzy fallback
        # Reached only when no candidate had consonant ed ≤ 2.
        result = fuzzy_search(query, meta_candidates, METAPHONE_FUZZY_THRESHOLD, mode)
        if result:
            matched_term, score, _ = result
            return {
                "canonical": SYNONYMS.get(matched_term, matched_term),
                "layer":     "metaphone_fuzzy",
                "score":     score,
            }

        # 1c — Raw edit distance (last resort inside bucket)
        best = metaphone_best(query, meta_candidates)
        if best:
            edit_dist  = levenshtein_distance(query, best)
            word_len   = max(len(query), len(best))
            real_score = round((1 - edit_dist / word_len) * 100)
            return {
                "canonical": SYNONYMS.get(best, best),
                "layer":     "metaphone_edit",
                "score":     real_score,
            }

    # ── Stage 2 — Phonetic neighbor prefix ────────────────
    # Only reached when metaphone yielded zero candidates.
    neighbor_pool = neighbor_candidates(query)
    result = fuzzy_search(query, neighbor_pool, threshold, mode)
    if result:
        matched_term, score, _ = result
        return {
            "canonical": SYNONYMS.get(matched_term, matched_term),
            "layer":     "neighbor_fuzzy",
            "score":     score,
        }

    # ── Stage 3 — Semantic embedding ──────────────────────
    if len(query) > 4:
        semantic_result = semantic_match(query, threshold=0.78)
        if semantic_result:
            return {
                "canonical": semantic_result["canonical"],
                "layer":     "semantic",
                "score":     semantic_result["score"],
            }

    return None


# ─────────────────────────────────────────────────────────
# EXTRACT INGREDIENTS
# ─────────────────────────────────────────────────────────

def extract_ingredients(text: str) -> list:

    text_lower  = normalize_text(text)
    raw_matches = []

    # ── Layer 1 — Exact Aho-Corasick ──────────────────────
    for end_idx, (source, original, canonical) in INGREDIENT_AUTOMATON.iter(text_lower):
        start_idx  = end_idx - len(original) + 1
        match_text = text_lower[start_idx: end_idx + 1]
        before = text_lower[start_idx - 1] if start_idx > 0 else " "
        after  = text_lower[end_idx + 1]   if end_idx + 1 < len(text_lower) else " "
        if before in WORD_BOUNDARIES and after in WORD_BOUNDARIES:
            raw_matches.append({
                "text":      match_text,
                "canonical": canonical,
                "span":      (start_idx, end_idx),
                "layer":     "exact_aho",
            })

    exact_matches     = resolve_overlaps(raw_matches)
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

    # ── Unigrams — two-pass ────────────────────────────────
    #
    # Pass 1: POS-filtered tokens (high confidence).
    # Pass 2: rescue pass — ALL whitespace tokens still
    #   unmatched, at threshold+5. Catches Hindi/domain words
    #   that spaCy mis-tags (lehsoon, adrak, meda, jeera…).

    # Pass 1 — POS-filtered
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

    # Pass 2 — Rescue (all unmatched whitespace tokens)
    for word_match in re.finditer(r"\w+", text_lower):
        clean = word_match.group()
        s     = word_match.start()
        e     = word_match.end() - 1
        if already_matched(s, e) or not valid_candidate(clean):
            continue
        if clean in COMMON_WORDS_BLACKLIST:
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

    # ── Bigrams ───────────────────────────────────────────
    for i in range(len(pos_tokens) - 1):
        t1, t2 = pos_tokens[i], pos_tokens[i + 1]
        if t2["start"] - t1["end"] > 2:
            continue
        s, e   = t1["start"], t2["end"]
        phrase = f"{t1['text']} {t2['text']}"
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

    # Trigrams
    for i in range(len(pos_tokens) - 2):
        t1, t2, t3 = pos_tokens[i], pos_tokens[i + 1], pos_tokens[i + 2]
        if t2["start"] - t1["end"] > 2 or t3["start"] - t2["end"] > 2:
            continue
        s, e   = t1["start"], t3["end"]
        phrase = f"{t1['text']} {t2['text']} {t3['text']}"
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

    all_matches = resolve_overlaps(exact_matches + fuzzy_matches)
    all_matches = deduplicate(all_matches)
    return sorted(all_matches, key=lambda x: x["span"][0])


# ─────────────────────────────────────────────────────────
# EXTRACT UTENSILS
#
# Kept separate from extract_ingredients because:
#   1. Utensils need exact matching only — no fuzzy pipeline.
#   2. "cup" is both a utensil and a unit of measurement;
#      context rule here keeps it out of ingredients.
# ─────────────────────────────────────────────────────────

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

        # "cup" conflict: skip when preceded by a quantity
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