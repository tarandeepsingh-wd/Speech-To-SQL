from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from metaphone import doublemetaphone
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from speech_to_sql.common.config import load_recipe_db_config, load_sql_schema_config


DATA_DIR = Path(__file__).parent / "data"
SQL_OPERATOR_LEXICON_PATH = DATA_DIR / "sql_operator_lexicon.json"
MAX_PHRASE_TOKENS = 4
MAX_ROLE_CUE_TOKENS = 3

NUMBER_ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
NUMBER_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
NUMBER_SCALES = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
}
NUMBER_CONNECTORS = {"and"}
NUMBER_WORDS = {
    **{term: ("value", value) for term, value in NUMBER_ONES.items()},
    **{term: ("value", value) for term, value in NUMBER_TENS.items()},
    **{term: ("scale", value) for term, value in NUMBER_SCALES.items()},
}


@dataclass(frozen=True)
class LexiconTerm:
    term: str
    kind: str
    source: str
    operator_type: str = ""


@dataclass(frozen=True)
class TokenSpan:
    text: str
    start: int
    end: int


def normalize_schema_terms(text: str) -> dict[str, Any]:
    """
    Correct likely ASR/typing noise for SQL-critical terms.

    The correction vocabulary is built from project config/schema files. The
    matching is phonetic + contextual, so noisy words can map to schema terms
    without adding handwritten typo pairs.
    """
    if not text.strip():
        return {"text": text, "changed": False, "corrections": []}

    number_result = _normalize_spoken_numbers(text)
    phrase_result = _normalize_phrase_terms(number_result["text"])
    token_result = _normalize_single_tokens(phrase_result["text"])
    role_result = _normalize_role_cues(token_result["text"])
    corrections = [
        *number_result["corrections"],
        *phrase_result["corrections"],
        *token_result["corrections"],
        *role_result["corrections"],
    ]

    return {
        "text": role_result["text"],
        "changed": role_result["text"] != text,
        "corrections": corrections,
    }


def _normalize_spoken_numbers(text: str) -> dict[str, Any]:
    tokens = _token_spans(text)
    updates = []
    used_positions: set[int] = set()
    protected_spans = _compare_phrase_spans(text)

    index = 0
    while index < len(tokens):
        best = None
        for end_index in range(min(len(tokens), index + 7), index, -1):
            window = tokens[index:end_index]
            start = window[0].start
            end = window[-1].end
            if _overlaps_any((start, end), protected_spans):
                continue
            positions = set(range(start, end))
            if positions & used_positions:
                continue
            parsed = _parse_number_phrase(window)
            if not parsed:
                continue
            value, confidence, has_scale_or_tens = parsed
            if not _number_context_supports(text, (start, end), has_scale_or_tens):
                continue
            source = text[start:end]
            best = (start, end, source, str(value), confidence)
            break

        if not best:
            index += 1
            continue

        start, end, source, target, confidence = best
        used_positions.update(range(start, end))
        updates.append((start, end, source, target, confidence))
        while index < len(tokens) and tokens[index].end <= end:
            index += 1

    if not updates:
        return {"text": text, "changed": False, "corrections": []}

    updated = []
    last = 0
    corrections = []
    for start, end, source, target, confidence in sorted(updates, key=lambda item: item[0]):
        if start < last:
            continue
        updated.append(text[last:start])
        updated.append(target)
        last = end
        corrections.append({
            "source": source.lower(),
            "target": target,
            "layer": "spoken_number_grammar",
            "kind": "number",
            "confidence": round(confidence / 100, 3),
            "evidence": "english_number_grammar",
        })
    updated.append(text[last:])

    normalized = "".join(updated)
    return {
        "text": normalized,
        "changed": normalized != text,
        "corrections": corrections,
    }


def _parse_number_phrase(tokens: list[TokenSpan]) -> tuple[int, float, bool] | None:
    if not tokens:
        return None

    parsed_tokens = []
    fuzzy_scores = []
    for token in tokens:
        normalized = _normalize_term(token.text)
        if normalized in NUMBER_CONNECTORS:
            parsed_tokens.append(("connector", 0, normalized))
            fuzzy_scores.append(100.0)
            continue
        parsed = _canonical_number_token(normalized)
        if not parsed:
            return None
        kind, value, canonical, score = parsed
        parsed_tokens.append((kind, value, canonical))
        fuzzy_scores.append(score)

    while parsed_tokens and parsed_tokens[0][0] == "connector":
        parsed_tokens.pop(0)
        fuzzy_scores.pop(0)
    while parsed_tokens and parsed_tokens[-1][0] == "connector":
        parsed_tokens.pop()
        fuzzy_scores.pop()
    if not parsed_tokens:
        return None
    if len(parsed_tokens) == 1 and parsed_tokens[0][0] != "value":
        return None

    total = 0
    current = 0
    seen_number = False
    has_scale_or_tens = False
    previous_kind = ""
    for index, (kind, value, canonical) in enumerate(parsed_tokens):
        if kind == "connector":
            if previous_kind not in {"value", "scale"}:
                return None
            if index == len(parsed_tokens) - 1:
                return None
            continue

        seen_number = True
        if kind == "value":
            if value >= 20:
                has_scale_or_tens = True
            if previous_kind == "value" and current and value >= 20:
                return None
            current += int(value)
        elif kind == "scale":
            has_scale_or_tens = True
            scale = int(value)
            if scale == 100:
                current = max(1, current) * scale
            else:
                total += max(1, current) * scale
                current = 0
        previous_kind = kind

    if not seen_number:
        return None
    value = total + current
    if value < 0:
        return None
    if len(parsed_tokens) == 1 and min(fuzzy_scores or [100.0]) < 100 and value < 20:
        return None
    return value, min(fuzzy_scores) if fuzzy_scores else 100.0, has_scale_or_tens


def _canonical_number_token(token: str) -> tuple[str, int, str, float] | None:
    if token in NUMBER_WORDS:
        kind, value = NUMBER_WORDS[token]
        return kind, int(value), token, 100.0
    if len(token) < 3:
        return None

    shortlist: set[str] = set()
    for term, _score, _index in process.extract(
        token,
        list(NUMBER_WORDS),
        scorer=fuzz.WRatio,
        limit=12,
        score_cutoff=58,
    ):
        shortlist.add(str(term))

    token_codes = _phonetic_codes(token)
    if token_codes:
        for term in NUMBER_WORDS:
            if token_codes & _phonetic_codes(term):
                shortlist.add(term)

    best_term = ""
    best_score = 0.0
    best_edit_distance = 10_000
    for term in shortlist:
        if not _compatible_start(token, term):
            continue
        lexical = float(fuzz.WRatio(token, term))
        phonetic = _phonetic_score(token, term)
        if lexical < 58:
            continue
        if lexical < 78 and phonetic < 92:
            continue
        score = lexical if lexical < 78 else max(lexical, phonetic)
        edit_distance = Levenshtein.distance(_compact_term(token), _compact_term(term))
        if score > best_score or (abs(score - best_score) < 0.001 and edit_distance < best_edit_distance):
            best_term = term
            best_score = score
            best_edit_distance = edit_distance

    if not best_term:
        return None
    kind, value = NUMBER_WORDS[best_term]
    return kind, int(value), best_term, best_score


def _number_context_supports(
    text: str,
    span: tuple[int, int],
    has_scale_or_tens: bool,
) -> bool:
    window = _context_window(text, span, radius=52).lower()
    if re.search(r"\d", window):
        return True
    if _near_number_unit(window):
        return True
    if _near_numeric_domain_token(window):
        return True
    if _near_compare_expression(window):
        return True
    return has_scale_or_tens and _near_limit_context(window)


def _near_number_unit(window: str) -> bool:
    return bool(re.search(r"\b(?:kcal|calories|calorie|cals|grams?|g|mg|minutes?|mins?|hours?|hrs?)\b", window))


def _near_limit_context(window: str) -> bool:
    return bool(re.search(r"\b(?:top|limit|first|last|lowest|highest|under|below|over|above)\b", window))


def _near_numeric_domain_token(window: str) -> bool:
    for token in re.findall(r"[a-z][a-z-]{2,}", window):
        if token in NUMBER_CONNECTORS:
            continue
        if token in NUMBER_WORDS:
            continue
        if token in _numeric_domain_terms():
            return True
        candidate = _best_token_candidate(token, _single_token_terms())
        if not candidate:
            continue
        term, score = candidate
        lexicon_term = _single_token_terms()[term]
        if lexicon_term.kind in {"nutrition", "time"} and score >= 78 and _compatible_start(token, term):
            return True
    return False


def _near_compare_expression(window: str) -> bool:
    normalized = _normalize_term(window)
    tokens = normalized.split()
    compare_terms = _compare_phrase_terms()
    for width in range(3, 0, -1):
        for index in range(0, len(tokens) - width + 1):
            phrase = " ".join(tokens[index:index + width])
            if phrase in compare_terms:
                return True
            candidate = _best_phrase_candidate(phrase, compare_terms)
            if not candidate:
                continue
            term, score = candidate
            if score >= 84 and _compatible_start(phrase.split()[0], term.split()[0]):
                return True
    return False


def _normalize_phrase_terms(text: str) -> dict[str, Any]:
    choices = _phrase_terms()
    if not choices:
        return {"text": text, "changed": False, "corrections": []}

    tokens = _token_spans(text)
    updates = []
    used_positions: set[int] = set()
    for width in range(MAX_PHRASE_TOKENS, 1, -1):
        for index in range(0, len(tokens) - width + 1):
            window = tokens[index:index + width]
            if any(re.search(r"\d", token.text) for token in window):
                continue
            start = window[0].start
            end = window[-1].end
            positions = set(range(start, end))
            if positions & used_positions:
                continue

            surface = text[start:end]
            normalized_surface = _normalize_term(surface)
            if normalized_surface in choices:
                continue

            candidate = _best_phrase_candidate(normalized_surface, choices)
            if not candidate:
                continue
            term, score = candidate
            lexicon_term = choices[term]
            if not _should_apply_phrase(text, (start, end), normalized_surface, lexicon_term, score):
                continue

            used_positions |= positions
            updates.append((start, end, surface, lexicon_term, score))

    if not updates:
        return {"text": text, "changed": False, "corrections": []}
    return _apply_updates(text, updates, layer="schema_phrase_lexicon")


def _normalize_single_tokens(text: str) -> dict[str, Any]:
    choices = _single_token_terms()
    if not choices:
        return {"text": text, "changed": False, "corrections": []}

    updates = []
    protected_spans = [*_exact_phrase_spans(text), *_compare_phrase_spans(text)]
    for match in re.finditer(r"\b[a-zA-Z][a-zA-Z-]{2,}\b", text):
        if _overlaps_any(match.span(), protected_spans):
            continue
        token = match.group(0)
        normalized_token = _normalize_term(token)
        if normalized_token in choices:
            continue

        candidate = _best_token_candidate(normalized_token, choices)
        if not candidate:
            continue
        term, score = candidate
        lexicon_term = choices[term]
        if not _should_apply_token(text, match.span(), normalized_token, lexicon_term, score):
            continue
        updates.append((match.start(), match.end(), token, lexicon_term, score))

    if not updates:
        return {"text": text, "changed": False, "corrections": []}
    return _apply_updates(text, updates, layer="schema_lexicon")


def _normalize_role_cues(text: str) -> dict[str, Any]:
    choices = _role_cue_terms()
    if not choices:
        return {"text": text, "changed": False, "corrections": []}

    tokens = _token_spans(text)
    updates = []
    used_positions: set[int] = set()
    max_width = min(MAX_ROLE_CUE_TOKENS, _max_term_tokens(choices))
    for width in range(max_width, 0, -1):
        for index in range(0, len(tokens) - width + 1):
            window = tokens[index:index + width]
            start = window[0].start
            end = window[-1].end
            positions = set(range(start, end))
            if positions & used_positions:
                continue

            surface = text[start:end]
            normalized_surface = _normalize_term(surface)
            if not normalized_surface or normalized_surface in choices:
                continue

            candidate = _best_role_cue_candidate(normalized_surface, choices)
            if not candidate:
                continue
            term, score = candidate
            lexicon_term = choices[term]
            if not _should_apply_role_cue(text, (start, end), normalized_surface, lexicon_term, score):
                continue

            used_positions |= positions
            updates.append((start, end, surface, lexicon_term, score))

    if not updates:
        return {"text": text, "changed": False, "corrections": []}
    return _apply_updates(text, updates, layer="role_cue_lexicon")


def _best_role_cue_candidate(
    surface: str,
    choices: dict[str, LexiconTerm],
) -> tuple[str, float] | None:
    compact = _compact_term(surface)
    if len(compact) < 3:
        return None

    shortlist: set[str] = set()
    for term, _score, _index in process.extract(
        surface,
        list(choices.keys()),
        scorer=fuzz.WRatio,
        limit=16,
        score_cutoff=48,
    ):
        shortlist.add(str(term))

    surface_codes = _phonetic_codes(surface)
    if surface_codes:
        for term in choices:
            if surface_codes & _phonetic_codes(term):
                shortlist.add(term)

    best_term = ""
    best_score = 0.0
    best_edit_distance = 10_000
    best_length = 0
    surface_token_count = len(surface.split())
    for term in shortlist:
        if surface_token_count != len(term.split()):
            continue
        term_compact = _compact_term(term)
        if abs(len(compact) - len(term_compact)) > max(2, len(term_compact) // 2):
            continue
        score = _phrase_similarity(surface, term) if " " in term else _combined_similarity(surface, term)
        edit_distance = Levenshtein.distance(compact, term_compact)
        if (
            score > best_score
            or (
                abs(score - best_score) < 0.001
                and (edit_distance < best_edit_distance or len(term_compact) > best_length)
            )
        ):
            best_term = term
            best_score = score
            best_edit_distance = edit_distance
            best_length = len(term_compact)

    if not best_term or best_score < 60:
        return None
    return best_term, best_score


def _should_apply_role_cue(
    text: str,
    span: tuple[int, int],
    surface: str,
    candidate: LexiconTerm,
    score: float,
) -> bool:
    if candidate.kind not in {"nutrition_presence_cue", "nutrition_negation_cue"}:
        return False
    if not _presence_cue_context_supports(text, span):
        return False
    return _role_cue_similarity_supported(surface, candidate.term, score)


def _role_cue_similarity_supported(surface: str, target: str, score: float) -> bool:
    surface_compact = _compact_term(surface)
    target_compact = _compact_term(target)
    edit_distance = Levenshtein.distance(surface_compact, target_compact)
    compatible_start = _compatible_start(surface, target)
    phonetic = _phonetic_score(surface, target)

    if score >= 92 and compatible_start:
        return True
    if phonetic >= 92 and score >= 76 and compatible_start:
        return True
    if edit_distance <= 1 and score >= 70 and compatible_start:
        return True
    return False


def _presence_cue_context_supports(text: str, span: tuple[int, int]) -> bool:
    suffix = text[span[1]:span[1] + 72]
    prefix = text[max(0, span[0] - 56):span[0]]
    return (
        _presence_target_in_window(suffix, leading=True)
        or _presence_target_in_window(prefix, leading=False)
    )


def _presence_target_in_window(window: str, *, leading: bool) -> bool:
    normalized = _normalize_word_sequence(window)
    tokens = normalized.split()
    if not tokens:
        return False
    scoped_tokens = tokens[:6] if leading else tokens[-6:]
    scope = " ".join(scoped_tokens)
    if not scope:
        return False

    target_terms = _presence_target_terms()
    for term in target_terms:
        if _normalized_phrase_in_text(term, scope):
            return True

    single_targets = _presence_single_target_terms()
    for token in scoped_tokens:
        if token in single_targets:
            return True
        if len(token) < 4:
            continue
        candidate = _best_token_candidate(token, _presence_target_choices())
        if not candidate:
            continue
        target, score = candidate
        if score >= 90 and _compatible_start(token, target):
            return True
    return False


def _normalized_phrase_in_text(phrase: str, text: str) -> bool:
    return bool(re.search(rf"(?:^|\s){re.escape(phrase)}(?:$|\s)", text))


def _exact_phrase_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for phrase in sorted(_phrase_terms(), key=len, reverse=True):
        pattern = _phrase_pattern(phrase)
        spans.extend(match.span() for match in pattern.finditer(text))
    return spans


def _compare_phrase_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for phrase in sorted(_compare_phrase_terms(), key=len, reverse=True):
        if len(phrase.split()) < 2:
            continue
        pattern = _phrase_pattern(phrase)
        spans.extend(match.span() for match in pattern.finditer(text))

    choices = _compare_phrase_terms()
    tokens = _token_spans(text)
    for width in range(MAX_PHRASE_TOKENS, 1, -1):
        for index in range(0, len(tokens) - width + 1):
            window = tokens[index:index + width]
            if any(re.search(r"\d", token.text) for token in window):
                continue
            span = (window[0].start, window[-1].end)
            if _overlaps_any(span, spans):
                continue
            surface = text[span[0]:span[1]]
            normalized_surface = _normalize_term(surface)
            if normalized_surface in choices:
                spans.append(span)
                continue
            candidate = _best_phrase_candidate(normalized_surface, choices)
            if not candidate:
                continue
            term, score = candidate
            lexicon_term = choices[term]
            if lexicon_term.kind != "sql_operator" or lexicon_term.operator_type != "COMPARE":
                continue
            if _should_apply_phrase(text, span, normalized_surface, lexicon_term, score):
                spans.append(span)
    return spans


def _phrase_pattern(phrase: str) -> re.Pattern:
    parts = [re.escape(token) for token in phrase.split()]
    return re.compile(r"\b" + r"[\s-]+".join(parts) + r"\b", re.IGNORECASE)


def _overlaps_any(span: tuple[int, int], protected_spans: list[tuple[int, int]]) -> bool:
    return any(span[0] < protected[1] and protected[0] < span[1] for protected in protected_spans)


def _apply_updates(text: str, updates: list[tuple[int, int, str, LexiconTerm, float]], *, layer: str) -> dict[str, Any]:
    updated = []
    last = 0
    corrections = []
    for start, end, surface, lexicon_term, score in sorted(updates, key=lambda item: item[0]):
        if start < last:
            continue
        updated.append(text[last:start])
        updated.append(_preserve_case(surface, lexicon_term.term))
        last = end
        corrections.append({
            "source": surface.lower(),
            "target": lexicon_term.term,
            "layer": layer,
            "kind": lexicon_term.kind,
            "confidence": round(score / 100, 3),
            "evidence": lexicon_term.source,
        })
    updated.append(text[last:])

    normalized = "".join(updated)
    return {
        "text": normalized,
        "changed": normalized != text,
        "corrections": corrections,
    }


def _best_token_candidate(token: str, choices: dict[str, LexiconTerm]) -> tuple[str, float] | None:
    if len(token) < 4:
        return None

    shortlist: set[str] = set()
    for term, _score, _index in process.extract(
        token,
        list(choices.keys()),
        scorer=fuzz.WRatio,
        limit=20,
        score_cutoff=58,
    ):
        shortlist.add(str(term))

    token_codes = _phonetic_codes(token)
    if token_codes:
        for term in choices:
            if token_codes & _phonetic_codes(term):
                shortlist.add(term)

    best_term = ""
    best_score = 0.0
    for term in shortlist:
        target_len = len(term)
        if target_len < 4:
            continue
        if abs(len(token) - target_len) > max(2, target_len // 2):
            continue
        score = _combined_similarity(token, term)
        if score > best_score:
            best_term = term
            best_score = score

    if not best_term or best_score < 76:
        return None
    return best_term, best_score


def _best_phrase_candidate(phrase: str, choices: dict[str, LexiconTerm]) -> tuple[str, float] | None:
    if len(phrase.replace(" ", "")) < 5:
        return None

    best_term = ""
    best_score = 0.0
    phrase_tokens = phrase.split()
    for term in choices:
        term_tokens = term.split()
        if len(phrase_tokens) != len(term_tokens):
            continue
        score = _phrase_similarity(phrase, term)
        if score > best_score:
            best_term = term
            best_score = score

    if not best_term or best_score < 78:
        return None
    return best_term, best_score


def _should_apply_phrase(
    text: str,
    span: tuple[int, int],
    phrase: str,
    candidate: LexiconTerm,
    score: float,
) -> bool:
    phonetic = _phonetic_score(phrase, candidate.term) >= 92
    same_phrase_start = _compatible_start(phrase.split()[0], candidate.term.split()[0])

    if candidate.kind == "sql_operator":
        if candidate.operator_type == "COMPARE":
            return (
                same_phrase_start
                and _near_number_or_numeric_field(text, span, fuzzy_domain=True)
                and (score >= 84 or phonetic)
            )
        return False
    if candidate.kind == "nutrition_qualifier":
        if re.search(r"\d", phrase):
            return False
        return score >= 88 or (score >= 82 and phonetic)
    if candidate.kind == "time":
        return _near_number_or_time_unit(text, span) and (score >= 84 or phonetic)
    if candidate.kind == "sort":
        return score >= 90 or (score >= 84 and phonetic)
    return False


def _should_apply_token(
    text: str,
    span: tuple[int, int],
    token: str,
    candidate: LexiconTerm,
    score: float,
) -> bool:
    phonetic = _phonetic_score(token, candidate.term) >= 92
    same_start = token[:1] == candidate.term[:1]
    compatible_start = _compatible_start(token, candidate.term)

    if candidate.kind == "sql_operator":
        return _operator_context_supports(text, span, candidate, score, phonetic, same_start)
    if candidate.kind == "nutrition":
        return _nutrition_context_supports(text, span, token, candidate.term, score, phonetic, compatible_start)
    if candidate.kind == "time":
        if _looks_like_inflection(token, candidate.term):
            return False
        return _time_context_supports(text, span, score, phonetic, same_start)
    if candidate.kind in {"cuisine", "diet", "ingredient_flavor"}:
        return len(candidate.term) >= 5 and score >= 88 and compatible_start
    if candidate.kind in {"sql_field", "sql_table"}:
        return score >= 92 and (same_start or phonetic)
    if score >= 94 and (same_start or phonetic):
        return True
    return False


def _operator_context_supports(
    text: str,
    span: tuple[int, int],
    candidate: LexiconTerm,
    score: float,
    phonetic: bool,
    same_start: bool,
) -> bool:
    if candidate.operator_type != "COMPARE":
        return False
    if not _near_number_or_numeric_field(text, span):
        return False
    if score >= 86 and same_start:
        return True
    return score >= 78 and phonetic


def _nutrition_context_supports(
    text: str,
    span: tuple[int, int],
    token: str,
    target: str,
    score: float,
    phonetic: bool,
    same_start: bool,
) -> bool:
    if not _near_number_or_numeric_field(text, span):
        return False
    if not same_start:
        return False
    if not _single_token_spelling_supported(token, target, phonetic):
        return False
    if score >= 88 and same_start:
        return True
    return score >= 78 and phonetic


def _single_token_spelling_supported(source: str, target: str, phonetic: bool) -> bool:
    source_compact = _compact_term(source)
    target_compact = _compact_term(target)
    if not source_compact or not target_compact:
        return False
    lexical = float(fuzz.WRatio(source_compact, target_compact))
    if lexical >= 84:
        return True
    if not phonetic:
        return False
    edit_distance = Levenshtein.distance(source_compact, target_compact)
    max_edit_distance = max(1, min(3, len(target_compact) // 3))
    return lexical >= 65 and edit_distance <= max_edit_distance


def _time_context_supports(
    text: str,
    span: tuple[int, int],
    score: float,
    phonetic: bool,
    same_start: bool,
) -> bool:
    if not _near_number_or_time_unit(text, span):
        return False
    return (score >= 86 and same_start) or (score >= 78 and phonetic)


def _looks_like_inflection(source: str, target: str) -> bool:
    return (
        source.startswith(target)
        and len(source) > len(target)
        and len(source) - len(target) <= 3
    )


def _near_number_or_numeric_field(
    text: str,
    span: tuple[int, int],
    *,
    fuzzy_domain: bool = False,
) -> bool:
    window = _context_window(text, span).lower()
    if re.search(r"\d", window):
        return True
    if fuzzy_domain and _near_numeric_domain_token(window):
        return True
    numeric_terms = {
        term
        for term, lexicon_term in _single_token_terms().items()
        if lexicon_term.kind in {"nutrition", "time", "sql_field"}
    }
    window_terms = set(re.findall(r"[a-z][a-z-]{2,}", window))
    return bool(window_terms & numeric_terms)


def _near_number_or_time_unit(text: str, span: tuple[int, int]) -> bool:
    window = _context_window(text, span).lower()
    if re.search(r"\d", window):
        return True
    return bool(re.search(r"\b(?:minute|minutes|min|mins|hour|hours|hr|hrs)\b", window))


def _context_window(text: str, span: tuple[int, int], radius: int = 48) -> str:
    return text[max(0, span[0] - radius):span[1] + radius]


@lru_cache(maxsize=1)
def _single_token_terms() -> dict[str, LexiconTerm]:
    terms: dict[str, LexiconTerm] = {}
    _add_sql_operator_terms(terms)
    _add_schema_terms(terms)
    _add_recipe_config_terms(terms)
    return terms


@lru_cache(maxsize=1)
def _phrase_terms() -> dict[str, LexiconTerm]:
    terms: dict[str, LexiconTerm] = {}
    _add_sql_operator_terms(terms, phrases_only=True)
    return terms


@lru_cache(maxsize=1)
def _numeric_domain_terms() -> set[str]:
    return {
        term
        for term, lexicon_term in _single_token_terms().items()
        if lexicon_term.kind in {"nutrition", "time", "sql_field"}
    }


@lru_cache(maxsize=1)
def _compare_phrase_terms() -> dict[str, LexiconTerm]:
    choices: dict[str, LexiconTerm] = {}
    for term, lexicon_term in _single_token_terms().items():
        if lexicon_term.kind == "sql_operator" and lexicon_term.operator_type == "COMPARE":
            choices[term] = lexicon_term
    for term, lexicon_term in _phrase_terms().items():
        if lexicon_term.kind == "sql_operator" and lexicon_term.operator_type == "COMPARE":
            choices[term] = lexicon_term
    return choices


@lru_cache(maxsize=1)
def _role_cue_terms() -> dict[str, LexiconTerm]:
    terms: dict[str, LexiconTerm] = {}
    try:
        schema = load_sql_schema_config()
    except (OSError, json.JSONDecodeError):
        schema = {}

    config = schema.get("nutrition_presence", {}) if isinstance(schema, dict) else {}
    if isinstance(config, dict):
        for term in config.get("cues", []):
            _add_role_cue_term(
                terms,
                term=term,
                kind="nutrition_presence_cue",
                source="sql_schema.nutrition_presence.cues",
            )
        for term in config.get("broad_alias_cues", []):
            _add_role_cue_term(
                terms,
                term=term,
                kind="nutrition_presence_cue",
                source="sql_schema.nutrition_presence.broad_alias_cues",
            )
        for term in config.get("negation_cues", []):
            _add_role_cue_term(
                terms,
                term=term,
                kind="nutrition_negation_cue",
                source="sql_schema.nutrition_presence.negation_cues",
            )

    return terms


@lru_cache(maxsize=1)
def _presence_target_terms() -> frozenset[str]:
    terms: set[str] = set()
    try:
        schema = load_sql_schema_config()
    except (OSError, json.JSONDecodeError):
        return frozenset()

    nutrition_columns = schema.get("relations", {}).get("nutrition", {}).get("columns", {})
    for term in nutrition_columns:
        _add_presence_target_term(terms, term)

    nutrition_aliases = schema.get("semantic_aliases", {}).get("nutrition", {})
    for alias, target in nutrition_aliases.items():
        _add_presence_target_term(terms, alias)
        _add_presence_target_term(terms, target)

    for group, config in schema.get("nutrition_groups", {}).items():
        _add_presence_target_term(terms, group)
        if not isinstance(config, dict):
            continue
        for alias in config.get("aliases", []):
            _add_presence_target_term(terms, alias)
        for field in config.get("fields", []):
            _add_presence_target_term(terms, field)

    presence_config = schema.get("nutrition_presence", {})
    if isinstance(presence_config, dict):
        for term in presence_config.get("presence_alias_required_terms", []):
            _add_presence_target_term(terms, term)

    return frozenset(terms)


@lru_cache(maxsize=1)
def _presence_single_target_terms() -> frozenset[str]:
    return frozenset(term for term in _presence_target_terms() if " " not in term)


@lru_cache(maxsize=1)
def _presence_target_choices() -> dict[str, LexiconTerm]:
    return {
        term: LexiconTerm(
            term=term,
            kind="nutrition_presence_target",
            source="sql_schema.nutrition_terms",
        )
        for term in _presence_single_target_terms()
        if len(term) >= 4
    }


def _add_sql_operator_terms(
    terms: dict[str, LexiconTerm],
    *,
    phrases_only: bool = False,
) -> None:
    try:
        payload = json.loads(SQL_OPERATOR_LEXICON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for term, config in payload.get("terms", {}).items():
        _add_term(
            terms,
            term=term,
            kind="sql_operator",
            source="sql_operator_lexicon",
            operator_type=str(config.get("type") or ""),
            phrases_only=phrases_only,
        )


def _add_recipe_config_terms(terms: dict[str, LexiconTerm]) -> None:
    try:
        payload = load_recipe_db_config()
    except (OSError, json.JSONDecodeError):
        return
    for term in payload.get("cuisine_aliases", {}):
        _add_term(terms, term=term, kind="cuisine", source="recipe_db.cuisine_aliases")
    for term in payload.get("diet_aliases", {}):
        _add_term(terms, term=term, kind="diet", source="recipe_db.diet_aliases")
    for term in payload.get("time_fields", {}).get("terms", {}):
        _add_term(terms, term=term, kind="time", source="recipe_db.time_fields")
    for term in payload.get("time_fields", {}).get("quick_terms", []):
        _add_term(terms, term=term, kind="time", source="recipe_db.time_fields.quick_terms")


def _add_recipe_phrase_terms(terms: dict[str, LexiconTerm]) -> None:
    try:
        payload = load_recipe_db_config()
    except (OSError, json.JSONDecodeError):
        return

    for rule in payload.get("intent_enrichment", {}).get("nutrition_qualifiers", []):
        if not isinstance(rule, dict):
            continue
        for term in rule.get("terms", []):
            _add_term(
                terms,
                term=term,
                kind="nutrition_qualifier",
                source="recipe_db.intent_enrichment.nutrition_qualifiers",
                phrases_only=True,
            )

    for values in payload.get("sort_direction_phrases", {}).values():
        for term in values:
            _add_term(terms, term=term, kind="sort", source="recipe_db.sort_direction_phrases", phrases_only=True)

    for aliases in payload.get("sort_terms", {}).values():
        for term in aliases:
            _add_term(terms, term=term, kind="sort", source="recipe_db.sort_terms", phrases_only=True)


def _add_schema_terms(terms: dict[str, LexiconTerm]) -> None:
    try:
        schema = load_sql_schema_config()
    except (OSError, json.JSONDecodeError):
        return

    nutrition_columns = schema.get("relations", {}).get("nutrition", {}).get("columns", {})
    constrained_nutrition = set(schema.get("value_constraints", {}).get("nutrition", {}))
    for term in nutrition_columns:
        if constrained_nutrition and term not in constrained_nutrition:
            continue
        _add_term(terms, term=term, kind="nutrition", source="sql_schema.relations.nutrition")

    nutrition_aliases = schema.get("semantic_aliases", {}).get("nutrition", {})
    for term in nutrition_aliases:
        _add_term(terms, term=term, kind="nutrition", source="sql_schema.semantic_aliases.nutrition")

    for group, config in schema.get("nutrition_groups", {}).items():
        _add_term(terms, term=group, kind="nutrition", source="sql_schema.nutrition_groups")
        for alias in config.get("aliases", []):
            _add_term(terms, term=alias, kind="nutrition", source="sql_schema.nutrition_groups")


def _add_term(
    terms: dict[str, LexiconTerm],
    *,
    term: Any,
    kind: str,
    source: str,
    operator_type: str = "",
    phrases_only: bool = False,
) -> None:
    normalized = _normalize_term(str(term))
    if not normalized or len(normalized.replace(" ", "")) < 4:
        return
    is_phrase = len(re.findall(r"[a-z0-9]+", normalized)) > 1
    if phrases_only != is_phrase:
        return
    terms.setdefault(
        normalized,
        LexiconTerm(
            term=normalized,
            kind=kind,
            source=source,
            operator_type=operator_type,
        ),
    )


def _add_role_cue_term(
    terms: dict[str, LexiconTerm],
    *,
    term: Any,
    kind: str,
    source: str,
) -> None:
    normalized = _normalize_term(str(term))
    if not normalized or len(_compact_term(normalized)) < 3:
        return
    terms.setdefault(
        normalized,
        LexiconTerm(
            term=normalized,
            kind=kind,
            source=source,
        ),
    )


def _add_presence_target_term(terms: set[str], term: Any) -> None:
    normalized = _normalize_word_sequence(str(term))
    if not normalized or len(normalized.replace(" ", "")) < 4:
        return
    terms.add(normalized)


def _combined_similarity(source: str, target: str) -> float:
    lexical = float(fuzz.WRatio(source, target))
    phonetic = _phonetic_score(source, target)
    return max(lexical, phonetic)


def _phrase_similarity(source: str, target: str) -> float:
    lexical = float(fuzz.WRatio(source, target))
    phonetic = _phonetic_score(source, target)
    source_tokens = source.split()
    target_tokens = target.split()
    token_score = 0.0
    if len(source_tokens) == len(target_tokens):
        token_scores = [
            _combined_similarity(source_token, target_token)
            for source_token, target_token in zip(source_tokens, target_tokens)
        ]
        token_score = sum(token_scores) / len(token_scores)
    return max(lexical, phonetic, token_score)


def _phonetic_score(source: str, target: str) -> float:
    source_codes = _phonetic_codes(source)
    target_codes = _phonetic_codes(target)
    if not source_codes or not target_codes:
        return 0.0
    if source_codes & target_codes:
        return 100.0
    return max(
        float(fuzz.ratio(source_code, target_code))
        for source_code in source_codes
        for target_code in target_codes
    )


def _compatible_start(source: str, target: str) -> bool:
    source_start = _start_sound(source)
    target_start = _start_sound(target)
    return bool(source_start and target_start and source_start == target_start)


def _start_sound(value: str) -> str:
    match = re.search(r"[a-z]", value.lower())
    if not match:
        return ""
    char = match.group(0)
    if char in {"c", "k", "q"}:
        return "k"
    if char in {"s", "z", "x"}:
        return "s"
    if char in {"f", "v"}:
        return "f"
    if char in {"d", "t"}:
        return "t"
    if char in {"g", "j"}:
        return "j"
    if char in {"b", "p"}:
        return "p"
    return char


@lru_cache(maxsize=4096)
def _phonetic_codes(value: str) -> frozenset[str]:
    normalized = _normalize_term(value)
    compact = re.sub(r"[^a-z]+", "", normalized)
    codes = {code.lower() for code in doublemetaphone(compact) if code}
    asr_key = _asr_sound_key(compact)
    if asr_key:
        codes.add(asr_key)
    return frozenset(codes)


def _asr_sound_key(value: str) -> str:
    value = value.lower()
    replacements = (
        ("tion", "shun"),
        ("sion", "shun"),
        ("ph", "f"),
        ("ght", "t"),
        ("gh", "g"),
        ("ck", "k"),
        ("qu", "k"),
        ("wh", "w"),
        ("th", "t"),
        ("dh", "t"),
        ("sh", "s"),
        ("ch", "j"),
    )
    for source, target in replacements:
        value = value.replace(source, target)

    groups = {
        "b": "p", "p": "p",
        "c": "k", "k": "k", "q": "k",
        "d": "t", "t": "t",
        "f": "f", "v": "f", "w": "f",
        "g": "j", "j": "j",
        "l": "l", "r": "r",
        "m": "n", "n": "n",
        "s": "s", "x": "s", "z": "s",
        "y": "y",
    }
    key = []
    for char in value:
        mapped = groups.get(char)
        if not mapped:
            continue
        if not key or key[-1] != mapped:
            key.append(mapped)
    return "".join(key)


def _token_spans(text: str) -> list[TokenSpan]:
    return [
        TokenSpan(match.group(0), match.start(), match.end())
        for match in re.finditer(r"\b[a-zA-Z][a-zA-Z-]*\b|\b\d+(?:\.\d+)?\b", text)
    ]


def _normalize_term(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s-]+", " ", value.lower())
    cleaned = cleaned.replace("_", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_word_sequence(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", value.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _compact_term(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _max_term_tokens(terms: dict[str, LexiconTerm]) -> int:
    return max((len(term.split()) for term in terms), default=1)


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement
