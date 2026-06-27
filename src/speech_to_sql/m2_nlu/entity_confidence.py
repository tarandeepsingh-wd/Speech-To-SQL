"""Self-supervised confidence scoring for utensil and cooking-action matches."""

from __future__ import annotations

import math
import hashlib
import json
import pickle
import re
from functools import lru_cache
from pathlib import Path

from Levenshtein import distance as levenshtein_distance
from metaphone import doublemetaphone
from rapidfuzz import fuzz, process
from sklearn.linear_model import LogisticRegression

from speech_to_sql.m2_nlu.data.db_loader import DB_ACTIONS, DB_INGREDIENTS, DB_UTENSILS


MODEL_SCHEMA_VERSION = 2
MODEL_DIR = Path(__file__).parent / "models" / "entity_confidence"
EXACT_LAYERS = {"exact_aho", "exact_direct"}
MORPHOLOGY_LAYERS = {"exact_morphology"}
MAX_FREQUENCY = max(
    [*DB_UTENSILS.values(), *(item["frequency"] for item in DB_ACTIONS.values())],
    default=1,
)


def _normalize(value: str) -> str:
    value = re.sub(r"[^a-z0-9\\s]", " ", str(value).lower())
    return re.sub(r"\\s+", " ", value).strip()


def _consonant_key(value: str) -> str:
    return re.sub(r"[aeiou\\s]", "", value)


def _asr_form(value: str) -> str:
    collapsed = re.sub(r"(.)\1+", r"\1", value.lower())
    return re.sub(r"[aeiou]", "A", collapsed)


def _edit_similarity(left: str, right: str) -> float:
    return 1.0 - (
        levenshtein_distance(left, right) / max(len(left), len(right), 1)
    )


def _single_adjacent_transposition(left: str, right: str) -> float:
    left = left.replace(" ", "")
    right = right.replace(" ", "")
    if len(left) != len(right) or left == right or len(left) < 2:
        return 0.0
    diffs = [index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]]
    if len(diffs) != 2 or diffs[1] != diffs[0] + 1:
        return 0.0
    first, second = diffs
    return float(left[first] == right[second] and left[second] == right[first])


def _metaphone_overlap(left: str, right: str) -> float:
    left_codes = {code for code in doublemetaphone(left) if code}
    right_codes = {code for code in doublemetaphone(right) if code}
    return float(bool(left_codes & right_codes))


def _features(surface: str, canonical: str, frequency: int) -> list[float]:
    surface = _normalize(surface)
    canonical = _normalize(canonical)
    compact_surface = surface.replace(" ", "")
    compact_canonical = canonical.replace(" ", "")
    consonant_surface = _consonant_key(surface)
    consonant_canonical = _consonant_key(canonical)
    return [
        fuzz.ratio(surface, canonical) / 100.0,
        fuzz.WRatio(surface, canonical) / 100.0,
        _edit_similarity(surface, canonical),
        _edit_similarity(consonant_surface, consonant_canonical),
        fuzz.ratio(_asr_form(surface), _asr_form(canonical)) / 100.0,
        _metaphone_overlap(surface, canonical),
        float(bool(compact_surface and compact_canonical and compact_surface[0] == compact_canonical[0])),
        _single_adjacent_transposition(compact_surface, compact_canonical),
        float(bool(consonant_surface and consonant_surface == consonant_canonical)),
        math.log1p(max(frequency, 0)) / math.log1p(MAX_FREQUENCY),
    ]


def _mutations(term: str) -> set[str]:
    """Create deterministic ASR/typing perturbations without term-specific rules."""
    compact = _normalize(term)
    if len(compact) < 4:
        return set()

    midpoint = len(compact) // 2
    variants = {
        compact[:midpoint] + compact[midpoint + 1:],
        compact[:midpoint] + compact[midpoint] + compact[midpoint:],
    }
    if midpoint > 0:
        variants.add(compact[:midpoint - 1] + compact[midpoint:])
    if midpoint + 1 < len(compact):
        variants.add(compact[:midpoint] + compact[midpoint + 1] + compact[midpoint] + compact[midpoint + 2:])
    compact_no_vowels = re.sub(r"[aeiou]", "", compact)
    if len(compact_no_vowels) >= 3:
        variants.add(compact_no_vowels)
    for index in range(len(compact) - 1):
        variants.add(compact[:index] + compact[index + 1] + compact[index] + compact[index + 2:])
    for vowel in "aeiou":
        if vowel in compact:
            variants.add(compact.replace(vowel, "a", 1))
            break
    first = compact[:1]
    for replacement in {
        "c": "k", "k": "c", "s": "z", "z": "s", "f": "v", "v": "f",
        "b": "p", "p": "b", "d": "t", "t": "d", "o": "u", "u": "o",
        "a": "e", "e": "a",
    }.get(first, ""):
        variants.add(replacement + compact[1:])
    return {variant for variant in variants if variant and variant != compact}


@lru_cache(maxsize=2)
def _vocabulary(kind: str) -> dict[str, int]:
    result: dict[str, int] = {}
    if kind == "utensil":
        source = DB_UTENSILS.items()
    elif kind == "action":
        source = ((name, metadata["frequency"]) for name, metadata in DB_ACTIONS.items())
    else:
        raise ValueError(f"Unsupported entity kind: {kind}")

    for name, frequency in source:
        key = _normalize(str(name))
        if key:
            result[key] = max(result.get(key, 0), int(frequency))
    return result


def _negative_surfaces(kind: str, candidates: list[str]) -> list[tuple[str, str]]:
    """Use other entity vocabularies to learn likely cross-type collisions."""
    own_terms = set(candidates)
    competing_terms = set(DB_INGREDIENTS)
    competing_terms.update(DB_ACTIONS if kind == "utensil" else DB_UTENSILS)
    competing_choices = sorted({
        _normalize(term)
        for term in competing_terms
        if _normalize(term) and _normalize(term) not in own_terms
    })

    negatives: set[tuple[str, str]] = set()
    for candidate in candidates:
        for surface, score, _ in process.extract(
            candidate,
            competing_choices,
            scorer=fuzz.WRatio,
            score_cutoff=55,
            limit=24,
        ):
            if len(surface) >= 3:
                negatives.add((surface, candidate))
    return sorted(negatives)


def _vocabulary_signature(kind: str) -> str:
    payload = json.dumps(_vocabulary(kind), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_path(kind: str) -> Path:
    return MODEL_DIR / f"{kind}_confidence.pkl"


def _load_saved_model(kind: str):
    path = _model_path(kind)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    if (
        payload.get("schema_version") != MODEL_SCHEMA_VERSION
        or payload.get("vocabulary_signature") != _vocabulary_signature(kind)
    ):
        return None
    return payload.get("model")


def _save_model(kind: str, model: LogisticRegression) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "vocabulary_signature": _vocabulary_signature(kind),
        "model": model,
    }
    with _model_path(kind).open("wb") as handle:
        pickle.dump(payload, handle)


@lru_cache(maxsize=2)
def _model(kind: str) -> LogisticRegression:
    saved_model = _load_saved_model(kind)
    if saved_model is not None:
        return saved_model

    vocabulary = _vocabulary(kind)
    candidates = sorted(_normalize(term) for term in vocabulary)
    rows: list[list[float]] = []
    labels: list[int] = []

    for candidate in candidates:
        frequency = vocabulary[candidate]
        for surface in {candidate, *_mutations(candidate)}:
            rows.append(_features(surface, candidate, frequency))
            labels.append(1)

    for surface, candidate in _negative_surfaces(kind, candidates):
        rows.append(_features(surface, candidate, vocabulary[candidate]))
        labels.append(0)

    # Pairing canonical terms with the wrong candidate teaches the model to
    # distinguish nearby CSV labels such as "saucer" and "saucepan".
    for index, candidate in enumerate(candidates):
        wrong_candidate = candidates[(index + 1) % len(candidates)]
        rows.append(_features(candidate, wrong_candidate, vocabulary[wrong_candidate]))
        labels.append(0)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=17,
    )
    model.fit(rows, labels)
    _save_model(kind, model)
    return model


def match_confidence(kind: str, surface: str, canonical: str, layer: str) -> float:
    """Return a calibrated probability that a candidate belongs to its CSV lane."""
    if layer in EXACT_LAYERS:
        return 1.0
    if layer in MORPHOLOGY_LAYERS:
        return 0.98

    vocabulary = _vocabulary(kind)
    frequency = vocabulary.get(_normalize(canonical), 0)
    probability = _model(kind).predict_proba([
        _features(surface, canonical, frequency)
    ])[0][1]
    return round(float(probability), 3)


def accepts_match(kind: str, surface: str, canonical: str, layer: str) -> bool:
    """Use the model's natural 0.5 decision boundary for non-exact matches."""
    return match_confidence(kind, surface, canonical, layer) >= 0.5
