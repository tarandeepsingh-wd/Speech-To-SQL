import math
import os
import pickle
import re
from pathlib import Path

from metaphone import doublemetaphone
from rapidfuzz import fuzz

from module1_asr.data.db_loader import DB_INGREDIENTS


MODEL_DIR = Path(__file__).parent / "models" / "ingredient_reranker"
MODEL_FILE = MODEL_DIR / "model.pkl"
DEFAULT_THRESHOLD = 0.50
FEATURE_SCHEMA_VERSION = 2

NUMERIC_FEATURE_NAMES = [
    "extractor_score",
    "surface_len",
    "canonical_len",
    "length_ratio",
    "token_count",
    "relative_start",
    "is_exact",
    "is_exact_component",
    "is_synonym_source",
    "is_fuzzy",
    "is_metaphone",
    "is_neighbor",
    "is_rescue",
    "is_semantic",
    "surface_canonical_wratio",
    "surface_canonical_ratio",
    "metaphone_overlap",
    "near_number",
    "canonical_log_frequency",
]

_MODEL_CACHE = {
    "path": None,
    "mtime": None,
    "payload": None,
}


def _score(match: dict) -> float:
    value = match.get("score")
    if value is None:
        return 100.0 if match.get("layer", "").startswith("exact") else 0.0
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score <= 1.0:
        score *= 100.0
    return max(0.0, min(score, 100.0))


def _metaphone_overlap(surface: str, canonical: str) -> float:
    surface_codes = {code for code in doublemetaphone(surface) if code}
    canonical_codes = {code for code in doublemetaphone(canonical) if code}
    if not surface_codes or not canonical_codes:
        return 0.0
    return float(bool(surface_codes & canonical_codes))


def _context_text(text: str, start: int, end: int, window: int = 36) -> str:
    left = text[max(0, start - window):start]
    right = text[end + 1:min(len(text), end + 1 + window)]
    return f"{left} {right}".lower()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z-]*|\d+(?:\.\d+)?", text.lower())


def _feature_key(prefix: str, value: str) -> str:
    value = re.sub(r"\s+", "_", value.lower().strip())
    return f"{prefix}={value}" if value else f"{prefix}=<empty>"


def _canonical_frequency(canonical: str) -> float:
    payload = DB_INGREDIENTS.get(canonical.lower().strip(), {})
    try:
        frequency = float(payload.get("frequency", 0))
    except (TypeError, ValueError, AttributeError):
        frequency = 0.0
    return math.log1p(max(frequency, 0.0)) / 16.0


def build_candidate_features(query: str, match: dict) -> dict[str, float]:
    """Build data-derived features for the candidate ingredient classifier."""
    surface = str(match.get("text") or "").lower().strip()
    canonical = str(match.get("canonical") or "").lower().strip()
    layer = str(match.get("layer") or "")
    source = str(match.get("source") or "")
    start, end = match.get("span", (0, -1))
    text_len = max(len(query), 1)
    surface_len = len(surface)
    canonical_len = len(canonical)
    length_ratio = (
        min(surface_len, canonical_len) / max(surface_len, canonical_len)
        if surface_len and canonical_len else 0.0
    )
    left_context = query[max(0, int(start) - 36):int(start)]
    right_context = query[int(end) + 1:min(len(query), int(end) + 37)]
    context = f"{left_context} {right_context}".lower()

    features = {
        "extractor_score": _score(match) / 100.0,
        "surface_len": min(surface_len / 32.0, 1.0),
        "canonical_len": min(canonical_len / 48.0, 1.0),
        "length_ratio": length_ratio,
        "token_count": min(len(surface.split()) / 4.0, 1.0),
        "relative_start": max(0.0, min(float(start) / text_len, 1.0)),
        "is_exact": float(layer.startswith("exact")),
        "is_exact_component": float(layer == "exact_component"),
        "is_synonym_source": float(source == "synonym"),
        "is_fuzzy": float("fuzzy" in layer),
        "is_metaphone": float("metaphone" in layer),
        "is_neighbor": float("neighbor" in layer),
        "is_rescue": float("rescue" in layer),
        "is_semantic": float("semantic" in layer),
        "surface_canonical_wratio": (
            fuzz.WRatio(surface, canonical) / 100.0
            if surface and canonical else 0.0
        ),
        "surface_canonical_ratio": (
            fuzz.ratio(surface, canonical) / 100.0
            if surface and canonical else 0.0
        ),
        "metaphone_overlap": _metaphone_overlap(surface, canonical),
        "near_number": float(bool(re.search(r"\d+(?:\.\d+)?", context))),
        "canonical_log_frequency": _canonical_frequency(canonical),
        _feature_key("surface", surface): 1.0,
        _feature_key("canonical", canonical): 1.0,
        _feature_key("layer", layer): 1.0,
        _feature_key("source", source): 1.0,
    }
    for token in _tokens(left_context)[-4:]:
        features[_feature_key("left", token)] = 1.0
    for token in _tokens(right_context)[:4]:
        features[_feature_key("right", token)] = 1.0
    return features


def _model_path() -> Path:
    configured = os.getenv("TADKA_INGREDIENT_RERANKER_MODEL")
    return Path(configured) if configured else MODEL_FILE


def _load_model():
    path = _model_path()
    if not path.exists():
        _MODEL_CACHE.update({"path": path, "mtime": None, "payload": None})
        return None
    mtime = path.stat().st_mtime
    if (
        _MODEL_CACHE["path"] == path
        and _MODEL_CACHE["mtime"] == mtime
    ):
        return _MODEL_CACHE["payload"]
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        _MODEL_CACHE.update({"path": path, "mtime": mtime, "payload": None})
        return None
    if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        _MODEL_CACHE.update({"path": path, "mtime": mtime, "payload": None})
        return None
    _MODEL_CACHE.update({"path": path, "mtime": mtime, "payload": payload})
    return payload


def _threshold(payload: dict) -> float:
    env_value = os.getenv("TADKA_INGREDIENT_RERANKER_THRESHOLD")
    if env_value:
        try:
            return float(env_value)
        except ValueError:
            pass
    return float(payload.get("threshold", DEFAULT_THRESHOLD))


def _predict_probability(payload: dict, features: list[float]) -> float:
    model = payload["model"]
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba([features])[0][1])
    return float(model.predict([features])[0])


def rerank_ingredient_matches(query: str, matches: list[dict]) -> list[dict]:
    """
    Filter candidate ingredients with a trained candidate classifier.

    If no saved model exists, this deliberately returns the original matches so
    existing demos/tests keep their current behavior until training is run.
    """
    payload = _load_model()
    if not payload or not matches:
        return matches

    cutoff = _threshold(payload)
    kept = []
    for match in matches:
        features = build_candidate_features(query, match)
        probability = _predict_probability(payload, features)
        enriched = dict(match)
        enriched["reranker_confidence"] = round(probability, 3)
        if probability >= cutoff:
            kept.append(enriched)
    return kept
