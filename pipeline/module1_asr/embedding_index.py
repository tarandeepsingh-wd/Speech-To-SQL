import numpy as np
from pathlib import Path

from module1_asr.data.db_loader import DB_INGREDIENTS
from module1_asr.data.ingredient_synonyms import SYNONYMS

# ─────────────────────────────────────────────────────────
# CHANGE 1: Was computing embeddings at import time.
#
# WHY THIS IS WRONG:
#   The original code ran MODEL.encode(ALL_TERMS) the moment
#   the module was imported. ALL_TERMS has ~5,000 strings.
#   Encoding 5,000 strings takes 15-30 seconds on CPU.
#   This ran EVERY time the app restarted — even when nothing
#   had changed.
#
# FIX: Save the embeddings to disk after the first compute.
#   Subsequent runs load from disk in under 1 second.
#   Cache is invalidated only if ALL_TERMS changes (we
#   compare a hash of the sorted term list).
#
# CHANGE 2: Model is now loaded lazily via get_model()
#   so this file can be imported without triggering
#   SentenceTransformer to load.
# ─────────────────────────────────────────────────────────

_CACHE_DIR  = Path(__file__).parent / "data" / "_cache"
_TERMS_FILE = _CACHE_DIR / "ingredient_terms.npy"
_EMBED_FILE = _CACHE_DIR / "ingredient_embeddings.npy"

# Only canonical ingredient names and synonym aliases go here.
# CATEGORY_SYNONYMS are NOT included — categories are separate
# entities handled by entity_tagger, not the ingredient index.
ALL_TERMS = list(set(
    [x.lower().strip() for x in DB_INGREDIENTS] +
    [x.lower().strip() for x in SYNONYMS.keys()]
))

# Stable sort so the hash is deterministic
ALL_TERMS = sorted(ALL_TERMS)


def _terms_hash() -> str:
    """Cheap fingerprint of the current term list."""
    return str(hash(tuple(ALL_TERMS)))


def _load_or_build() -> np.ndarray:
    """Load embeddings from disk cache, or build and save them."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if cached version matches current term list
    if _TERMS_FILE.exists() and _EMBED_FILE.exists():
        try:
            cached_terms = list(np.load(_TERMS_FILE, allow_pickle=True))
            if cached_terms == ALL_TERMS:
                return np.load(_EMBED_FILE)
        except Exception:
            pass  # corrupted cache — rebuild below

    # Build: only happens on first run or when ingredients change
    print("[embedding_index] Building embedding cache — this runs once...")
    from module1_asr.embedding_model import get_model
    model = get_model()
    embeddings = model.encode(
        ALL_TERMS,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )
    np.save(_TERMS_FILE, np.array(ALL_TERMS, dtype=object))
    np.save(_EMBED_FILE, embeddings)
    print("[embedding_index] Cache saved.")
    return embeddings


# Module-level — but only loads from disk (fast), never
# re-encodes unless the term list has changed.
TERM_EMBEDDINGS: np.ndarray = _load_or_build()