import numpy as np
from pathlib import Path

from speech_to_sql.m2_nlu.data.db_loader import DB_INGREDIENTS
from speech_to_sql.m2_nlu.data.ingredient_synonyms import SYNONYMS

# Ingredient embeddings are cached on disk and rebuilt only when the term list
# changes. The sentence-transformer model is loaded lazily during cache rebuild.

_CACHE_DIR  = Path(__file__).parent / "data" / "_cache"
_TERMS_FILE = _CACHE_DIR / "ingredient_terms.npy"
_EMBED_FILE = _CACHE_DIR / "ingredient_embeddings.npy"

# Only canonical ingredient names and synonym aliases go here.
# CATEGORY_SYNONYMS are NOT included - categories are separate
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

    # Check if cached version matches current term list.
    if _TERMS_FILE.exists() and _EMBED_FILE.exists():
        try:
            cached_terms = list(np.load(_TERMS_FILE, allow_pickle=True))
            if cached_terms == ALL_TERMS:
                return np.load(_EMBED_FILE)
        except Exception:
            pass

    print("[embedding_index] Building embedding cache - this runs once...")
    from speech_to_sql.m2_nlu.embedding_model import get_model
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


TERM_EMBEDDINGS: np.ndarray = _load_or_build()
