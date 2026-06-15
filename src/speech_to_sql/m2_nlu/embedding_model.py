# ---------------------------------------------------------
# CHANGE: Model was loaded at module import time.
#
# WHY THIS IS WRONG:
#   Importing ANY file in the package (automaton, extractor,
#   etc.) would trigger SentenceTransformer to load a 90MB
#   model into memory immediately - even if semantic matching
#   is never used. On CPU this takes 5-10 seconds on cold
#   start, and wastes ~500MB RAM for simple exact-match
#   queries.
#
# FIX: Lazy-load the model behind a function. The model is
#   only loaded the first time semantic_match() is actually
#   called, and cached in _MODEL for subsequent calls.
# ---------------------------------------------------------

from sentence_transformers import SentenceTransformer

_MODEL = None


def get_model() -> SentenceTransformer:
    """Return the embedding model, loading it on first call."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL