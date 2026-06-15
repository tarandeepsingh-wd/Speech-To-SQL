from sentence_transformers import SentenceTransformer

_MODEL = None


def get_model() -> SentenceTransformer:
    """Return the embedding model, loading it on first call."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL
