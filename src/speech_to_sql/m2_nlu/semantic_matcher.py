import numpy as np

from speech_to_sql.m2_nlu.embedding_model import get_model
from speech_to_sql.m2_nlu.embedding_index import ALL_TERMS, TERM_EMBEDDINGS
from speech_to_sql.m2_nlu.data.ingredient_synonyms import canonicalize_ingredient


def semantic_match(query: str, threshold: float = 0.78) -> dict | None:
    """
    Stage 3 semantic similarity match.
    Only called for tokens > 4 chars that all prior stages missed.

    Args:
        query:     Single token or short phrase to match.
        threshold: Cosine similarity cutoff (0-1). Default 0.78
                   matches the value used in extractor.py.
                   The original semantic_matcher.py had 0.72
                   as default but extractor.py called it with
                   0.78 - we use 0.78 here to make it consistent.

    Returns:
        {"matched": str, "canonical": str, "score": float} or None
    """
    model = get_model()

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    similarities = np.dot(TERM_EMBEDDINGS, query_embedding)

    best_idx   = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score < threshold:
        return None

    matched_term = ALL_TERMS[best_idx]

    # SYNONYMS maps alias -> canonical ingredient name.
    # If the matched term is a DB ingredient (not an alias),
    # SYNONYMS.get returns the term itself as canonical.
    canonical = canonicalize_ingredient(matched_term)

    return {
        "matched":   matched_term,
        "canonical": canonical,
        "score":     best_score,
    }
