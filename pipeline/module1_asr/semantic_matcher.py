import numpy as np

from module1_asr.embedding_model import MODEL
from module1_asr.embedding_index import (
    ALL_TERMS,
    TERM_EMBEDDINGS
)

from module1_asr.data.ingredient_synonyms import SYNONYMS
def semantic_match(query, threshold=0.72):
    
    query_embedding = MODEL.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities = np.dot(
        TERM_EMBEDDINGS,
        query_embedding
    )

    best_idx = np.argmax(similarities)

    best_score = similarities[best_idx]

    if best_score < threshold:
        return None

    matched_term = ALL_TERMS[best_idx]

    canonical = SYNONYMS.get(
        matched_term,
        matched_term
    )

    return {
        "matched": matched_term,
        "canonical": canonical,
        "score": float(best_score)
    }