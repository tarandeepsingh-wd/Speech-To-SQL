import numpy as np

from module1_asr.embedding_model import MODEL
from module1_asr.data.db_loader import DB_INGREDIENTS
from module1_asr.data.ingredient_synonyms import SYNONYMS
ALL_TERMS = list(set(
    list(DB_INGREDIENTS) +
    list(SYNONYMS.keys())
))
TERM_EMBEDDINGS = MODEL.encode(
    ALL_TERMS,
    convert_to_numpy=True,
    normalize_embeddings=True
)