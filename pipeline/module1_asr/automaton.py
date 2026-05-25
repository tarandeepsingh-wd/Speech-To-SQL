import ahocorasick

from module1_asr.data.db_loader import DB_INGREDIENTS, DB_UTENSILS
from module1_asr.data.ingredient_synonyms import SYNONYMS

# ─────────────────────────────────────────────────────────
# CHANGE: Was one single automaton for ingredients only.
#
# WHY THIS IS WRONG:
#   The system needs to extract utensils too (bowl, pan,
#   skillet, wok etc.) — they are query dimensions just like
#   ingredients. A user saying "recipes that use a skillet"
#   needs "skillet" detected. Without a utensil automaton,
#   the entity_tagger had zero utensil awareness.
#
# FIX: Two separate automatons:
#   INGREDIENT_AUTOMATON — exact DB ingredients + synonym aliases
#   UTENSIL_AUTOMATON    — exact DB utensils
#
# They are kept separate because:
#   1. They feed different extraction passes in extractor.py
#   2. "cup" is both a utensil AND a unit of measurement —
#      keeping them separate lets entity_tagger apply
#      context rules (cup preceded by a number = unit, not
#      utensil) without touching ingredient matching.
#   3. resolve_overlaps in extractor.py uses layer priority
#      to break ties — mixing entity types in one automaton
#      would confuse the priority ordering.
# ─────────────────────────────────────────────────────────


def build_ingredient_automaton() -> ahocorasick.Automaton:
    """
    Builds an Aho-Corasick automaton over:
      - All ingredient names from DB (exact canonical)
      - All synonym aliases from SYNONYMS dict
        (maps alias → canonical)

    NOTE: CATEGORY_SYNONYMS are deliberately NOT included here.
    They belong to a separate entity type (dietary category)
    and feeding them into the ingredient automaton would cause
    the extractor to return DietRx category names like
    "Plant Derivative" as extracted ingredient strings.
    """
    A = ahocorasick.Automaton()

    for ingredient in DB_INGREDIENTS:
        key = ingredient.lower().strip()
        if key:
            A.add_word(key, ("db", key, key))

    for alias, canonical in SYNONYMS.items():
        key      = alias.lower().strip()
        canon    = canonical.lower().strip()
        if key:
            A.add_word(key, ("synonym", key, canon))

    A.make_automaton()
    return A


def build_utensil_automaton() -> ahocorasick.Automaton:
    """
    Builds an Aho-Corasick automaton over utensil names.
    Utensil synonyms (e.g. "food processor" → "processor",
    "cast iron" → "skillet") can be added to a future
    utensil_synonyms.py file and included here the same way
    ingredient synonyms are included above.
    """
    A = ahocorasick.Automaton()

    for utensil in DB_UTENSILS:
        key = utensil.lower().strip()
        if key:
            A.add_word(key, ("db_utensil", key, key))

    A.make_automaton()
    return A


# Module-level singletons — built once at import
INGREDIENT_AUTOMATON = build_ingredient_automaton()
UTENSIL_AUTOMATON    = build_utensil_automaton()