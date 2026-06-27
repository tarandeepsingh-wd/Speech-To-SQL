import ahocorasick

from speech_to_sql.m2_nlu.data.db_loader import DB_ACTIONS, DB_INGREDIENTS, DB_UTENSILS
from speech_to_sql.m2_nlu.data.ingredient_synonyms import SYNONYMS, canonicalize_ingredient
from speech_to_sql.m2_nlu.data.utensil_synonyms import UTENSIL_SYNONYMS

# Ingredients, utensils, and actions are separate automatons because they feed
# different entity types and need different context rules.


def build_ingredient_automaton() -> ahocorasick.Automaton:
    """
    Builds an Aho-Corasick automaton over:
      - All ingredient names from DB (exact canonical)
      - All synonym aliases from SYNONYMS dict
        (maps alias -> canonical)

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
            A.add_word(key, ("db", key, canonicalize_ingredient(key)))

    for alias, canonical in SYNONYMS.items():
        key      = alias.lower().strip()
        canon    = canonicalize_ingredient(canonical)
        if key:
            A.add_word(key, ("synonym", key, canon))

    A.make_automaton()
    return A


def build_utensil_automaton() -> ahocorasick.Automaton:
    """
    Builds an Aho-Corasick automaton over utensil names and aliases.
    """
    A = ahocorasick.Automaton()

    for utensil in DB_UTENSILS:
        key = utensil.lower().strip()
        if key:
            A.add_word(key, ("db_utensil", key, key))

    for alias, canonical in UTENSIL_SYNONYMS.items():
        key = alias.lower().strip()
        canonical = canonical.lower().strip()
        if key and canonical in DB_UTENSILS:
            A.add_word(key, ("utensil_synonym", key, canonical))

    A.make_automaton()
    return A


def build_action_automaton() -> ahocorasick.Automaton:
    """
    Builds an Aho-Corasick automaton over CSV-backed cooking action names.
    """
    A = ahocorasick.Automaton()

    for action in DB_ACTIONS:
        key = action.lower().strip()
        if key:
            A.add_word(key, ("db_action", key, key))

    A.make_automaton()
    return A


# Module-level singletons - built once at import
INGREDIENT_AUTOMATON = build_ingredient_automaton()
UTENSIL_AUTOMATON    = build_utensil_automaton()
ACTION_AUTOMATON     = build_action_automaton()
