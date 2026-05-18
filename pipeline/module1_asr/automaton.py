
import ahocorasick
from module1_asr.data.db_loader import DB_INGREDIENTS
from module1_asr.data.ingredient_synonyms import SYNONYMS


def build_automaton() -> ahocorasick.Automaton:
    A = ahocorasick.Automaton()

    # exact DB ingredients
    for ingredient in DB_INGREDIENTS:
        ingredient = ingredient.lower().strip()

        A.add_word(
            ingredient,
            ("db", ingredient, ingredient)
        )

    # synonym aliases
    for alias, canonical in SYNONYMS.items():
        alias = alias.lower().strip()
        canonical = canonical.lower().strip()

        A.add_word(
            alias,
            ("synonym", alias, canonical)
        )

    A.make_automaton()
    return A


AUTOMATON = build_automaton()