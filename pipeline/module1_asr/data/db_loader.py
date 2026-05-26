import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent

INGREDIENT_CSV = DATA_DIR / "RecipeDB_Ingredient_Category.csv"
UTENSIL_CSV   = DATA_DIR / "utensils.csv"
ACTION_CSV    = DATA_DIR / "actions.csv"

# ─────────────────────────────────────────────────────────
# DIRTY DATA FIXES
# These are confirmed corruptions in the RecipeDB export.
# "salt pepper"  → parsing artifact; "and" was stripped
# "purpose flour" → prefix "all-" was cut during export
# ─────────────────────────────────────────────────────────
INGREDIENT_DIRTY_FIXES = {
    "salt pepper":   "salt and pepper",
    "purpose flour": "all-purpose flour",
}


def load_ingredients() -> dict:
    """
    Returns {ingredient_name: {category, ing_id, frequency}}

    CHANGE from original: previously returned a plain set of names.
    Now returns a dict with metadata so that:
      - frequency can be used for tiebreaking in fuzzy matching
      - category (DietRx) is available for SQL generation downstream
      - dirty rows are cleaned before entering the automaton
    """
    df = pd.read_csv(INGREDIENT_CSV)
    df["ingredient"] = (
        df["ingredient"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(INGREDIENT_DIRTY_FIXES)
    )
    df = df[(df["ingredient"].str.len() > 0) & (df["ingredient"] != "nan")]
    df["IngID"] = pd.to_numeric(df["IngID"], errors="coerce")
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce")
    df = df.dropna(subset=["IngID", "frequency"])

    result = {}
    for row in df.to_dict("records"):
        name = row["ingredient"]
        result[name] = {
            "category":  str(row["Dietrx_Category"]).strip(),
            "ing_id":    int(row["IngID"]),
            "frequency": int(row["frequency"]),
        }
    return result


def load_utensils() -> dict:
    """
    Returns {utensil_name: frequency}

    NEW: was completely absent before. Needed so entity_tagger
    can detect utensil mentions like "skillet", "wok", "blender".

    Uses positional column access (col 0 = name, col 1 = frequency)
    so it works regardless of exact column header spelling.
    """
    df = pd.read_csv(UTENSIL_CSV)
    # Normalize column names: strip whitespace + lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Find name col (first col) and frequency col robustly
    name_col = df.columns[0]
    freq_col = next(
        (c for c in df.columns if "freq" in c),
        df.columns[1]
    )

    df[name_col] = df[name_col].astype(str).str.strip().str.lower()
    df[freq_col] = pd.to_numeric(df[freq_col], errors="coerce")
    df = df[(df[name_col].str.len() > 0) & (df[name_col] != "nan")]
    df = df.dropna(subset=[freq_col])

    return {
        row[name_col]: int(row[freq_col])
        for row in df.to_dict("records")
    }


def load_actions() -> dict:
    """
    Returns {action_name: {frequency, type}}
    type is either 'state_change' or 'procedural'.

    NEW: was completely absent before.
    state_change actions (boil, bake, fry...) are useful SQL filters.
    procedural actions (stir, add, mix...) appear in almost every
    recipe and are NOT useful for filtering — the type field lets
    downstream SQL generation ignore procedural ones.

    Also filters out the null/empty row in the original image data
    (row 2, frequency 1,324,329, no action name).
    """
    df = pd.read_csv(ACTION_CSV)
    df.columns = [c.strip().lower() for c in df.columns]

    name_col = df.columns[0]
    freq_col = next((c for c in df.columns if "freq" in c), df.columns[1])
    type_col = next((c for c in df.columns if "type" in c), None)

    df[name_col] = df[name_col].astype(str).str.strip().str.lower()
    df[freq_col] = pd.to_numeric(df[freq_col], errors="coerce")
    df = df[(df[name_col].str.len() > 0) & (df[name_col] != "nan")]
    df = df.dropna(subset=[freq_col])

    result = {}
    for row in df.to_dict("records"):
        name = row[name_col]
        freq = int(row[freq_col])
        action_type = (
            str(row[type_col]).strip()
            if type_col and str(row[type_col]) != "nan"
            else "procedural"
        )
        result[name] = {"frequency": freq, "type": action_type}
    return result


# Module-level singletons — loaded once at import
DB_INGREDIENTS = load_ingredients()
DB_UTENSILS    = load_utensils()
DB_ACTIONS     = load_actions()

# Plain name sets — used by automaton and extractor
INGREDIENT_NAMES = set(DB_INGREDIENTS.keys())
UTENSIL_NAMES    = set(DB_UTENSILS.keys())
ACTION_NAMES     = set(DB_ACTIONS.keys())
