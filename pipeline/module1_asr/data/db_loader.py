import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).parent / "RecipeDB_Ingredient_Category.csv"

def load_db() -> set:
    df = pd.read_csv(CSV_PATH)
    return set(df["ingredient"].tolist())

DB_INGREDIENTS = load_db()