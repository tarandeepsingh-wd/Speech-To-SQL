import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UTENSILS_CSV = PROJECT_ROOT / "src" / "speech_to_sql" / "m2_nlu" / "data" / "utensils.csv"

print(pd.read_csv(UTENSILS_CSV).columns.tolist())
