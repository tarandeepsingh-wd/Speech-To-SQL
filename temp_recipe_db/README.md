# Temp RecipeDB SQL Fixture

This folder is only for local SQL verification. It does not change the main
speech-to-SQL package logic.

Build the temporary SQLite database:

```powershell
$env:PYTHONPATH="src"
python temp_recipe_db\seed_temp_db.py
```

Show the actual table names, row counts, and table columns:

```powershell
python temp_recipe_db\run_query.py --schema
```

Run a natural-language query through the existing pipeline and execute the
generated SQL against the fixture:

```powershell
python temp_recipe_db\run_query.py --rebuild "show recipes that contain thiamine and chicken"
python temp_recipe_db\run_query.py "vegan recipes with spinach under 300 calories" --brief --explain
python temp_recipe_db\run_query.py "recipes rich in folate"
```

Run raw SQL directly against the same temp tables:

```powershell
python temp_recipe_db\run_query.py --sql "SELECT recipe_id, recipe_title FROM micronutrients WHERE calories < ?" --params "[400]"
```

The generated database file is `recipe_test.db` and can be deleted/rebuilt at
any time. The fixture now creates 100 recipe rows plus matching ingredient,
ingredient-nutrition, and instruction rows.
