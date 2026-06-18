# Professor Verification Plan

This project generates SQL from speech/text queries using a fixed RecipeDB schema
contract. The goal of this note is to show how we can prove schema compatibility
before running on the real database, and how we can demonstrate that the query
pipeline is working end to end.

## Current Evidence

Run from the project root:

```powershell
$env:PYTHONPATH='src'
python -m speech_to_sql.m4_sql.schema_audit --skip-pdf
pytest -q
python scripts\stress_verify_queries.py --count 1000 --rebuild-db --report docs\verification\stress_1000_latest.json
```

Latest verified results:

```text
Schema audit: PASS
Pytest: 85 passed
Stress suite: 1000 passed, 0 failed
```

The stress suite covers 100 cases each for:

```text
ingredient_pair
ingredient_exclusion
region_ingredient_instruction
diet_ingredient_flavor
recipe_nutrition
ingredient_nutrition
recipe_and_ingredient_nutrition
instruction_only
base_process_or_utensil
time_instruction_ingredient
```

## Schema Contract

The project currently expects these physical tables:

```text
recipes
micronutrients
ingredient_phrase
ingredient_nutrition
ingredient_flavor
instructions
```

The audited column counts are:

```text
ingredient_flavor: 12
ingredient_nutrition: 153
ingredient_phrase: 13
instructions: 3
micronutrients: 167
recipes: 25
```

The required join relationships are:

```text
micronutrients.recipe_id -> recipes.recipe_id
ingredient_phrase.recipe_no -> recipes.recipe_id
ingredient_nutrition.recipe_id -> recipes.recipe_id
instructions.recipe_id -> recipes.recipe_id
ingredient_phrase.ing_id -> ingredient_flavor.ingid
ingredient_phrase.ndb_id -> ingredient_nutrition.ndb_id
```

## What This Proves

The pipeline has been validated against a dummy SQLite database generated from
the same schema contract used by the SQL generator. This means:

- generated SQL uses only configured table and column names;
- multi-table joins are generated through configured relationships;
- recipe-level nutrition uses `micronutrients`;
- ingredient-level nutrition uses `ingredient_nutrition`;
- ingredient phrases use `ingredient_phrase`;
- flavor/category joins use `ingredient_flavor`;
- instruction-step filters use `instructions`;
- query outputs are executed and row-verified on the dummy database.

## What Still Must Be Checked On The Real Database

Before claiming that a real professor-provided database will work, we must check
that its physical schema matches the contract above.

If the real DB has the same table and column names, the generated SQL should run
smoothly.

If the real DB uses different physical names, for example prefixed names such as
`RecipeDB_micronutrients_v1`, then the pipeline logic does not need to be
rewritten, but the schema mapping must be updated or database views must be
created.

## Real Database Compatibility Check

For Athena or any SQL database with `information_schema`, first check table names:

```sql
SELECT table_name
FROM information_schema.tables
WHERE table_name IN (
  'recipes',
  'micronutrients',
  'ingredient_phrase',
  'ingredient_nutrition',
  'ingredient_flavor',
  'instructions'
)
ORDER BY table_name;
```

Then check columns:

```sql
SELECT table_name, column_name
FROM information_schema.columns
WHERE table_name IN (
  'recipes',
  'micronutrients',
  'ingredient_phrase',
  'ingredient_nutrition',
  'ingredient_flavor',
  'instructions'
)
ORDER BY table_name, ordinal_position;
```

Compare this output with `config/schema.sql` or `config/sql_schema.json`.

## Suggested Professor Demo Flow

1. Show the schema audit:

```powershell
$env:PYTHONPATH='src'
python -m speech_to_sql.m4_sql.schema_audit --skip-pdf
```

2. Rebuild the dummy database from the schema:

```powershell
python temp_recipe_db\seed_temp_db.py
```

3. Run the 1000-query verifier:

```powershell
$env:PYTHONPATH='src'
python scripts\stress_verify_queries.py --count 1000 --rebuild-db --report docs\verification\stress_1000_latest.json
```

4. Run one visible CLI query:

```powershell
python scripts\run_cli.py
```

Recommended query:

```text
vegan recipes with spinach under 300 calories
```

5. If the real database is provided, run the compatibility check first. If it
matches, run generated SQL on the real database. If it does not match, update
the schema mapping or create compatibility views.

## API Fallback

The RecipeDB API also works on the IIITD network using:

```text
RDB2_API_KEY
RDB2_BASE_URL=http://cosylab.iiitd.edu.in:6969
```

The project uses:

```text
Authorization: Bearer <key>
```

The live API has a trial limit of 25 requests per minute, so demonstrations
should use a small number of carefully selected queries.
