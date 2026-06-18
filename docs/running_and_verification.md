# Running And Verification

Run commands from the repository root: `D:\IP_Project`.

## Full Interactive CLI

```powershell
python scripts\run_cli.py
```

Menu:

```text
[1] Speak a query (microphone)
[2] Type a query  (text)
[q] Quit
```

Typed mode skips M1 Speech and shows module-by-module output for:

- M1 Speech
- M2 NLU
- M3 QueryIR
- M4 SQL
- M5 Execution / Verification

## Dummy DB SQL Verification

Rebuild the local SQLite fixture and run one query:

```powershell
python temp_recipe_db\run_query.py --rebuild "vegan recipes with spinach under 300 calories" --brief --explain
```

Run without rebuilding:

```powershell
python temp_recipe_db\run_query.py "vigan recipes with spinach under 300 calories" --brief --explain
```

Show fixture schema:

```powershell
python temp_recipe_db\run_query.py --schema
```

Run raw SQL against the fixture:

```powershell
python temp_recipe_db\run_query.py --sql "SELECT recipe_id, recipe_title FROM micronutrients WHERE calories < ?" --params "[400]"
```

## Tests

```powershell
python -m pytest -q
```

Focused checks:

```powershell
python tests\unit\test_m2_nlu_basic.py
python -m pytest tests\unit\test_query_pipeline.py -q
```

Compile check:

```powershell
python -m compileall -q apps src scripts temp_recipe_db tests
```

Schema audit:

```powershell
$env:PYTHONPATH="src"
python -m speech_to_sql.m4_sql.schema_audit --skip-pdf
```

## API And Web Entrypoints

```powershell
uvicorn apps.api.nlu_api:app --reload --port 8000
python apps\api\search_app.py
npm --prefix apps\proxy install
npm --prefix apps\proxy start
npm --prefix apps\web install
npm --prefix apps\web start
```

The React app is a lightweight query console for the Python `/query` API. The
CLI and dummy DB runner remain the most complete local demo paths.
