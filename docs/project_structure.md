# Project Structure

The project is organized around a five-stage Speech-to-SQL pipeline.

```text
apps/
  api/              FastAPI/Flask entry points
  cli/              interactive local CLI
  proxy/            Node proxy service
  web/              React frontend

src/speech_to_sql/
  m1_speech/        microphone audio transcription
  m2_nlu/           entity tagging, SQL operators, intent extraction
  m3_query_ir/      schema grounding and QueryIR construction
  m4_sql/           SQL validation, planning, and rendering
  m5_execution/     RecipeDB API planning, execution helpers, verification
  common/           shared config loading

config/
  recipe_db.json    RecipeDB API and alias configuration
  schema.sql        readable schema contract
  sql_schema.json   executable schema contract used by SQL generation

data/
  generated_cases/  generated NL-to-SQL cases
  query_cache/      optional local discovery/cache files

scripts/
  run_cli.py        convenient wrapper for the full CLI
  smoke_query.py    quick text-query smoke runner

temp_recipe_db/
  seed_temp_db.py   builds the local SQLite fixture
  run_query.py      compiles and verifies generated SQL against the fixture
  recipe_test.db    local generated SQLite database

tests/
  unit/             unit and pipeline contract tests
  integration/      broader NLU checks
```

The main user-facing local entry point is:

```powershell
python scripts\run_cli.py
```

Use `PYTHONPATH=src` when running package modules directly:

```powershell
$env:PYTHONPATH="src"
python -m speech_to_sql.m4_sql.schema_audit --skip-pdf
```
