# IP Project

Speech-to-SQL project for gastronomic databases. The repository is organized so
each stage has a clear place:

```text
apps/
  api/              FastAPI/Flask entry points
  cli/              interactive local CLI
  proxy/            Node proxy service
  web/              React frontend
src/speech_to_sql/
  m1_speech/        speech/audio transcription
  m2_nlu/           entity tagging, SQL operators, intent extraction
  m3_query_ir/      schema grounding and QueryIR construction
  m4_sql/           SQL validation, planning, and rendering
  m5_execution/     RecipeDB API execution, merging, filtering, ranking
  common/           shared configuration
config/             schema and RecipeDB configuration
data/               generated cases and local query cache
tests/              unit and integration checks
scripts/            utility and smoke-test scripts
temp_recipe_db/     local SQLite fixture for SQL testing
docs/               current architecture and run/verification notes
```

From the repository root, use `PYTHONPATH=src` when running package modules
directly:

```powershell
$env:PYTHONPATH="src"
python scripts\smoke_query.py "vegan spinach under 300 calories" --verbose
python temp_recipe_db\run_query.py --rebuild "vegan recipes with spinach under 300 calories" --brief --explain
python -m pytest
```

Useful local entry points:

```powershell
python apps\cli\main.py        # full M1-M5 interactive CLI dashboard
python scripts\run_cli.py      # same CLI, convenient wrapper
uvicorn apps.api.nlu_api:app --reload --port 8000
python apps\api\search_app.py
npm --prefix apps\proxy install
npm --prefix apps\proxy start
npm --prefix apps\web install
npm --prefix apps\web start
```

Current project documentation is in `docs/README.md`.
