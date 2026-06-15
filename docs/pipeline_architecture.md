# Pipeline Architecture

The system converts a spoken or typed recipe search request into structured
intent, grounded QueryIR, parameterized SQL, and verification output.

## M1 Speech

Location: `src/speech_to_sql/m1_speech/`

M1 records microphone audio in the CLI and transcribes it with Whisper. Typed
queries skip this stage and pass the text directly to M2.

Important files:

- `apps/cli/main.py`
- `src/speech_to_sql/m1_speech/transcriber.py`

## M2 NLU

Location: `src/speech_to_sql/m2_nlu/`

M2 extracts the structured intent:

- ingredients
- include/exclude ingredient logic
- dietary styles
- utensils
- cooking actions
- nutrition filters
- SQL operator hints

Important files:

- `entity_tagger.py`
- `extractor.py`
- `query_decomposer.py`
- `ingredient_reranker.py`
- `sql_ops/extractor.py`
- `data/RecipeDB_Ingredient_Category.csv`
- `data/actions.csv`
- `data/utensils.csv`
- `data/sql_operator_lexicon.json`

Ingredient extraction uses exact matching first, then fuzzy/phonetic rescue,
then optional reranking with the local ingredient reranker model. Dietary
aliases and common ASR/typing slips are handled before ingredient extraction so
terms like `vegan` do not become ingredient false positives.

## M3 QueryIR

Location: `src/speech_to_sql/m3_query_ir/`

M3 prepares, enriches, links, and grounds the M2 intent against the schema
contract. It builds a `QueryIR` object instead of letting any model write SQL.

Important files:

- `intent_preparer.py`
- `intent_enricher.py`
- `intent_grounder.py`
- `schema_linker.py`
- `schema_registry.py`
- `sql_intent_builder.py`
- `query_ir_contract.py`
- `query_relaxer.py`

## M4 SQL

Location: `src/speech_to_sql/m4_sql/`

M4 validates QueryIR and renders parameterized SQL. It uses the schema contract
in `config/sql_schema.json` for valid fields, operators, joins, limits,
semantic guards, and allowed dietary values.

Important files:

- `sql_query_builder.py`
- `sql_validator.py`
- `sql_plan_builder.py`
- `sql_renderer.py`
- `schema_audit.py`

The SQL renderer must receive validated QueryIR. It should not receive raw SQL
from an LLM or any other model.

## M5 Execution And Verification

Location: `src/speech_to_sql/m5_execution/` and `temp_recipe_db/`

M5 plans RecipeDB API calls and verifies generated SQL against the local dummy
SQLite fixture. The CLI does not call the external RecipeDB API by default; it
shows planned API calls and executes the generated SQL locally.

Important files:

- `src/speech_to_sql/m5_execution/query_planner.py`
- `src/speech_to_sql/m5_execution/api_verifier.py`
- `src/speech_to_sql/m5_execution/search.py`
- `temp_recipe_db/run_query.py`
- `temp_recipe_db/seed_temp_db.py`

If SQL is valid but returns zero rows because the request is semantically
contradictory, for example `vegan` with `chicken`, the verifier reports the
semantic conflict instead of treating that as a broken SQL query.
