# Module 3 SQL Architecture

Module 3 uses a schema-grounded SQL pipeline. The important rule is that no
model or retriever is allowed to write raw SQL.

## Current Implemented Path

1. Module 1 returns structured tags: ingredients, nutrition filters, diet tags,
   time filters, and SQL operators such as limit/order intent.
2. `intent_preparer.py` enriches and canonicalizes the tags using config-backed
   aliases.
3. `intent_grounder.py` proves the prepared intent against the schema contract.
   It records grounding evidence and rejects unresolved fields, unsupported
   aggregates, invalid category filters, and unsafe unsupported relation paths.
4. `schema_registry.py` loads `config/sql_schema.json`, which is the compact
   knowledge graph for valid tables, columns, indexes, and join assumptions.
5. `sql_intent_builder.py` converts grounded intent into `QueryIR`.
6. `sql_validator.py` checks that every source, field, operator, quantifier,
   order key, and limit exists in the schema contract.
7. `sql_plan_builder.py` dynamically builds the SQL plan from the validated IR.
   It chooses base-table predicates, joins, grouped ingredient match sets, or
   correlated `EXISTS` clauses based on the schema relationships.
8. `sql_renderer.py` renders the final parameterized SQL and params.

For SQL generation, this is the production path. The REST/API path remains for
endpoint-backed search and API-response verification; it is not the source of
truth for SQL rendering.

## Optional LLM/RAG Path

If an LLM is added later, it should sit before `QueryIR` only:

```text
natural language + Module 1 tags
  -> schema/RAG context from schema.sql + sql_schema.json
  -> constrained LLM JSON
  -> query_ir_contract.py
  -> intent_grounder.py
  -> sql_validator.py
  -> sql_plan_builder.py
  -> sql_renderer.py
```

The LLM must output constrained `QueryIR` JSON, not SQL. `query_ir_contract.py`
rejects raw SQL keys, unknown JSON fields, unsafe select expressions, invalid
limits, nested values, and anything the schema validator cannot prove from
`sql_schema.json`.

No external model API is required for the current implementation. If a local
model is added later, prefer a small instruction model with strong structured
JSON behavior. The best current fit for this project is `Qwen2.5-7B-Instruct`;
`Llama-3.1-8B-Instruct`, `Mistral-7B-Instruct-v0.3`, and `Phi-3-mini-4k` are
reasonable fallback candidates. The model boundary stays unchanged: model output
enters `query_ir_contract.py`, never `sql_renderer.py`.

## Schema Sources

- `config/schema.sql` is the human-readable logical SQL contract built from the
  PDF and MongoDB index screenshots.
- `config/sql_schema.json` is the executable schema contract used by code.
- MongoDB screenshots did not show declared foreign keys, so joins are stored as
  index-supported or inferred assumptions and exposed in compiled SQL metadata.

## Guardrails

`config/sql_schema.json` now stores more than table names. It also declares
predicate-count limits, value ranges for numeric fields, preferred vs fallback
join paths, allowed quantifiers, time/nutrition units, and relaxation priority.
`sql_validator.py` enforces those constraints before SQL rendering. This blocks
model-style hallucinations such as negative calories, 50 generated predicates,
or unknown dietary values.

Dietary OR is represented explicitly in `QueryIR` with `or_exists`, so
``vegan or vegetarian`` becomes an OR over valid dietary flag columns instead
of accidentally requiring both flags.

## Efficiency Choice

The deterministic path is the default because it is faster and less
hallucination-prone. RAG or LLM should be used only when Module 1 cannot produce
enough structured intent; even then, the generated output is boxed into
`QueryIR` and revalidated before SQL is rendered.

Ingredient matching is controlled by `config/sql_schema.json`. The current
schema uses `RecipeDB_ingredient_phrase.matching_values` as the primary
canonical lookup and `ingredient_lower LIKE ?` as a phrase fallback. Multiple
required ingredients are converted into one grouped match set before joining
back to the recipe root, avoiding one recipe join per ingredient.

If a future SQL executor returns zero rows, `query_relaxer.py` can produce
schema-configured fallback `QueryIR` variants. The current policy first drops
excluded ingredients, then relaxes required ingredients from all-of to any-of,
then drops cuisine/category filters, then drops nutrition filters.

Accuracy cannot be promised as 100% without verified data access, but the code
now follows a fail-closed rule: if a requested field, aggregate, category, or
join path cannot be proven from config, SQL generation stops.
