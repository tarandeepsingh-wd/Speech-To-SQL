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
   It chooses base-table predicates or correlated `EXISTS` clauses based on the
   schema relationships.
8. `sql_renderer.py` renders the final parameterized SQL and params.

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

## Schema Sources

- `config/schema.sql` is the human-readable logical SQL contract built from the
  PDF and MongoDB index screenshots.
- `config/sql_schema.json` is the executable schema contract used by code.
- MongoDB screenshots did not show declared foreign keys, so joins are stored as
  index-supported or inferred assumptions and exposed in compiled SQL metadata.

## Efficiency Choice

The deterministic path is the default because it is faster and less
hallucination-prone. RAG or LLM should be used only when Module 1 cannot produce
enough structured intent; even then, the generated output is boxed into
`QueryIR` and revalidated before SQL is rendered.

Accuracy cannot be promised as 100% without verified data access, but the code
now follows a fail-closed rule: if a requested field, aggregate, category, or
join path cannot be proven from config, SQL generation stops.
