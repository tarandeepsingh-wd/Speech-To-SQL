# Next Semester Extension Architecture

This document describes a 4-month extension plan for turning the current
Speech-to-SQL RecipeDB workbench into a more general, production-style
conversational food query system.

The goal is not to replace the current M1-M5 pipeline. The goal is to keep the
existing deterministic schema-aware SQL generation, then add stronger intent
understanding, ranking, multi-backend execution, and evaluation around it.

## Current Baseline

The current system already supports:

- typed and speech input
- noisy text normalization
- ingredient inclusion and exclusion
- fuzzy ingredient correction
- nutrition filters
- time filters
- dietary flags
- utensils and cooking actions
- QueryIR grounding
- parameterized SQL generation
- Athena-ready SQL rendering
- live Athena execution from the UI

Current strength:

- explicit queries work well, for example:

```text
show chicken recipes under 500 calories with more than 40 grams protein
```

Current limitation:

- the system is still mostly command/query-pattern based. It is weaker for
  vague, preference-based, or conversational requests such as:

```text
I want something light and healthy for dinner
```

## Target System

The next version should support both direct database queries and broader food
recommendation intent.

Target examples:

```text
What can I make quickly with paneer and tomatoes?
```

```text
I want a low calorie vegetarian dinner with decent protein.
```

```text
Give me something light, not too oily, and ready fast.
```

```text
Find recipes like butter chicken but healthier.
```

## Proposed Architecture

```text
User
 |
 v
M1 Speech / Typed Input
 |
 v
M2 NLU
 |-- text normalization
 |-- entity extraction
 |-- ingredient correction
 |-- dietary detection
 |-- vague preference detection
 |-- dish/title ranking hints
 |
 v
M2.5 Intent Resolver
 |-- converts vague phrases to structured constraints
 |-- asks clarification questions when confidence is low
 |-- maps "healthy", "light", "quick", etc. to policy-backed filters
 |
 v
M3 QueryIR
 |-- schema linking
 |-- schema grounding
 |-- contradiction detection
 |-- backend-independent query contract
 |
 v
M4 Query Generation
 |-- Athena SQL renderer
 |-- MongoDB query renderer
 |-- optional PostgreSQL/MySQL renderer
 |
 v
M5 Execution
 |-- Athena execution
 |-- MongoDB execution
 |-- local test DB execution
 |
 v
M6 Ranking And Explanation
 |-- title similarity
 |-- ingredient coverage score
 |-- nutrition preference score
 |-- dietary confidence score
 |-- result explanation
 |
 v
UI / CLI Output
```

## New Module: M2.5 Intent Resolver

Add a layer between M2 NLU and M3 QueryIR.

Purpose:

- convert vague food phrases into structured filters
- separate hard constraints from soft preferences
- decide when to ask clarification questions
- avoid forcing every query into strict SQL immediately

Example:

```text
food with low calories
```

Resolved intent:

```json
{
  "hard_filters": [
    {"field": "calories", "operator": "<=", "value": 500}
  ],
  "soft_preferences": [],
  "clarification_needed": false
}
```

Example:

```text
healthy dinner
```

Resolved intent:

```json
{
  "hard_filters": [],
  "soft_preferences": [
    {"field": "calories", "direction": "lower"},
    {"field": "protein", "direction": "higher"},
    {"field": "fat", "direction": "lower"}
  ],
  "clarification_needed": true,
  "clarification_question": "Do you mean low calorie, high protein, low fat, or balanced?"
}
```

## QueryIR Extension

The current QueryIR is good for hard filters. Next semester, extend it to
distinguish:

- hard filters
- soft preferences
- ranking features
- contradiction warnings
- clarification questions
- backend target

Example future QueryIR:

```json
{
  "filters": [
    {"source": "ingredients", "field": "ingredient", "op": "=", "value": "paneer"},
    {"source": "nutrition", "field": "calories", "op": "<=", "value": 500}
  ],
  "preferences": [
    {"field": "protein", "direction": "DESC", "weight": 0.3},
    {"field": "total_time", "direction": "ASC", "weight": 0.2}
  ],
  "ranking": [
    {"type": "title_similarity", "phrase": "paneer butter masala", "weight": 0.5}
  ],
  "warnings": [],
  "backend": "athena"
}
```

## Multi-Backend Support

The project can become stronger by generating queries for multiple database
systems from the same QueryIR.

Target backends:

- Athena SQL
- MongoDB aggregation pipeline
- local SQLite for test fixtures
- optional PostgreSQL or MySQL SQL

Why this matters:

- professor data may live in MongoDB
- Athena is used for real large-scale testing
- SQLite remains useful for deterministic unit tests
- backend-independent QueryIR makes the project research-grade

Example:

```text
chicken recipes under 500 calories
```

One intent should generate:

- Athena SQL
- MongoDB query
- local SQLite SQL for testing

## Ranking System

Filtering alone is not enough. Many valid recipes can match a query, but the
best result should appear first.

Ranking signals:

- exact ingredient coverage
- excluded ingredient safety
- recipe title similarity
- nutrition preference score
- dietary match confidence
- prep/cook/total time score
- user-requested sort

Important rule:

- explicit user sort wins over ranking boost.

Example:

```text
highest protein butter chicken
```

The system should preserve protein sorting. It should not force title match to
the top if the user explicitly asked for highest protein.

## Data Needed

### Query Evaluation Dataset

Create a labeled dataset of 500-1000 natural language queries.

Columns:

- query text
- expected ingredients
- expected excluded ingredients
- expected nutrition filters
- expected time filters
- expected dietary flags
- expected sort
- expected limit
- expected backend query type
- expected clarification question, if any
- difficulty/category

Query categories:

- single ingredient
- multiple ingredients
- exclusions
- nutrition
- time
- dietary
- utensils
- cooking actions
- mixed constraints
- vague/general queries
- typo-heavy queries
- contradiction queries
- ranking-only queries

### Food Phrase Policy Dataset

Needed for vague terms.

Examples:

| Phrase | Interpretation |
| --- | --- |
| low calorie | calories <= 500 |
| sugar free | sugar <= 0.5 |
| quick | total_time <= 30 |
| less time to prepare | prep_time <= 20 |
| high protein | protein >= 30 |
| light meal | soft preference for lower calories and fat |
| healthy | clarification or soft multi-objective ranking |

This should be configurable, not hardcoded in random Python branches.

### Backend Schema Catalog

Maintain a schema catalog for each backend:

- table/collection names
- column/field names
- types
- join keys
- nested fields
- array fields
- numeric casting rules
- null handling rules
- dialect-specific syntax

Files can live under:

```text
config/backends/
  athena_schema.json
  mongodb_schema.json
  sqlite_schema.json
```

## Model Work

Do not train a model to directly write SQL. Keep SQL deterministic and
schema-grounded.

Good model targets:

- ingredient candidate reranking
- vague phrase classification
- query intent classification
- clarification-needed classification
- title/semantic result reranking

Avoid:

- free-form LLM SQL generation without schema validation
- query-specific hardcoding
- model decisions that bypass QueryIR

Recommended model types:

- logistic regression or gradient boosting for small interpretable classifiers
- sentence embeddings for semantic similarity and ranking
- DistilBERT only when enough labeled data exists
- rules first, ML as fallback or reranker

## UI Requirements

Add explainability so the professor can see why the query was generated.

Useful panels:

- recognized ingredients
- excluded ingredients
- nutrition filters
- time filters
- dietary filters
- ranking signals
- backend selected
- generated QueryIR
- generated Athena SQL
- generated MongoDB query
- result explanation

Example result explanation:

```text
Shown because it contains chicken and garlic, has 42g protein,
is under 500 calories, and its title closely matches "butter chicken".
```

## Evaluation Metrics

Track:

- intent extraction accuracy
- ingredient precision and recall
- exclusion precision and recall
- nutrition filter accuracy
- time filter accuracy
- SQL validity rate
- Athena execution success rate
- MongoDB execution success rate
- result relevance score
- average latency
- clarification correctness

For demos, report:

```text
Intent accuracy: 92%
SQL validity: 98%
Athena execution success: 96%
Average latency: 3.2s
```

## 4-Month Roadmap

### Month 1: Dataset And Intent Taxonomy

Deliverables:

- 500+ labeled natural language queries
- final intent taxonomy
- vague phrase policy file
- contradiction policy
- baseline accuracy report for current system

Main work:

- define query categories
- create annotation format
- generate and manually verify test cases
- identify current failure patterns

### Month 2: M2.5 Intent Resolver And QueryIR Upgrade

Deliverables:

- `M2.5 Intent Resolver`
- hard vs soft constraint separation
- clarification question support
- upgraded QueryIR contract
- tests for vague/general queries

Main work:

- add policy-backed phrase mappings
- add soft preference ranking metadata
- preserve existing explicit query accuracy
- add contradiction handling

### Month 3: Multi-Backend Query Generation

Deliverables:

- Athena renderer improvements
- MongoDB query renderer
- backend schema registry
- backend-specific validation tests

Main work:

- separate logical QueryIR from backend query syntax
- generate MongoDB aggregation pipeline
- add backend selection setting
- run real Athena and MongoDB tests

### Month 4: Ranking, UI, And Evaluation

Deliverables:

- ranking engine
- explanation panel
- evaluation dashboard or Excel report
- final presentation demo script
- final project report

Main work:

- add result scoring
- add title/semantic ranking
- add explanation text
- run benchmark suite
- prepare professor demo cases

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Vague terms are subjective | Use configurable policy + clarification |
| Model reduces current accuracy | Keep deterministic rules first; ML only reranks |
| Athena syntax breaks | Add renderer tests and live smoke tests |
| MongoDB schema differs | Build backend schema catalog |
| Too many features in 4 months | Prioritize dataset, QueryIR, backend, evaluation |
| Slow live execution | Cache schema and limit result counts |

## What We Would Need

Access:

- Athena database name
- Athena workgroup
- Athena S3 output location
- MongoDB connection URI
- MongoDB database and collection names
- sample documents from MongoDB
- permission to run limited test queries

Data:

- recipe schema for Athena
- recipe schema for MongoDB
- sample result exports
- ingredient vocabulary
- nutrition field map
- dietary field map
- labeled query evaluation dataset

Engineering:

- backend schema registry
- QueryIR contract upgrade
- Athena renderer tests
- MongoDB renderer tests
- UI explainability panel
- ranking engine
- benchmark runner

Presentation:

- 20 simple demo queries
- 20 compound demo queries
- 20 vague/general demo queries
- 20 backend stress-test queries
- accuracy and execution report

## Recommended Final Scope

For a strong next-semester project, focus on:

1. Generalized intent understanding
2. QueryIR as a backend-independent contract
3. Athena + MongoDB generation
4. Ranking and explanation
5. Evaluation on a labeled benchmark

This gives a clear research and engineering contribution without weakening the
current deterministic SQL system.
