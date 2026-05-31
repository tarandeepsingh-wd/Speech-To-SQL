# Module 1 Architecture

## Ingredient Extraction

Runtime flow:

1. ASR/text input
2. Text normalization with span preservation
3. Exact matching with Aho-Corasick over RecipeDB ingredients and configured synonyms
4. POS-filtered fuzzy matching for misspellings
5. Metaphone + g2p phoneme reranking
6. Phonetic neighbor fallback
7. Optional semantic matcher
8. Overlap resolution and deduplication
9. Logistic-regression candidate reranker to reject false positives
10. Ingredient boolean logic: include, exclude, AND/OR connectors
11. SQL filter-plan generation

Important implementation files:

- `extractor.py`
- `ingredient_reranker.py`
- `train_ingredient_reranker.py`

The current extractor keeps the architecture you wanted: exact layer first, fuzzy/phonetic rescue after, overlap resolution at the end. We do not pre-warm phonemes for the full DB at import time because it makes demos and API startup hang; phoneme keys are cached lazily.

The ingredient reranker is a candidate-validation layer, not a replacement for
the extractor. The extractor still generates high-recall candidates using
RecipeDB, synonyms, fuzzy matching, and phonetic matching. The reranker then
uses a trained `DictVectorizer + LogisticRegression` model to score whether each
candidate is valid in the current query. Its features come from extractor
metadata and generated training data: fuzzy score, metaphone overlap, extraction
layer/source, surface/canonical length, candidate/canonical tokens, local
context tokens, nearby numbers, and RecipeDB frequency. If no saved reranker
model exists, runtime falls back to the original extractor output.

The reranker is integrated in `entity_tagger.py` after `extract_ingredients()`
and before ingredient boolean decomposition. This means false-positive
ingredient candidates are removed before `ingredient_logic`, `sql_filter_plan`,
and `unified_sql_plan` are built.

Ingredient corrections can be logged from the local demo or from the API
endpoint `/feedback/ingredients`. Those rows are stored in
`data/feedback/ingredient_feedback.jsonl` and are automatically included by
`train_ingredient_reranker.py` on the next training run. The runtime loader
checks the saved model timestamp, so a newly trained `model.pkl` is picked up by
future tagging calls without code changes.

## SQL Operator Extraction

Runtime flow:

1. Input text
2. Rule extractor using Aho-Corasick and symbolic operators
3. DistilBERT token-classification model for BIO NER, if a trained checkpoint exists
4. Fallback fuzzy match against configured lexicon
5. Fusion and overlap resolution
6. Normalization to canonical SQL tokens
7. Context validation
8. API-friendly output schema

This is intentionally separate from ingredient extraction. SQL extraction does
not call the ingredient extractor, and ingredient extraction does not call the
SQL extractor. They meet only inside `entity_tagger.py`, where the orchestrator
combines independent outputs.

Implicit WikiSQL-style clause predictions are emitted as `layer: "ml_clause"`.
They can predict labels such as `SELECT`, `WHERE`, `COUNT`, and `=` even when no
literal span exists in the sentence. Those implicit predictions do not suppress
ingredient extraction in the full `/tag` pipeline; only explicit SQL syntax
signals such as visible `where`, `select`, `group by`, or SQL symbols can mark a
query as structural SQL.

Important files:

- `sql_ops/extractor.py`
- `sql_ops/ner.py`
- `sql_ops/train.py`
- `sql_ops/train_clause_classifier.py`
- `sql_ops/prepare_dataset.py`
- `sql_ops/download_datasets.py`
- `data/sql_operator_lexicon.json`
- `data/sql_ner_training_config.json`

## ML Model

Model:

- `distilbert-base-uncased`
- HuggingFace `AutoModelForTokenClassification`
- BIO labels:
  - `B-CLAUSE`, `I-CLAUSE`
  - `B-LOGIC`, `I-LOGIC`
  - `B-COMPARE`, `I-COMPARE`
  - `B-AGG`, `I-AGG`
  - `B-SET`, `I-SET`
  - `B-MODIFY`, `I-MODIFY`
  - `O`

Why this model:

- Small enough for local CPU/GPU fine-tuning.
- Strong enough for token-level phrase detection.
- Fits the professor-facing rule + ML + fusion architecture.

## Training

Default config is in `data/sql_ner_training_config.json`.

Current model choice:

- `distilbert-base-uncased`
- token classification head
- optional multi-label sequence classification head for implicit SQL labels
- `DictVectorizer + LogisticRegression` for ingredient candidate validation
- optimizer: AdamW
- default learning rate: `5e-5`
- default batch size: `16`
- default max length: `128`
- default epochs: `3`

Train the ingredient false-positive reranker:

```powershell
cd C:\Users\wadhw\IP_Project\pipeline
python -m module1_asr.train_ingredient_reranker --source all
```

The training script generates candidate-level data from the existing annotated
Module 1 suites. Each extractor candidate becomes one row; candidates whose
canonical ingredient is in the gold ingredient list are labeled `1`, and all
other candidates are labeled `0`. The saved model is loaded from:

```text
module1_asr/models/ingredient_reranker/model.pkl
```

After ingredient feedback has been collected, retrain with:

```powershell
cd C:\Users\wadhw\IP_Project\pipeline
python -m module1_asr.train_ingredient_reranker --source all --feedback-jsonl module1_asr\data\feedback\ingredient_feedback.jsonl
```

Export synthetic training data without training:

```powershell
cd C:\Users\wadhw\IP_Project\pipeline
python -m module1_asr.train_sql_operator_ner --export-synthetic module1_asr\data\sql_synthetic_train.jsonl
```

Train later with your dataset:

```powershell
cd C:\Users\wadhw\IP_Project\pipeline
python -m module1_asr.train_sql_operator_ner --dataset-jsonl path\to\your_dataset.jsonl
```

Preferred packaged command:

```powershell
cd C:\Users\wadhw\IP_Project\pipeline
python -m module1_asr.sql_ops.train --dataset-jsonl module1_asr\datasets\processed\sql_operator_train.jsonl
```

Train WikiSQL-style implicit clause/operator labels:

```powershell
cd C:\Users\wadhw\IP_Project\pipeline
python -m module1_asr.sql_ops.train_clause_classifier --dataset-csv module1_asr\datasets\raw\wikisql_labeled.csv
```

Enable those implicit classifier predictions inside `extract_sql_operators`:

```powershell
$env:TADKA_ENABLE_SQL_CLAUSE_ML="1"
```

Dataset folders:

```text
module1_asr/datasets/raw/
module1_asr/datasets/processed/
```

Convert raw/manual JSON or JSONL into training JSONL:

```powershell
cd C:\Users\wadhw\IP_Project\pipeline
python -m module1_asr.sql_ops.prepare_dataset --input module1_asr\datasets\raw\manual --output module1_asr\datasets\processed\sql_operator_train.jsonl
```

Optional public dataset download:

```powershell
cd C:\Users\wadhw\IP_Project\pipeline
python -m module1_asr.sql_ops.download_datasets --dataset spider
python -m module1_asr.sql_ops.download_datasets --dataset wikisql
```

Train with only configured synthetic data:

```powershell
cd C:\Users\wadhw\IP_Project\pipeline
python -m module1_asr.train_sql_operator_ner
```

Expected JSONL format:

```json
{"text":"show employees where salary is greater than 50000","spans":[{"start":15,"end":20,"type":"CLAUSE"},{"start":31,"end":43,"type":"COMPARE"}]}
```

`end` is exclusive.

## Feedback Loop

Feedback logging file:

- `feedback_logger.py`

Corrected SQL operator examples are stored in:

- `data/feedback/sql_operator_feedback.jsonl`
- `data/feedback/sql_clause_feedback.jsonl`

The NER and clause-classifier training scripts automatically include their
matching feedback files if they exist. This lets the models improve from
failed/ambiguous cases.

## Dataset Sources

Good SQL/operator training sources:

- Spider: cross-domain text-to-SQL data. Use natural-language questions plus SQL to derive operator spans.
- WikiSQL: larger simple text-to-SQL data. Good for WHERE, comparisons, aggregations, and SELECT-style patterns.
- Manual JSONL: best for our exact output schema because spans are explicit.

Good ingredient/recipe sources:

- Current RecipeDB ingredient/category CSV already used by the extractor.
- RecipeNLG/Kaggle can be used for more recipe ingredient phrases and user-query generation.
- RecipeDB paper/data is useful for ingredient categories and recipe-domain coverage.

Do not train directly on raw SQL strings only. Convert data into token/span labels first.

## What Not To Hardcode

Do not add random words into code files. Add domain data to:

- `data/sql_operator_lexicon.json` for SQL operators.
- `data/sql_ner_training_config.json` for synthetic training templates and hyperparameters.
- proper labeled JSONL datasets for ML.
