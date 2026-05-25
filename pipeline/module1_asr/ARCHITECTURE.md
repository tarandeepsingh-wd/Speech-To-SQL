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
9. Ingredient boolean logic: include, exclude, AND/OR connectors
10. SQL filter-plan generation

Important implementation file:

- `extractor.py`

The current extractor keeps the architecture you wanted: exact layer first, fuzzy/phonetic rescue after, overlap resolution at the end. We do not pre-warm phonemes for the full DB at import time because it makes demos and API startup hang; phoneme keys are cached lazily.

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

Important files:

- `sql_ops/extractor.py`
- `sql_ops/ner.py`
- `sql_ops/train.py`
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
- optimizer: AdamW
- default learning rate: `5e-5`
- default batch size: `16`
- default max length: `128`
- default epochs: `3`

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

The training script automatically includes this file if it exists. This lets the model improve from failed/ambiguous cases.

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
