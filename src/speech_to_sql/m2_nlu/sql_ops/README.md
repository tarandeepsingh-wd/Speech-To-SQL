# SQL Operator Package

This package contains the complete SQL keyword/operator extraction pipeline.

## Files

- `extractor.py`: rule extraction, ML NER fusion, normalization, validation.
- `ner.py`: DistilBERT token-classification inference.
- `train.py`: configurable training pipeline.
- `train_clause_classifier.py`: multi-label classifier for implicit SQL labels.
- `prepare_dataset.py`: converts raw/manual JSON, JSONL, or CSV datasets to training JSONL.
- `download_datasets.py`: optional HuggingFace dataset downloader.

## Model

Two model heads are supported:

- SQL operator NER: `AutoModelForTokenClassification`, for visible spans such as `where`, `less than`, `count`.
- SQL clause classifier: `AutoModelForSequenceClassification`, for implicit labels such as `SELECT`, `WHERE`, `=`, `COUNT`.

## Dataset Location

Put datasets here:

```text
src/speech_to_sql/m2_nlu/datasets/raw/
```

Converted training file:

```text
src/speech_to_sql/m2_nlu/datasets/processed/sql_operator_train.jsonl
```

## Commands

Download public datasets:

```powershell
$env:PYTHONPATH="src"
python -m speech_to_sql.m2_nlu.sql_ops.download_datasets --dataset spider
python -m speech_to_sql.m2_nlu.sql_ops.download_datasets --dataset wikisql
```

Convert your dataset:

```powershell
$env:PYTHONPATH="src"
python -m speech_to_sql.m2_nlu.sql_ops.prepare_dataset --input src\speech_to_sql\m2_nlu\datasets\raw\manual --output src\speech_to_sql\m2_nlu\datasets\processed\sql_operator_train.jsonl
```

Train:

```powershell
$env:PYTHONPATH="src"
python -m speech_to_sql.m2_nlu.sql_ops.train --dataset-jsonl src\speech_to_sql\m2_nlu\datasets\processed\sql_operator_train.jsonl
```

Train the WikiSQL-style clause/operator classifier from CSV:

```powershell
$env:PYTHONPATH="src"
python -m speech_to_sql.m2_nlu.sql_ops.train_clause_classifier --dataset-csv src\speech_to_sql\m2_nlu\datasets\raw\wikisql_labeled.csv
```

Recommended final food/nutrition training dataset:

```powershell
$env:PYTHONPATH="src"
python -m speech_to_sql.m2_nlu.sql_ops.train_clause_classifier --dataset-csv src\speech_to_sql\m2_nlu\datasets\raw\wikisql_food_nutrition_v4.csv --epochs 2 --batch-size 8 --max-length 64 --progress-every 25
```

Evaluate the saved clause classifier on a labeled CSV:

```powershell
$env:PYTHONPATH="src"
python -m speech_to_sql.m2_nlu.sql_ops.evaluate_clause_classifier --dataset-csv src\speech_to_sql\m2_nlu\datasets\raw\wikisql_food_nutrition_v4.csv --show-errors 10
```

Use classifier predictions directly:

```python
from speech_to_sql.m2_nlu.sql_ops import predict_sql_clause_labels

predict_sql_clause_labels("how many recipes have protein above 30")
```

Fuse classifier output into `extract_sql_operators`:

```powershell
$env:SPEECH_TO_SQL_ENABLE_SQL_CLAUSE_ML="1"
```

Classifier predictions use `layer: "ml_clause"` and negative spans because
implicit labels such as `SELECT` and `WHERE` are whole-query outputs, not visible
text spans. In the full `/tag` pipeline, `ml_clause` predictions do not suppress
ingredient extraction.

The local `main.py` demo enables the clause classifier by default and can save
manual corrections to:

```text
src/speech_to_sql/m2_nlu/data/feedback/sql_clause_feedback.jsonl
```

The clause-classifier training command automatically includes that feedback file
on future runs.
