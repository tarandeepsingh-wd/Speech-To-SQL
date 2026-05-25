# SQL Operator Package

This package contains the complete SQL keyword/operator extraction pipeline.

## Files

- `extractor.py`: rule extraction, ML NER fusion, normalization, validation.
- `ner.py`: DistilBERT token-classification inference.
- `train.py`: configurable training pipeline.
- `prepare_dataset.py`: converts raw/manual datasets to training JSONL.
- `download_datasets.py`: optional HuggingFace dataset downloader.

## Model

`distilbert-base-uncased` with `AutoModelForTokenClassification`.

## Dataset Location

Put datasets here:

```text
module1_asr/datasets/raw/
```

Converted training file:

```text
module1_asr/datasets/processed/sql_operator_train.jsonl
```

## Commands

Download public datasets:

```powershell
python -m module1_asr.sql_ops.download_datasets --dataset spider
python -m module1_asr.sql_ops.download_datasets --dataset wikisql
```

Convert your dataset:

```powershell
python -m module1_asr.sql_ops.prepare_dataset --input module1_asr\datasets\raw\manual --output module1_asr\datasets\processed\sql_operator_train.jsonl
```

Train:

```powershell
python -m module1_asr.sql_ops.train --dataset-jsonl module1_asr\datasets\processed\sql_operator_train.jsonl
```

