# Datasets

Put external datasets here. Keep raw downloads separate from model-ready JSONL.

## Folder Layout

```text
pipeline/module1_asr/datasets/
  raw/
    spider/
    wikisql/
    manual/
  processed/
    sql_operator_train.jsonl
```

## Manual Labeled Format

Preferred format for highest quality:

```json
{"text":"show employees where salary is greater than 50000","spans":[{"start":15,"end":20,"type":"CLAUSE"},{"start":31,"end":43,"type":"COMPARE"}]}
```

`end` is exclusive.

## Convert Raw JSON/JSONL

The converter looks for text fields named `text`, `query`, `question`, `utterance`, or `sentence`.

```powershell
cd C:\Users\wadhw\IP_Project\pipeline
python -m module1_asr.sql_ops.prepare_dataset --input module1_asr\datasets\raw\manual --output module1_asr\datasets\processed\sql_operator_train.jsonl
```

## Train

```powershell
cd C:\Users\wadhw\IP_Project\pipeline
python -m module1_asr.sql_ops.train --dataset-jsonl module1_asr\datasets\processed\sql_operator_train.jsonl
```

