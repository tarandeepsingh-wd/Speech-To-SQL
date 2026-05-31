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

## CSV Format

CSV files are also accepted. Use one of these text column names:

```csv
text
show recipes where calories are less than 500
find meals with protein at least 30
```

Optional manual labels can be supplied in a `spans` column as JSON:

```csv
text,spans
"show recipes where calories are less than 500","[{""start"":13,""end"":18,""type"":""CLAUSE""},{""start"":32,""end"":41,""type"":""COMPARE""}]"
```

If `spans` is missing, the converter labels operator spans from `data/sql_operator_lexicon.json`.

WikiSQL-style CSV files with `question`, `labels`, and `label_string` columns are used by
the clause classifier, not by the token-span NER trainer. Those labels describe the
whole query, so they can teach implicit output like `SELECT` and `WHERE`.

For that classifier:

- Input text comes from `question`.
- Target labels come from `labels`, falling back to `label_string`.
- The model predicts a set of SQL labels for the whole sentence; it does not
  produce character spans for implicit labels.

## Convert Raw JSON/JSONL/CSV

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

## Train Clause Classifier

```powershell
cd C:\Users\wadhw\IP_Project\pipeline
python -m module1_asr.sql_ops.train_clause_classifier --dataset-csv module1_asr\datasets\raw\wikisql_labeled.csv
```
