# ML Models

The project uses deterministic rules plus local ML models. No external model
API is required for the current CLI or dummy DB flow.

## M1 Speech: Whisper

File: `src/speech_to_sql/m1_speech/transcriber.py`

Microphone mode uses Whisper through `whisper.load_model()`. The default model
size is `small`. Typed mode does not load or test Whisper.

## M2 Ingredient Reranker

Files:

- `src/speech_to_sql/m2_nlu/ingredient_reranker.py`
- `src/speech_to_sql/m2_nlu/models/ingredient_reranker/model.pkl`

The extractor first generates ingredient candidates using exact, fuzzy, and
phonetic matching. The reranker then scores candidates with a saved
`DictVectorizer + LogisticRegression` payload. If the model file is missing or
invalid, the runtime deliberately falls back to the original extractor output.

Train or retrain:

```powershell
$env:PYTHONPATH="src"
python -m speech_to_sql.m2_nlu.train_ingredient_reranker --source all
```

With feedback:

```powershell
$env:PYTHONPATH="src"
python -m speech_to_sql.m2_nlu.train_ingredient_reranker --source all --feedback-jsonl src\speech_to_sql\m2_nlu\data\feedback\ingredient_feedback.jsonl
```

## M2 SQL Operator NER

Files:

- `src/speech_to_sql/m2_nlu/sql_ops/ner.py`
- `src/speech_to_sql/m2_nlu/models/sql_operator_ner/model.safetensors`

This is a DistilBERT token-classification model for explicit SQL operator
spans. It is optional at runtime. Enable it with:

```powershell
$env:SPEECH_TO_SQL_ENABLE_SQL_ML="1"
```

By default, SQL operator extraction uses rules and fuzzy lexicon fallbacks. The
NER model is used when enabled and when there are no rule candidates, unless
forced:

```powershell
$env:SPEECH_TO_SQL_FORCE_SQL_ML="1"
```

## M2 SQL Clause Classifier

Files:

- `src/speech_to_sql/m2_nlu/sql_ops/clause_classifier.py`
- `src/speech_to_sql/m2_nlu/models/sql_clause_classifier/model.safetensors`
- `src/speech_to_sql/m2_nlu/models/sql_clause_classifier/labels.json`

This is a DistilBERT multi-label sequence classifier for implicit SQL labels
such as `SELECT`, `WHERE`, `COUNT`, `ORDER BY`, and `LIMIT`.

Enable it in operator extraction:

```powershell
$env:SPEECH_TO_SQL_ENABLE_SQL_CLAUSE_ML="1"
```

The merged output may not always show `layer: "ml_clause"` because duplicate ML
predictions are removed when a rule or NER layer already found the same
canonical operator.

Directly inspect model scores:

```powershell
$env:PYTHONPATH="src"
python -c "from speech_to_sql.m2_nlu.sql_ops import score_sql_clause_labels; print(score_sql_clause_labels('top 5 vegan recipes highest protein'))"
```

Train:

```powershell
$env:PYTHONPATH="src"
python -m speech_to_sql.m2_nlu.sql_ops.train_clause_classifier --dataset-csv src\speech_to_sql\m2_nlu\datasets\raw\wikisql_labeled.csv
```

## M3/M4 Model Boundary

Models are allowed to help produce structured intent or QueryIR-like JSON, but
they must not write raw SQL directly. SQL is rendered only after QueryIR has
passed schema grounding and validation.
