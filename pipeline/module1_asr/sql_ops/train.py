import json
import random
import re
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


BASE_DIR = Path(__file__).parents[1]
LEXICON_PATH = BASE_DIR / "data" / "sql_operator_lexicon.json"
CONFIG_PATH = BASE_DIR / "data" / "sql_ner_training_config.json"
MODEL_DIR = BASE_DIR / "models" / "sql_operator_ner"
SQL_FEEDBACK_PATH = BASE_DIR / "data" / "feedback" / "sql_operator_feedback.jsonl"


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _operator_terms_by_type(lexicon: dict) -> dict[str, list[str]]:
    grouped = {}
    for term, payload in lexicon["terms"].items():
        grouped.setdefault(payload["type"], []).append(term)
    return grouped


def _render_examples(lexicon: dict, config: dict, limit: int | None = None) -> list[dict]:
    training_config = config.get("training", {})
    rng = random.Random(training_config.get("random_seed", 42))
    grouped_terms = _operator_terms_by_type(lexicon)
    examples = []

    compare_terms = grouped_terms.get("COMPARE", [])
    agg_terms = grouped_terms.get("AGG", [])
    fields = config["fields"]
    tables = config["tables"]
    values = config["values"]

    for template in config["templates"]:
        for operator in compare_terms:
            for _ in range(12):
                field = rng.choice(fields)
                field2 = rng.choice([item for item in fields if item != field])
                text = template.format(
                    table=rng.choice(tables),
                    field=field,
                    field2=field2,
                    operator=operator,
                    agg=rng.choice(agg_terms or ["count"]),
                    value=rng.choice(values),
                    value2=rng.choice(values),
                )
                examples.append(_label_text(text, lexicon))

    rng.shuffle(examples)
    return examples[:limit] if limit else examples


def _load_jsonl_examples(path: Path, lexicon: dict) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text") or row.get("query")
            if not text:
                raise ValueError(f"{path}:{line_no} missing text/query")
            spans = row.get("spans")
            if spans:
                examples.append({"text": text, "spans": spans})
            else:
                examples.append(_label_text(text, lexicon))
    return examples


def _find_operator_spans(text: str, lexicon: dict) -> list[tuple[int, int, str]]:
    spans = []
    lowered = text.lower()
    terms = sorted(lexicon["terms"].items(), key=lambda item: len(item[0]), reverse=True)
    for term, payload in terms:
        for match in re.finditer(r"\b" + re.escape(term) + r"\b", lowered):
            spans.append((match.start(), match.end(), payload["type"]))
    for symbol, payload in lexicon["symbols"].items():
        for match in re.finditer(re.escape(symbol), text):
            spans.append((match.start(), match.end(), payload["type"]))
    return _resolve_char_overlaps(spans)


def _resolve_char_overlaps(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    ranked = sorted(spans, key=lambda item: (item[0], -(item[1] - item[0])))
    accepted = []
    used = set()
    for start, end, label_type in ranked:
        positions = set(range(start, end))
        if positions & used:
            continue
        accepted.append((start, end, label_type))
        used |= positions
    return sorted(accepted)


def _label_text(text: str, lexicon: dict) -> dict:
    return {
        "text": text,
        "spans": [
            {"start": start, "end": end, "type": label_type}
            for start, end, label_type in _find_operator_spans(text, lexicon)
        ],
    }


def _label_for_offset(offset: tuple[int, int], spans: list[dict]) -> str:
    start, end = offset
    if start == end:
        return "O"
    for span in spans:
        if start >= span["start"] and end <= span["end"]:
            prefix = "B" if start == span["start"] else "I"
            return f"{prefix}-{span['type']}"
    return "O"


class SqlOperatorDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, label2id: dict[str, int]):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        encoded = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
            return_offsets_mapping=True,
        )
        labels = []
        for offset in encoded.pop("offset_mapping"):
            if offset[0] == offset[1]:
                labels.append(-100)
            else:
                labels.append(self.label2id[_label_for_offset(offset, example["spans"])])
        encoded["labels"] = labels
        return {key: torch.tensor(value) for key, value in encoded.items()}


def train(
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    limit: int | None = None,
    dataset_jsonl: str | None = None,
    export_synthetic: str | None = None,
) -> None:
    lexicon = _load_json(LEXICON_PATH)
    config = _load_json(CONFIG_PATH)
    training_config = config.get("training", {})
    labels = config["labels"]
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    synthetic_limit = limit if limit is not None else training_config.get("synthetic_limit")
    examples = _render_examples(lexicon, config, limit=synthetic_limit)

    if export_synthetic:
        export_path = Path(export_synthetic)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        print(f"exported synthetic training data to {export_path}")
        return

    if dataset_jsonl:
        examples.extend(_load_jsonl_examples(Path(dataset_jsonl), lexicon))

    if SQL_FEEDBACK_PATH.exists():
        examples.extend(_load_jsonl_examples(SQL_FEEDBACK_PATH, lexicon))

    rng = random.Random(training_config.get("random_seed", 42))
    rng.shuffle(examples)
    split = int(len(examples) * training_config.get("train_split", 0.85))
    train_examples = examples[:split]
    val_examples = examples[split:]

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForTokenClassification.from_pretrained(
        config["model_name"],
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    train_data = SqlOperatorDataset(train_examples, tokenizer, label2id)
    val_data = SqlOperatorDataset(val_examples, tokenizer, label2id)
    batch_size = batch_size or training_config.get("batch_size", 16)
    learning_rate = learning_rate or training_config.get("learning_rate", 5e-5)
    epochs = epochs or training_config.get("epochs", 3)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(output.loss.detach())

        metrics = _entity_f1(model, val_loader, device, id2label)
        print(
            f"epoch={epoch} loss={total_loss / max(len(train_loader), 1):.4f} "
            f"entity_f1={metrics['f1']:.3f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(MODEL_DIR)
            model.save_pretrained(MODEL_DIR)
            print(f"  new best checkpoint saved to {MODEL_DIR} (f1={best_f1:.3f})")

        if epoch - best_epoch >= 2:
            print(f"  early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    print(f"training complete. best entity_f1={best_f1:.3f} at epoch={best_epoch}")


def _entity_f1(model, loader, device, id2label: dict[int, str]) -> dict:
    """
    Compute entity-level F1 for BIO tags. Falls back to token accuracy if
    seqeval is not installed.
    """
    model.eval()
    true_spans_all = []
    pred_spans_all = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }
            predictions = model(**inputs).logits.argmax(dim=-1)

            for i in range(labels.shape[0]):
                true_labels = [
                    id2label[int(label)]
                    for label in labels[i]
                    if int(label) != -100
                ]
                pred_labels = [
                    id2label[int(pred)]
                    for pred, label in zip(predictions[i], labels[i])
                    if int(label) != -100
                ]
                true_spans_all.append(true_labels)
                pred_spans_all.append(pred_labels)

    if not true_spans_all:
        return {"f1": 0.0, "report": "No validation examples."}

    try:
        from seqeval.metrics import classification_report, f1_score

        return {
            "f1": float(f1_score(true_spans_all, pred_spans_all)),
            "report": classification_report(true_spans_all, pred_spans_all),
        }
    except ImportError:
        correct = sum(
            pred == true
            for true_seq, pred_seq in zip(true_spans_all, pred_spans_all)
            for true, pred in zip(true_seq, pred_seq)
        )
        total = sum(len(seq) for seq in true_spans_all)
        return {
            "f1": correct / total if total else 0.0,
            "report": "Install seqeval for entity-level F1.",
        }


def _parse_args():
    parser = argparse.ArgumentParser(description="Train SQL operator BIO NER model.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset-jsonl", default=None)
    parser.add_argument("--export-synthetic", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        limit=args.limit,
        dataset_jsonl=args.dataset_jsonl,
        export_synthetic=args.export_synthetic,
    )
