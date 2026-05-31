import argparse
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from module1_asr.sql_ops.label_utils import (
    load_label_list,
    normalize_sql_label,
    read_labeled_csv,
    save_label_list,
)
from module1_asr.sql_ops.train import CONFIG_PATH, _load_json


BASE_DIR = Path(__file__).parents[1]
DEFAULT_DATASET = BASE_DIR / "datasets" / "raw" / "wikisql_labeled.csv"
MODEL_DIR = BASE_DIR / "models" / "sql_clause_classifier"
LABELS_FILE = MODEL_DIR / "labels.json"
SQL_CLAUSE_FEEDBACK_PATH = BASE_DIR / "data" / "feedback" / "sql_clause_feedback.jsonl"


def _read_csv_examples(path: Path) -> list[dict]:
    if path.is_dir():
        examples = []
        for child in sorted(path.rglob("*.csv")):
            examples.extend(read_labeled_csv(child))
        return examples
    return read_labeled_csv(path)


def _load_feedback_examples(path: Path) -> list[dict]:
    if not path.exists():
        return []

    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text") or row.get("query")
            raw_labels = row.get("labels") or row.get("corrected_labels") or []
            if not text or not isinstance(raw_labels, list):
                raise ValueError(f"{path}:{line_no} missing text/query or labels")
            labels = []
            for raw_label in raw_labels:
                label = normalize_sql_label(str(raw_label))
                if label and label not in labels:
                    labels.append(label)
            if labels:
                examples.append({"text": text, "labels": labels})
    return examples


def _labels_from_examples(examples: list[dict]) -> list[str]:
    labels = sorted({label for example in examples for label in example["labels"]})
    preferred = [
        "=",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        "SELECT",
        "FROM",
        "WHERE",
        "JOIN",
        "AND",
        "OR",
        "NOT",
        "COUNT",
        "SUM",
        "AVG",
        "MIN",
        "MAX",
        "GROUP BY",
        "ORDER BY",
        "HAVING",
        "LIMIT",
        "BETWEEN",
        "IN",
        "NOT IN",
        "LIKE",
        "IS NULL",
        "IS NOT NULL",
        "DISTINCT",
    ]
    ordered = [label for label in preferred if label in labels]
    ordered.extend(label for label in labels if label not in ordered)
    return ordered


class SqlClauseDataset(Dataset):
    def __init__(
        self,
        examples: list[dict],
        tokenizer,
        labels: list[str],
        max_length: int,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.labels = labels
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        encoded = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        target = [0.0] * len(self.labels)
        for label in example["labels"]:
            label_index = self.label2id.get(label)
            if label_index is not None:
                target[label_index] = 1.0
        encoded["labels"] = target
        return {key: torch.tensor(value) for key, value in encoded.items()}


def _multi_label_metrics(
    model,
    loader,
    device,
    threshold: float,
) -> dict:
    model.eval()
    tp = fp = fn = exact = total = 0

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }
            logits = model(**inputs).logits
            predictions = (torch.sigmoid(logits) >= threshold).float()

            tp += int(((predictions == 1) & (labels == 1)).sum().item())
            fp += int(((predictions == 1) & (labels == 0)).sum().item())
            fn += int(((predictions == 0) & (labels == 1)).sum().item())
            exact += int((predictions == labels).all(dim=1).sum().item())
            total += int(labels.shape[0])

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact / total if total else 0.0,
    }


def _expand_classifier_head(model, old_label_count: int, new_label_count: int) -> None:
    if new_label_count <= old_label_count:
        return

    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, torch.nn.Linear):
        raise ValueError(
            "Cannot extend this classifier head automatically. "
            "Retrain with the full CSV so the model can create the new labels."
        )
    if classifier.out_features != old_label_count:
        raise ValueError(
            "Existing model label count does not match labels.json. "
            "Retrain with the full CSV to rebuild the classifier head."
        )

    expanded = torch.nn.Linear(classifier.in_features, new_label_count)
    expanded = expanded.to(
        device=classifier.weight.device,
        dtype=classifier.weight.dtype,
    )
    with torch.no_grad():
        expanded.weight[:old_label_count].copy_(classifier.weight)
        expanded.bias[:old_label_count].copy_(classifier.bias)
        initializer_range = float(getattr(model.config, "initializer_range", 0.02))
        torch.nn.init.normal_(
            expanded.weight[old_label_count:],
            mean=0.0,
            std=initializer_range,
        )
        torch.nn.init.zeros_(expanded.bias[old_label_count:])
    model.classifier = expanded


def train(
    dataset_csv: str | None = None,
    output_dir: str | None = None,
    feedback_jsonl: str | None = None,
    feedback_only: bool = False,
    init_model_dir: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    max_length: int | None = None,
    train_split: float | None = None,
    limit: int | None = None,
    threshold: float = 0.5,
    progress_every: int = 50,
) -> None:
    config = _load_json(CONFIG_PATH)
    training_config = config.get("training", {})
    dataset_path = Path(dataset_csv) if dataset_csv else DEFAULT_DATASET
    model_dir = Path(output_dir) if output_dir else MODEL_DIR
    feedback_path = Path(feedback_jsonl) if feedback_jsonl else SQL_CLAUSE_FEEDBACK_PATH
    init_model_path = Path(init_model_dir) if init_model_dir else None
    source_temp_dir = None

    feedback_examples = _load_feedback_examples(feedback_path)
    if feedback_only:
        examples = feedback_examples
        if limit:
            examples = examples[:limit]
        if not examples:
            raise ValueError(f"No feedback examples found at {feedback_path}")
        if init_model_path is None:
            init_model_path = model_dir
        if not init_model_path.exists():
            raise ValueError(
                "Feedback-only fine-tuning needs an existing model. "
                f"Expected one at {init_model_path}"
            )
        print(f"loaded {len(examples)} feedback examples from {feedback_path}")
    else:
        examples = _read_csv_examples(dataset_path)
        if limit:
            examples = examples[:limit]
        if feedback_examples:
            examples.extend(feedback_examples)
            print(
                f"loaded {len(feedback_examples)} feedback examples from "
                f"{feedback_path}"
            )
    if not examples:
        raise ValueError(f"No labeled CSV examples found at {dataset_path}")
    dataset_description = str(feedback_path) if feedback_only else str(dataset_path)

    model_source = init_model_path if init_model_path else config["model_name"]
    if init_model_path and init_model_path.resolve() == model_dir.resolve():
        source_temp_dir = tempfile.TemporaryDirectory(
            prefix="tadka_sql_clause_source_"
        )
        model_source = Path(source_temp_dir.name)
        shutil.copytree(init_model_path, model_source, dirs_exist_ok=True)
        print(f"copied current model to temporary training source {model_source}")
    labels_path = init_model_path / "labels.json" if init_model_path else None
    existing_labels = None
    new_feedback_labels = []
    if labels_path and labels_path.exists():
        existing_labels = load_label_list(labels_path)
        new_feedback_labels = [
            label for label in _labels_from_examples(examples)
            if label not in existing_labels
        ]
        if new_feedback_labels and not feedback_only:
            raise ValueError(
                "Feedback contains labels missing from the existing model: "
                f"{new_feedback_labels}. Retrain once with the full CSV so the "
                "classifier head includes those labels."
            )
        labels = existing_labels + new_feedback_labels
        if new_feedback_labels:
            print(
                "extending classifier labels with feedback-only labels: "
                f"{new_feedback_labels}"
            )
    elif feedback_only:
        raise ValueError(
            "Feedback-only fine-tuning needs the existing model labels file. "
            f"Expected {labels_path}"
        )
    else:
        labels = _labels_from_examples(examples)
    if not labels:
        raise ValueError("No SQL labels found in dataset.")

    rng = random.Random(training_config.get("random_seed", 42))
    rng.shuffle(examples)
    if feedback_only:
        train_examples = examples
        val_examples = examples
    else:
        split_ratio = train_split or training_config.get("train_split", 0.85)
        split = max(1, min(len(examples) - 1, int(len(examples) * split_ratio)))
        train_examples = examples[:split]
        val_examples = examples[split:]

    max_length = max_length or training_config.get("max_length", 128)
    batch_size = batch_size or training_config.get("batch_size", 16)
    learning_rate = learning_rate or training_config.get("learning_rate", 3e-5)
    epochs = epochs or training_config.get("epochs", 3)

    tokenizer = AutoTokenizer.from_pretrained(str(model_source))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    if existing_labels:
        existing_label2id = {label: i for i, label in enumerate(existing_labels)}
        existing_id2label = {i: label for label, i in existing_label2id.items()}
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_source),
            num_labels=len(existing_labels),
            id2label=existing_id2label,
            label2id=existing_label2id,
            problem_type="multi_label_classification",
        )
        _expand_classifier_head(model, len(existing_labels), len(labels))
        model.num_labels = len(labels)
        model.config.num_labels = len(labels)
        model.config.id2label = id2label
        model.config.label2id = label2id
        model.config.problem_type = "multi_label_classification"
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_source),
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            problem_type="multi_label_classification",
        )
    if init_model_path:
        print(f"continuing training from {init_model_path}")

    train_data = SqlClauseDataset(train_examples, tokenizer, labels, max_length)
    val_data = SqlClauseDataset(val_examples, tokenizer, labels, max_length)
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
        epoch_start = time.time()
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(output.loss.detach())

            if progress_every and step % progress_every == 0:
                avg_loss = total_loss / step
                elapsed = time.time() - epoch_start
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={avg_loss:.4f} elapsed={elapsed / 60:.1f}m",
                    flush=True,
                )

        metrics = _multi_label_metrics(model, val_loader, device, threshold)
        print(
            f"epoch={epoch} loss={total_loss / max(len(train_loader), 1):.4f} "
            f"micro_f1={metrics['f1']:.3f} exact_match={metrics['exact_match']:.3f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            model_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)
            save_label_list(labels, model_dir / "labels.json")
            with open(model_dir / "training_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset": dataset_description,
                        "feedback_jsonl": str(feedback_path),
                        "feedback_only": feedback_only,
                        "init_model": str(model_source),
                        "labels": labels,
                        "threshold": threshold,
                        "best_f1": best_f1,
                        "best_epoch": best_epoch,
                    },
                    f,
                    indent=2,
                )
            print(f"  new best checkpoint saved to {model_dir} (f1={best_f1:.3f})")

        if epoch - best_epoch >= 2:
            print(f"  early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    print(f"training complete. best micro_f1={best_f1:.3f} at epoch={best_epoch}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train multi-label SQL clause/operator classifier from CSV."
    )
    parser.add_argument("--dataset-csv", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(MODEL_DIR))
    parser.add_argument("--feedback-jsonl", default=str(SQL_CLAUSE_FEEDBACK_PATH))
    parser.add_argument(
        "--feedback-only",
        action="store_true",
        help="Continue training an existing classifier using only JSONL feedback.",
    )
    parser.add_argument(
        "--init-model-dir",
        default=None,
        help="Existing model directory to continue from. Defaults to --output-dir in feedback-only mode.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--train-split", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        dataset_csv=args.dataset_csv,
        output_dir=args.output_dir,
        feedback_jsonl=args.feedback_jsonl,
        feedback_only=args.feedback_only,
        init_model_dir=args.init_model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        train_split=args.train_split,
        limit=args.limit,
        threshold=args.threshold,
        progress_every=args.progress_every,
    )
