import argparse
from collections import defaultdict
from pathlib import Path

from speech_to_sql.m2_nlu.sql_ops.clause_classifier import predict_sql_clause_labels
from speech_to_sql.m2_nlu.sql_ops.label_utils import read_labeled_csv
from speech_to_sql.m2_nlu.sql_ops.train_clause_classifier import DEFAULT_DATASET


def _metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(
    dataset_csv: str,
    threshold: float | None = None,
    limit: int | None = None,
    show_errors: int = 10,
) -> None:
    examples = read_labeled_csv(Path(dataset_csv))
    if limit:
        examples = examples[:limit]
    if not examples:
        raise ValueError(f"No labeled examples found in {dataset_csv}")

    tp = fp = fn = exact = 0
    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    mistakes = []

    for example in examples:
        gold = set(example["labels"])
        pred = {
            item["canonical"]
            for item in predict_sql_clause_labels(example["text"], threshold=threshold)
        }

        tp_items = gold & pred
        fp_items = pred - gold
        fn_items = gold - pred
        tp += len(tp_items)
        fp += len(fp_items)
        fn += len(fn_items)
        exact += int(gold == pred)

        for label in tp_items:
            per_label[label]["tp"] += 1
        for label in fp_items:
            per_label[label]["fp"] += 1
        for label in fn_items:
            per_label[label]["fn"] += 1

        if gold != pred and len(mistakes) < show_errors:
            mistakes.append((example["text"], sorted(gold), sorted(pred)))

    micro = _metrics(tp, fp, fn)
    print(f"examples={len(examples)}")
    print(f"exact_match={exact / len(examples):.3f}")
    print(
        "micro "
        f"precision={micro['precision']:.3f} "
        f"recall={micro['recall']:.3f} "
        f"f1={micro['f1']:.3f}"
    )

    print("\nper-label:")
    for label in sorted(per_label):
        item = per_label[label]
        scores = _metrics(item["tp"], item["fp"], item["fn"])
        print(
            f"  {label:<12} "
            f"p={scores['precision']:.3f} "
            f"r={scores['recall']:.3f} "
            f"f1={scores['f1']:.3f} "
            f"tp={item['tp']} fp={item['fp']} fn={item['fn']}"
        )

    if mistakes:
        print("\nexample mistakes:")
        for text, gold, pred in mistakes:
            print(f"- {text}")
            print(f"  gold: {gold}")
            print(f"  pred: {pred}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved SQL clause classifier on a labeled CSV."
    )
    parser.add_argument("--dataset-csv", default=str(DEFAULT_DATASET))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show-errors", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        dataset_csv=args.dataset_csv,
        threshold=args.threshold,
        limit=args.limit,
        show_errors=args.show_errors,
    )
