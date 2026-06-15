import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

from speech_to_sql.m2_nlu.extractor import extract_ingredients
from speech_to_sql.m2_nlu.ingredient_reranker import (
    FEATURE_SCHEMA_VERSION,
    MODEL_DIR,
    build_candidate_features,
)
from speech_to_sql.m2_nlu.feedback_logger import INGREDIENT_FEEDBACK_FILE


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANDIDATE_DATASET = (
    PACKAGE_DIR / "datasets" / "processed" /
    "ingredient_reranker_candidates.jsonl"
)


def _ensure_tests_on_path() -> None:
    for directory in [
        PROJECT_ROOT / "tests" / "integration",
        PROJECT_ROOT / "tests" / "unit",
    ]:
        path = str(directory)
        if path not in sys.path:
            sys.path.insert(0, path)


def _full_suite_cases() -> list[tuple[str, list[str]]]:
    _ensure_tests_on_path()
    import test_m2_nlu_full

    rows = []
    for _category_name, cases in test_m2_nlu_full.CATEGORIES.values():
        for query, must_ing, _must_diet, _must_uten, must_empty, _must_nutr in cases:
            rows.append((query, [] if must_empty else list(must_ing)))
    return rows


def _stress_suite_cases() -> list[tuple[str, list[str]]]:
    _ensure_tests_on_path()
    import test_m2_nlu_stress

    rows = []
    for case in test_m2_nlu_stress.generate_cases():
        rows.append((case["query"], [] if case.get("empty") else list(case["ingredients"])))
    return rows


def load_gold_cases(source: str) -> list[tuple[str, list[str]]]:
    rows = []
    if source in {"full", "all"}:
        rows.extend(_full_suite_cases())
    if source in {"stress", "all"}:
        rows.extend(_stress_suite_cases())
    return rows


def _normalize_ingredient_list(values) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized = []
    for value in values:
        item = str(value).strip().lower()
        if item and item not in normalized:
            normalized.append(item)
    return normalized


def load_feedback_cases(path: Path) -> list[tuple[str, list[str]]]:
    if not path.exists():
        return []

    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query = str(row.get("text") or row.get("query") or "").strip()
            corrected = _normalize_ingredient_list(
                row.get("corrected_ingredients") or row.get("ingredients")
            )
            if not query:
                raise ValueError(f"{path}:{line_no} missing query/text")
            cases.append((query, corrected))
    return cases


def build_training_rows(cases: list[tuple[str, list[str]]]) -> list[dict]:
    rows = []
    candidate_cache: dict[str, list[dict]] = {}
    for query, gold_ingredients in cases:
        gold = {item.lower().strip() for item in gold_ingredients}
        if query not in candidate_cache:
            candidate_cache[query] = extract_ingredients(query)
        for match in candidate_cache[query]:
            canonical = str(match.get("canonical", "")).lower().strip()
            rows.append({
                "query": query,
                "gold_ingredients": sorted(gold),
                "match": match,
                "features": build_candidate_features(query, match),
                "label": int(canonical in gold),
            })
    return rows


def export_training_rows(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            match = row["match"]
            f.write(json.dumps({
                "query": row["query"],
                "gold_ingredients": row["gold_ingredients"],
                "candidate_text": match.get("text"),
                "canonical": match.get("canonical"),
                "span": list(match.get("span", [])),
                "layer": match.get("layer"),
                "source": match.get("source"),
                "extractor_score": match.get("score"),
                "features": row["features"],
                "label": row["label"],
            }, ensure_ascii=True) + "\n")


def _class_counts(labels: list[int]) -> dict[int, int]:
    return {label: labels.count(label) for label in sorted(set(labels))}


def _sample_weights(rows: list[dict], rare_positive_target: int = 8) -> list[float]:
    positives_by_canonical = Counter(
        str(row["match"].get("canonical", "")).lower().strip()
        for row in rows
        if row["label"] == 1
    )
    weights = []
    for row in rows:
        if row["label"] != 1:
            weights.append(1.0)
            continue
        canonical = str(row["match"].get("canonical", "")).lower().strip()
        count = max(positives_by_canonical.get(canonical, 1), 1)
        weights.append(max(1.0, rare_positive_target / count))
    return weights


def _top_weights(model, limit: int = 8) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    vectorizer = model.named_steps["dictvectorizer"]
    logistic = model.named_steps["logisticregression"]
    weights = list(zip(vectorizer.get_feature_names_out(), logistic.coef_[0]))
    positive = sorted(weights, key=lambda item: item[1], reverse=True)[:limit]
    negative = sorted(weights, key=lambda item: item[1])[:limit]
    return positive, negative


def _calibrate_threshold(y_true: list[int], probabilities: list[float]) -> float:
    from sklearn.metrics import precision_recall_fscore_support

    candidates = []
    for step in range(1, 1000):
        threshold = step / 1000.0
        y_pred = [int(probability >= threshold) for probability in probabilities]
        precision, recall, f1, _support = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        neg_precision, neg_recall, neg_f1, _support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0],
            average="macro",
            zero_division=0,
        )
        candidates.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "negative_precision": neg_precision,
            "negative_recall": neg_recall,
            "negative_f1": neg_f1,
        })

    eligible = [
        item for item in candidates
        if item["negative_recall"] >= 0.90
    ] or candidates
    best = max(
        eligible,
        key=lambda item: (
            item["f1"],
            item["recall"],
            item["negative_recall"],
        ),
    )
    return float(best["threshold"])


def train(
    source: str = "all",
    output_dir: str | None = None,
    feedback_jsonl: str | None = str(INGREDIENT_FEEDBACK_FILE),
    feedback_only: bool = False,
    feedback_repeat: int = 25,
    threshold: float | None = None,
    limit: int | None = None,
    random_seed: int = 42,
    export_candidates: str | None = str(DEFAULT_CANDIDATE_DATASET),
) -> dict:
    try:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is required for ingredient reranker training. "
            "Install requirements.txt first."
        ) from exc

    feedback_cases = (
        load_feedback_cases(Path(feedback_jsonl))
        if feedback_jsonl else []
    )
    cases = [] if feedback_only else load_gold_cases(source)
    repeat = max(feedback_repeat, 1)
    cases.extend(feedback_cases * repeat)
    if limit:
        cases = cases[:limit]

    rows = build_training_rows(cases)
    if not rows:
        raise ValueError("No ingredient candidates were generated for training.")
    if export_candidates:
        export_training_rows(rows, Path(export_candidates))

    X = [row["features"] for row in rows]
    y = [row["label"] for row in rows]
    weights = _sample_weights(rows)
    counts = _class_counts(y)
    if len(counts) < 2:
        raise ValueError(f"Need both positive and negative candidates; got {counts}.")

    stratify = y if min(counts.values()) >= 2 else None
    X_train, X_test, y_train, y_test, weight_train, _weight_test = train_test_split(
        X,
        y,
        weights,
        test_size=0.25,
        random_state=random_seed,
        stratify=stratify,
    )

    eval_model = make_pipeline(
        DictVectorizer(sparse=True),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_seed,
        ),
    )
    eval_model.fit(
        X_train,
        y_train,
        logisticregression__sample_weight=weight_train,
    )
    probabilities = [item[1] for item in eval_model.predict_proba(X_test)]
    selected_threshold = (
        _calibrate_threshold(y_test, probabilities)
        if threshold is None else threshold
    )
    predictions = [
        int(probability >= selected_threshold)
        for probability in probabilities
    ]

    model = make_pipeline(
        DictVectorizer(sparse=True),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_seed,
        ),
    )
    model.fit(
        X,
        y,
        logisticregression__sample_weight=weights,
    )

    output_path = Path(output_dir) if output_dir else MODEL_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    model_file = output_path / "model.pkl"
    payload = {
        "model": model,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "threshold": selected_threshold,
        "source": source,
        "feedback_jsonl": feedback_jsonl,
        "feedback_count": len(feedback_cases),
        "feedback_repeat": repeat,
        "candidate_count": len(rows),
        "class_counts": counts,
    }
    with open(model_file, "wb") as f:
        pickle.dump(payload, f)

    print(
        f"cases={len(cases)} feedback_cases={len(feedback_cases)} "
        f"candidates={len(rows)} labels={counts}"
    )
    if export_candidates:
        print(f"candidate_dataset={export_candidates}")
    print(f"saved={model_file}")
    print(f"threshold={selected_threshold:.3f}")
    print("confusion_matrix=")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions, digits=3))

    positive, negative = _top_weights(model)
    print("top_positive_features=" + ", ".join(f"{name}:{weight:.3f}" for name, weight in positive))
    print("top_negative_features=" + ", ".join(f"{name}:{weight:.3f}" for name, weight in negative))

    return payload


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train a logistic-regression ingredient candidate reranker."
    )
    parser.add_argument("--source", choices=["full", "stress", "all"], default="all")
    parser.add_argument("--output-dir", default=str(MODEL_DIR))
    parser.add_argument("--feedback-jsonl", default=str(INGREDIENT_FEEDBACK_FILE))
    parser.add_argument("--feedback-only", action="store_true")
    parser.add_argument("--feedback-repeat", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--export-candidates", default=str(DEFAULT_CANDIDATE_DATASET))
    parser.add_argument("--export-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.export_only:
        feedback_cases = (
            load_feedback_cases(Path(args.feedback_jsonl))
            if args.feedback_jsonl else []
        )
        cases = [] if args.feedback_only else load_gold_cases(args.source)
        repeat = max(args.feedback_repeat, 1)
        cases.extend(feedback_cases * repeat)
        if args.limit:
            cases = cases[:args.limit]
        rows = build_training_rows(cases)
        export_training_rows(rows, Path(args.export_candidates))
        print(
            f"cases={len(cases)} feedback_cases={len(feedback_cases)} "
            f"candidates={len(rows)}"
        )
        print(f"candidate_dataset={args.export_candidates}")
    else:
        train(
            source=args.source,
            output_dir=args.output_dir,
            feedback_jsonl=args.feedback_jsonl,
            feedback_only=args.feedback_only,
            feedback_repeat=args.feedback_repeat,
            threshold=args.threshold,
            limit=args.limit,
            random_seed=args.random_seed,
            export_candidates=args.export_candidates,
        )
