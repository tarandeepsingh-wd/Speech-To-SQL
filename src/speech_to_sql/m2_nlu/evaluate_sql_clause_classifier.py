from speech_to_sql.m2_nlu.sql_ops.evaluate_clause_classifier import evaluate


if __name__ == "__main__":
    from speech_to_sql.m2_nlu.sql_ops.evaluate_clause_classifier import _parse_args

    args = _parse_args()
    evaluate(
        dataset_csv=args.dataset_csv,
        threshold=args.threshold,
        limit=args.limit,
        show_errors=args.show_errors,
    )
