from speech_to_sql.m2_nlu.sql_ops.train_clause_classifier import train


if __name__ == "__main__":
    from speech_to_sql.m2_nlu.sql_ops.train_clause_classifier import _parse_args

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
