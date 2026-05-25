from module1_asr.sql_ops.train import train


if __name__ == "__main__":
    from module1_asr.sql_ops.train import _parse_args

    args = _parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        limit=args.limit,
        dataset_jsonl=args.dataset_jsonl,
        export_synthetic=args.export_synthetic,
    )
