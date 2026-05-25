import argparse
from pathlib import Path


BASE_DIR = Path(__file__).parents[1]
RAW_DIR = BASE_DIR / "datasets" / "raw"

DATASETS = {
    "spider": {
        "repo_id": "xlangai/spider",
        "local_dir": RAW_DIR / "spider",
    },
    "wikisql": {
        "repo_id": "Salesforce/wikisql",
        "local_dir": RAW_DIR / "wikisql",
    },
}


def download_dataset(name: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install it with: pip install huggingface_hub"
        ) from exc

    spec = DATASETS[name]
    target = spec["local_dir"]
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type="dataset",
        local_dir=target,
        local_dir_use_symlinks=False,
    )
    return target


def _parse_args():
    parser = argparse.ArgumentParser(description="Download SQL datasets into datasets/raw.")
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS.keys(), "all"],
        default="all",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    names = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    for name in names:
        path = download_dataset(name)
        print(f"downloaded {name} to {path}")
