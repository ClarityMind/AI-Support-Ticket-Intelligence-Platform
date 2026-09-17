"""Train reproducible baselines from the repository root."""

import argparse
import hashlib
import json
from pathlib import Path

from .category import train_category
from .config import ARTIFACT_DIR, CV_FOLDS, DEFAULT_DATA_PATH, TEST_SIZE
from .data import load_data
from .evaluate import save_artifacts
from .priority import train_priority


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train-category", "train-priority"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
        command.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
        command.add_argument("--test-size", type=float, default=TEST_SIZE)
        if name == "train-priority":
            command.add_argument("--cv-folds", type=int, default=CV_FOLDS)
            command.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        frame = load_data(args.data)
        task = args.command.removeprefix("train-")
        if task == "category":
            model, metrics = train_category(frame, args.test_size)
        else:
            model, metrics = train_priority(frame, args.test_size, args.cv_folds, args.n_jobs)
        with args.data.open("rb") as stream:
            metrics["dataset_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        paths = save_artifacts(model, metrics, task, args.artifacts)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=True, allow_nan=False))
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
