"""Shared reproducibility settings and relative paths."""

from pathlib import Path

RANDOM_STATE = 42
DEFAULT_DATA_PATH = Path("dataset/dataset-tickets-multi-lang-4-20k.csv")
ARTIFACT_DIR = Path("artifacts")
TEST_SIZE = 0.2
CV_FOLDS = 5
MAX_ITER = 5000
