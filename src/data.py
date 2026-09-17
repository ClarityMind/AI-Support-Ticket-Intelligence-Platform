"""Validate data and isolate creation-time inputs before splitting."""

from pathlib import Path
from collections.abc import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DEFAULT_DATA_PATH, RANDOM_STATE, TEST_SIZE


def validate_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def english_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an independent copy of rows eligible for the English-only v1."""
    validate_columns(frame, ("language",))
    return frame.loc[frame["language"].eq("en")].copy()


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read a UTF-8 CSV and retain exactly language == 'en'."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Supply a CSV with --data.")
    frame = pd.read_csv(path, encoding="utf-8")
    validate_columns(frame, ("language", "subject", "body"))
    return english_rows(frame)


def prepare_data(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Apply task filtering without modifying input data or relabeling rows."""
    if target not in ("queue", "priority"):
        raise ValueError("Target must be queue or priority.")
    validate_columns(frame, ("language", "subject", "body", target))
    clean = english_rows(frame)
    if target == "queue":
        clean = clean.dropna(subset=["body"])
    if clean.empty:
        raise ValueError("No eligible English rows remain for training.")
    if clean[target].isna().any() or clean[target].astype(str).str.strip().eq("").any():
        raise ValueError(f"Target {target!r} contains missing or empty labels.")
    return clean[["subject", "body"]], clean[target].astype(str)


def split_data(frame: pd.DataFrame, target: str, test_size: float = TEST_SIZE) -> tuple:
    X, y = prepare_data(frame, target)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("Stratified training requires at least two classes and two rows per class.")
    try:
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
    except ValueError as error:
        raise ValueError(f"Cannot create stratified split; provide more rows per class or adjust test_size: {error}") from error
