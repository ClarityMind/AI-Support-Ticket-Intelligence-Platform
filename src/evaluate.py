"""Shared evaluation and artifact persistence."""

import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from .config import ARTIFACT_DIR, RANDOM_STATE


def evaluate(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame,
             y_train: pd.Series, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    labels = model.classes_.tolist()
    report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)
    return {
        "train_rows": len(y_train), "test_rows": len(y_test), "random_state": RANDOM_STATE,
        "train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "majority_baseline_test_accuracy": float(y_test.eq(y_train.mode().iloc[0]).mean()),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "classification_report": report, "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
    }


def save_artifacts(model: Pipeline, metrics: dict, task: str,
                   directory: str | Path = ARTIFACT_DIR) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / ("category_pipeline.joblib" if task == "category" else "priority_model.joblib")
    metrics_path = directory / f"{task}_metrics.json"
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return model_path, metrics_path
