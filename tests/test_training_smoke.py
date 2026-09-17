import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV

from category import train_category
from cli import main
from data import split_data
from evaluate import save_artifacts
from priority import parameter_grid, train_priority


def tickets():
    rows = []
    for i in range(60):
        urgent = i % 2 == 0
        rows.append({"language": "en", "subject": "URGENT" if urgent else None,
                     "body": ("outage error please fix " if urgent else "billing invoice question ") + f"unique{i}",
                     "queue": "technical" if urgent else "billing",
                     "priority": "high" if urgent else "low", "answer": "forbidden"})
    return pd.DataFrame(rows)


class TrainingTests(unittest.TestCase):
    def test_category_training_vocabulary_and_roundtrip(self):
        frame = tickets()
        X_train, X_test, _, _ = split_data(frame, "queue")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model, metrics = train_category(frame)
        vocabulary = model.named_steps["tfidf"].vocabulary_
        for index in X_test.index:
            self.assertNotIn(f"unique{index}", vocabulary)
        for index in X_train.index:
            self.assertIn(f"unique{index}", vocabulary)
        self.assertGreater(metrics["test_accuracy"], 0.5)
        self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), len(X_test))
        self.check_artifact(model, metrics, "category", frame)

    def check_artifact(self, model, metrics, task, frame):
        inputs = frame[["subject", "body"]].iloc[:3]
        np.testing.assert_array_equal(model.predict(inputs), model.predict(frame.iloc[:3]))
        with tempfile.TemporaryDirectory() as directory:
            path, report = save_artifacts(model, metrics, task, directory)
            np.testing.assert_array_equal(model.predict(inputs), joblib.load(path).predict(inputs))
            self.assertEqual(json.loads(report.read_text(encoding="utf-8")), metrics)

    def test_priority_cv_uses_only_training_rows_and_selects_best_mean(self):
        frame = tickets()
        X_train, X_test, _, _ = split_data(frame, "priority")
        original_fit = GridSearchCV.fit
        observed = []

        def recording_fit(search, X, y, **kwargs):
            observed.extend(X.index)
            return original_fit(search, X, y, **kwargs)

        # One small configuration per supported family keeps the test fast.
        grids = parameter_grid()
        for grid in grids:
            for key, values in grid.items():
                grid[key] = values[:1]
            if "classifier__n_estimators" in grid:
                grid["classifier__n_estimators"] = [5]
        with patch("ai_support_ticket_intelligence_platform.priority.parameter_grid", return_value=grids), \
                patch.object(GridSearchCV, "fit", recording_fit):
            model, metrics = train_priority(frame, cv_folds=2)
            repeat, repeat_metrics = train_priority(frame, cv_folds=2)
        self.assertEqual(set(observed), set(X_train.index))
        self.assertFalse(set(observed) & set(X_test.index))
        self.assertEqual(len(metrics["cv_candidates"]), 3)
        self.assertEqual(metrics["cv_mean"], max(row["cv_mean"] for row in metrics["cv_candidates"]))
        self.assertEqual(metrics, repeat_metrics)
        np.testing.assert_array_equal(model.predict(X_test), repeat.predict(X_test))
        self.check_artifact(model, metrics, "priority", frame)

    def test_cv_rejects_insufficient_class_counts(self):
        with self.assertRaisesRegex(ValueError, "cv_folds"):
            train_priority(tickets(), cv_folds=50)

    def test_category_cli_saves_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "tickets.csv"
            artifacts = Path(directory) / "artifacts"
            tickets().to_csv(data, index=False)
            with patch("builtins.print"):
                main(["train-category", "--data", str(data), "--artifacts", str(artifacts)])
            report = json.loads((artifacts / "category_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["dataset_sha256"]), 64)
            self.assertTrue((artifacts / "category_pipeline.joblib").is_file())
