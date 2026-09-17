import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data import load_data, prepare_data, split_data, validate_columns


class DataTests(unittest.TestCase):
    def test_required_columns(self):
        with self.assertRaisesRegex(ValueError, "body"):
            validate_columns(pd.DataFrame({"subject": []}), ("subject", "body"))

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Dataset not found"):
                load_data(Path(directory) / "missing.csv")

    def test_english_filter_and_task_missing_values(self):
        frame = pd.DataFrame({"language": ["en", "en", "de", "EN"],
                              "subject": [None] * 4, "body": ["ticket", None, "other", "other"],
                              "queue": ["a"] * 4, "priority": ["low"] * 4})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tickets.csv"
            frame.to_csv(path, index=False)
            self.assertEqual(len(load_data(path)), 2)
        X, _ = prepare_data(frame, "queue")
        self.assertEqual(list(X.columns), ["subject", "body"])
        self.assertEqual(len(X), 1)
        self.assertEqual(len(prepare_data(frame, "priority")[0]), 2)
        self.assertTrue(pd.isna(frame.loc[0, "subject"]))

    def test_invalid_labels_and_empty_data(self):
        frame = pd.DataFrame({"language": ["en"], "subject": [""], "body": ["x"], "queue": [None]})
        with self.assertRaisesRegex(ValueError, "missing or empty"):
            prepare_data(frame, "queue")
        frame["language"] = "de"
        with self.assertRaisesRegex(ValueError, "No eligible"):
            prepare_data(frame, "queue")

    def test_small_class_error(self):
        frame = pd.DataFrame({"language": ["en", "en"], "subject": ["", ""],
                              "body": ["one", "two"], "queue": ["a", "b"]})
        with self.assertRaisesRegex(ValueError, "two rows per class"):
            split_data(frame, "queue")
