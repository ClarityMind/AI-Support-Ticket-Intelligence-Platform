import unittest

import numpy as np
import pandas as pd

from features import (
    FEATURE_COLUMNS, combine_text, extract_priority_features, priority_features,
)


class FeatureTests(unittest.TestCase):
    def test_missing_text_is_empty(self):
        frame = pd.DataFrame({"subject": [None, np.nan, pd.NA, ""],
                              "body": [pd.NA, None, np.nan, ""]})
        self.assertEqual(combine_text(frame).tolist(), [" "] * 4)
        result = priority_features(frame)
        self.assertEqual(list(result.columns), list(FEATURE_COLUMNS))
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        self.assertTrue((result == 0).all().all())
        self.assertTrue(all(pd.api.types.is_numeric_dtype(dtype) for dtype in result.dtypes))

    def test_notebook_feature_semantics(self):
        result = extract_priority_features({"subject": "URGENT error 12", "body": "error please fix"})
        self.assertEqual(result["urgency_keyword_count"], 1)
        self.assertEqual(result["incident_keyword_count"], 2)
        self.assertEqual(result["action_request_count"], 1)
        self.assertEqual(result["digit_count"], 2)
        self.assertEqual(result["uppercase_word_ratio"], 0.1667)
        self.assertEqual(result["subject_body_similarity"], 0.2)

    def test_forbidden_fields_do_not_change_features(self):
        frame = pd.DataFrame({"subject": ["ASAP"], "body": ["please fix outage"]}, index=[8])
        augmented = frame.assign(answer="secret", queue="secret", priority="secret", tag_1="secret")
        pd.testing.assert_frame_equal(priority_features(frame), priority_features(augmented))
        pd.testing.assert_series_equal(combine_text(frame), combine_text(augmented))
        pd.testing.assert_frame_equal(priority_features(frame), priority_features(frame))
