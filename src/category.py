"""Queue classification with training-only TF-IDF fitting."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .config import MAX_ITER, RANDOM_STATE, TEST_SIZE
from .data import split_data
from .evaluate import evaluate
from .features import combine_text


def train_category(frame: pd.DataFrame, test_size: float = TEST_SIZE) -> tuple[Pipeline, dict]:
    X_train, X_test, y_train, y_test = split_data(frame, "queue", test_size)
    model = Pipeline([
        ("text", FunctionTransformer(combine_text)),
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE)),
    ])
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_train, X_test, y_train, y_test)
    metrics.update(task="category", test_size=test_size, selected_model="LogisticRegression",
                   max_iter=MAX_ITER)
    return model, metrics
