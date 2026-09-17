"""Select priority models using training-only stratified cross-validation."""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from .config import CV_FOLDS, RANDOM_STATE, TEST_SIZE
from .data import split_data
from .evaluate import evaluate
from .features import FEATURE_COLUMNS, priority_features


def parameter_grid() -> list[dict]:
    """A compact regularized search informed by the original experiments."""
    return [
        {"classifier": [DecisionTreeClassifier(random_state=RANDOM_STATE)],
         "classifier__max_depth": [3, 5, 10], "classifier__min_samples_leaf": [1, 8]},
        {"classifier": [RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)],
         "classifier__max_depth": [5, 10], "classifier__min_samples_leaf": [4, 8],
         "classifier__n_estimators": [100, 200]},
        {"classifier": [GradientBoostingClassifier(random_state=RANDOM_STATE)],
         "classifier__max_depth": [1, 3], "classifier__n_estimators": [100],
         "classifier__learning_rate": [0.05, 0.1]},
    ]


def train_priority(frame: pd.DataFrame, test_size: float = TEST_SIZE,
                   cv_folds: int = CV_FOLDS, n_jobs: int = 1) -> tuple[Pipeline, dict]:
    X_train, X_test, y_train, y_test = split_data(frame, "priority", test_size)
    if cv_folds < 2 or y_train.value_counts().min() < cv_folds:
        raise ValueError("Each training class must have at least cv_folds rows; cv_folds must be >= 2.")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    # These features are stateless per-row calculations: no corpus statistics are fitted.
    features = FunctionTransformer(priority_features)
    training_features = features.fit_transform(X_train)
    search = GridSearchCV(
        Pipeline([("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
        parameter_grid(), scoring="accuracy", cv=cv, n_jobs=n_jobs, error_score="raise",
    )
    search.fit(training_features, y_train)
    classifier = search.best_estimator_.named_steps["classifier"]
    model = Pipeline([("features", features), ("classifier", classifier)])
    # The holdout is first transformed and evaluated after CV selection and refit.
    metrics = evaluate(model, X_train, X_test, y_train, y_test)
    candidates = []
    for params, mean, std in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"],
                                 search.cv_results_["std_test_score"], strict=True):
        candidates.append({"model": type(params["classifier"]).__name__,
                           "parameters": {k.removeprefix("classifier__"): v for k, v in params.items() if k != "classifier"},
                           "cv_mean": float(mean), "cv_std": float(std)})
    selected = candidates[search.best_index_]
    metrics.update(task="priority", test_size=test_size, cv_folds=cv_folds,
                   selection_metric="accuracy", selected_model=selected["model"],
                   selected_hyperparameters=classifier.get_params(),
                   cv_mean=selected["cv_mean"], cv_std=selected["cv_std"], cv_candidates=candidates,
                   feature_importance=dict(zip(FEATURE_COLUMNS, classifier.feature_importances_.tolist(), strict=True)))
    return model, metrics
