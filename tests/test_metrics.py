import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from tabtrace.evaluation.metrics import cross_validated_scores, failure_cases, holdout_metrics


def _toy_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] + X["b"] > 0).astype(int), name="target")
    return X, y


def test_holdout_metrics_perfect_classifier():
    X, y = _toy_data()

    class _PerfectEstimator:
        def predict(self, X):
            return y.values

        def predict_proba(self, X):
            proba = np.zeros((len(X), 2))
            proba[np.arange(len(X)), y.values] = 1.0
            return proba

    metrics = holdout_metrics(_PerfectEstimator(), X, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_cross_validated_scores_shape_and_bounds():
    X, y = _toy_data(n=100)
    model = DummyClassifier(strategy="stratified", random_state=0)
    result = cross_validated_scores(model, X, y, cv_folds=5, scoring="f1")
    assert result["cv_folds"] == 5
    assert len(result["fold_scores"]) == 5
    assert 0.0 <= result["mean"] <= 1.0


def test_failure_cases_returns_only_misclassified_rows():
    X, y = _toy_data()

    class _AlwaysZero:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    failures = failure_cases(_AlwaysZero(), X, y)
    assert (failures["true_label"] != failures["predicted_label"]).all()
    assert len(failures) == int((y == 1).sum())
