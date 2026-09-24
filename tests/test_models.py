import pandas as pd

from tabtrace.models.baseline import build_baseline_pipeline, train_baseline
from tabtrace.models.tuned import build_tuned_pipeline, train_tuned


def _toy_data():
    X = pd.DataFrame(
        {
            "mean area": [10.0, 20.0] * 10,
            "mean perimeter": [2.0, 5.0] * 10,
            "mean concavity": [1.0, 2.0] * 10,
            "mean symmetry": [0.5, 0.6] * 10,
            "worst radius": [30.0, 10.0] * 10,
            "mean radius": [10.0, 10.0] * 10,
        }
    )
    y = pd.Series([0, 1] * 10, name="target")
    return X, y


def test_baseline_pipeline():
    pipe = build_baseline_pipeline()
    assert pipe is not None
    assert "scaler" in pipe.named_steps
    assert "clf" in pipe.named_steps


def test_train_baseline():
    X, y = _toy_data()
    model = train_baseline(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)


def test_tuned_pipeline():
    pipe = build_tuned_pipeline()
    assert pipe is not None
    assert "scaler" in pipe.named_steps
    assert "clf" in pipe.named_steps


def test_train_tuned():
    X, y = _toy_data()
    model, summary = train_tuned(X, y, cv_folds=2)
    assert summary["cv_folds"] == 2
    assert "best_params" in summary
    preds = model.predict(X)
    assert len(preds) == len(X)
