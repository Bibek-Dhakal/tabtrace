from tabtrace.report import build_report, save_report


def test_build_report_and_save(tmp_path):
    baseline_cv = {"scoring": "f1", "mean": 0.95, "std": 0.01, "cv_folds": 5}
    tuned_cv = {"scoring": "f1", "mean": 0.85, "std": 0.02, "cv_folds": 5}
    tuned_search = {"cv_folds": 5, "scoring": "f1", "best_params": {"a": 1}}
    baseline_test = {
        "accuracy": 0.9,
        "precision": 0.9,
        "recall": 0.9,
        "f1": 0.9,
        "roc_auc": 0.9,
    }
    tuned_test = {
        "accuracy": 0.8,
        "precision": 0.8,
        "recall": 0.8,
        "f1": 0.8,
        "roc_auc": 0.8,
    }

    report = build_report(baseline_cv, tuned_cv, tuned_search, baseline_test, tuned_test)
    assert "Logistic Regression" in report
    assert "Observation" in report
    assert "baseline Logistic Regression model outperformed" in report

    # Test reverse observation where tuned > baseline
    tuned_cv["mean"] = 0.98
    report2 = build_report(baseline_cv, tuned_cv, tuned_search, baseline_test, tuned_test)
    assert "tuned Random Forest model outperformed" in report2

    path = tmp_path / "report.md"
    save_report(report, path)
    assert path.exists()
