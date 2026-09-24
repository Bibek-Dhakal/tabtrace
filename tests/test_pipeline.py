from tabtrace.pipeline import run_pipeline


def test_run_pipeline(tmp_path):
    results = run_pipeline(output_dir=tmp_path)

    assert "baseline_cv" in results
    assert "tuned_cv" in results
    assert "val_set_size" in results

    assert (tmp_path / "split_indices.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "cv_results.json").exists()
    assert (tmp_path / "baseline_model.joblib").exists()
    assert (tmp_path / "tuned_model.joblib").exists()
