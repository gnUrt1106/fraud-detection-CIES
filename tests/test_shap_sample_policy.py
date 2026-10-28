"""
Chính sách mẫu SHAP của chế độ chạy lại theo seed:
Sparkov = mẫu phân tầng cố định 20% test (111.144 dòng, 429/2.145 fraud), ULB = toàn bộ test; `merge` cảnh báo nếu các tổ hợp
của một dataset dùng mẫu SHAP khác nhau.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from src.explainability import rerun_stability as rs


N_TEST_SPARKOV, N_FRAUD_SPARKOV = 555_719, 2_145  # tập test Sparkov thật (fraudTest.csv)


def _sparkov_like_labels():
    y = np.zeros(N_TEST_SPARKOV, dtype=int)
    y[np.random.RandomState(7).choice(N_TEST_SPARKOV, N_FRAUD_SPARKOV, replace=False)] = 1
    return y


def test_sparkov_default_is_twenty_percent_of_the_test_set():
    assert rs.resolve_shap_rows("sparkov", N_TEST_SPARKOV) == 111_144
    assert rs.resolve_shap_rows("sparkov", N_TEST_SPARKOV, None) == 111_144


def test_ulb_default_is_the_whole_test_set():
    assert rs.resolve_shap_rows("ulb", 56_961) is None


@pytest.mark.parametrize("dataset", ["sparkov", "ulb"])
def test_all_forces_the_whole_test_and_an_integer_forces_that_many_rows(dataset):
    assert rs.resolve_shap_rows(dataset, 1000, "all") is None
    assert rs.resolve_shap_rows(dataset, 1000, "FULL") is None
    assert rs.resolve_shap_rows(dataset, 1000, 300) == 300
    assert rs.resolve_shap_rows(dataset, 1000, "300") == 300


def test_unknown_dataset_is_rejected_unless_the_size_is_explicit():
    with pytest.raises(ValueError):
        rs.resolve_shap_rows("other", 1000)
    assert rs.resolve_shap_rows("other", 1000, 50) == 50


def test_tiny_test_set_never_rounds_below_two_rows():
    assert rs.resolve_shap_rows("sparkov", 5) == 2


def test_default_sample_has_the_documented_size_and_fraud_count_and_is_reproducible():
    y = _sparkov_like_labels()
    n = rs.resolve_shap_rows("sparkov", len(y))
    idx = rs.stratified_eval_indices(y, n)
    assert len(idx) == 111_144
    assert int(y[idx].sum()) == 429  # round(111144 × 2145 / 555719) — số đã ghi trong báo cáo
    assert np.array_equal(idx, rs.stratified_eval_indices(y, n))  # cùng nhãn + cùng n → cùng tập dòng ở mọi lần chạy
    assert np.all(np.diff(idx) > 0)
    # Lệnh cũ `--shap-rows 111144` và mặc định mới chọn đúng cùng một tập dòng
    assert np.array_equal(idx, rs.stratified_eval_indices(y, 111_144))


def test_combo_job_applies_the_dataset_policy(monkeypatch):
    y_test = _sparkov_like_labels()[:20_000]
    seen = {}

    def fake_load(data_dir, dataset):
        return np.zeros((10, 2)), np.zeros(10), np.zeros((len(y_test), 2)), y_test, ["a", "b"], []

    def fake_run(model_name, technique, X_train, y_train, X_test, y_test_, feature_names, onehot_groups, seeds, **kw):
        seen.update(kw)
        return {"config": {}}

    monkeypatch.setattr(rs, "load_encoded_data", fake_load)
    monkeypatch.setattr(rs, "run_rerun_combo", fake_run)
    monkeypatch.setattr(rs, "dataset_params", lambda dataset, model: None)

    rs.combo_job(None, "xgboost", "smote", [0, 1], dataset="sparkov")
    assert seen["shap_rows"] == 4_000  # 20% của 20.000
    rs.combo_job(None, "xgboost", "smote", [0, 1], dataset="sparkov", shap_rows="all")
    assert seen["shap_rows"] is None
    rs.combo_job(None, "xgboost", "smote", [0, 1], dataset="sparkov", shap_rows=777)
    assert seen["shap_rows"] == 777
    rs.combo_job(None, "xgboost", "smote", [0, 1], dataset="ulb")
    assert seen["shap_rows"] is None


def _fake_result(combo, n_eval=None, n_fraud=None):
    model, tech = combo.split("__")
    config = {"data_fingerprint": "abc"}
    if n_eval is not None:
        config.update(n_eval_shap=n_eval, n_eval_fraud=n_fraud)
    return {"combination": combo, "model_name": model, "imbalance_technique": tech, "complete": True,
            "cies_metrics": {"cies_score": 0.9}, "config": config}


def test_merge_warns_when_combos_used_different_shap_samples():
    merged = rs.merge_rerun_results({
        "a.json": [_fake_result("xgboost__smote", 111_144, 429)],
        "b.json": [_fake_result("catboost__smote", 555_719, 2_145)],
    })
    assert any("Mẫu SHAP KHÁC" in p for p in merged["problems"])


def test_merge_is_quiet_when_every_combo_used_the_same_sample():
    merged = rs.merge_rerun_results({
        "a.json": [_fake_result("xgboost__smote", 111_144, 429)],
        "b.json": [_fake_result("catboost__smote", 111_144, 429), _fake_result("ann__smote", 111_144, 429)],
    })
    assert merged["problems"] == []
    assert len(merged["results"]) == 3


def test_merge_ignores_old_result_files_without_the_sample_fields():
    merged = rs.merge_rerun_results({
        "old.json": [_fake_result("xgboost__smote")],
        "new.json": [_fake_result("catboost__smote", 111_144, 429)],
    })
    assert merged["problems"] == []


def _load_cli():
    path = Path(rs.__file__).resolve().parents[2] / "scripts" / "run_rerun.py"
    spec = importlib.util.spec_from_file_location("run_rerun_cli", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_cli_shap_rows_accepts_all_or_an_integer():
    cli = _load_cli()
    assert cli._shap_rows_arg("all") == "all"
    assert cli._shap_rows_arg("111144") == 111_144
    for bad in ("abc", "1", "0", "-5"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli._shap_rows_arg(bad)
