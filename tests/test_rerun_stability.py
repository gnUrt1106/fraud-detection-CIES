"""
test_rerun_stability.py — Kiểm tra chế độ chạy lại theo seed (không bootstrap) + SHAP chia đợt.

Không cần torch (đường ANN/DeepSHAP chỉ kiểm được khi có torch — chạy thật bằng scripts/run_rerun.py pilot).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.explainability import rerun_stability as rs
from src.explainability.shap_utils import compute_mean_abs_shap, compute_shap
from src.models.train import build_model, train_model

FEATURES = [f"f{i}" for i in range(6)]


def make_data(n=500, fraud_rate=0.08, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 6)
    score = 2.0 * X[:, 0] + 1.0 * X[:, 1] + 0.3 * X[:, 2] + 0.5 * rng.randn(n)
    thr = np.quantile(score, 1 - fraud_rate)
    y = (score >= thr).astype(int)
    return X, y


@pytest.fixture(scope="module")
def data():
    X, y = make_data()
    Xt, yt = make_data(n=300, seed=1)
    return X, y, Xt, yt


def _full_mean_abs(model, name, X_eval, X_bg):
    sv = compute_shap(model, name, X_eval, X_background=X_bg)
    sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return np.abs(sv).mean(axis=0)


# ===== Tiện ích =====

def test_parse_seeds():
    assert rs.parse_seeds("0-3") == [0, 1, 2, 3]
    assert rs.parse_seeds("0,2,5") == [0, 2, 5]
    assert rs.parse_seeds("0-1,5") == [0, 1, 5]
    assert rs.parse_seeds(7) == [7]
    with pytest.raises(ValueError):
        rs.parse_seeds("3-1")
    with pytest.raises(ValueError):
        rs.parse_seeds("1,1")
    with pytest.raises(ValueError):
        rs.parse_seeds("")


def test_stratified_eval_indices(data):
    _, _, _, yt = data
    assert np.array_equal(rs.stratified_eval_indices(yt, None), np.arange(len(yt)))
    assert np.array_equal(rs.stratified_eval_indices(yt, len(yt) + 5), np.arange(len(yt)))
    idx = rs.stratified_eval_indices(yt, 100)
    assert len(idx) == len(set(idx)) and np.all(np.diff(idx) > 0)
    assert abs(len(idx) - 100) <= 1
    assert 0 < yt[idx].sum() < len(idx)
    assert abs(yt[idx].mean() - yt.mean()) < 0.03  # giữ ~tỷ lệ fraud thật
    assert np.array_equal(idx, rs.stratified_eval_indices(yt, 100))  # tất định


def test_data_fingerprint(data):
    X, y, _, _ = data
    a = rs.data_fingerprint(X, y)
    assert a == rs.data_fingerprint(X.copy(), y.copy())
    X2 = X.copy()
    X2[0, 0] += 1e-9
    assert a != rs.data_fingerprint(X2, y)


# ===== SHAP chia đợt =====

@pytest.mark.parametrize("name", ["logistic_regression", "random_forest"])
def test_batched_mean_abs_shap_matches_full(data, name):
    X, y, Xt, _ = data
    params = {"n_estimators": 20, "max_depth": 5} if name == "random_forest" else None
    model = build_model(name, input_dim=X.shape[1], params=params, seed=0)
    trained = train_model(model, X, y, model_name=name)
    bg = rs.pick_background(X, name, n=50)
    expected = _full_mean_abs(trained, name, Xt[:90], bg)
    for batch in (7, 90, 1000):  # đợt lẻ, vừa đúng, lớn hơn dữ liệu
        got, used = compute_mean_abs_shap(trained, name, Xt[:90], X_background=bg, batch_size=batch)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-9)
    assert used in ("linear", "tree")


@pytest.mark.parametrize("name,params", [
    ("xgboost", {"n_estimators": 30, "max_depth": 4}),
    ("catboost", {"iterations": 30, "depth": 4, "verbose": 0}),
])
def test_native_backend_equals_shap_treeexplainer(data, name, params):
    X, y, Xt, _ = data
    trained = train_model(build_model(name, input_dim=X.shape[1], params=params, seed=0), X, y, model_name=name)
    via_shap, used_shap = compute_mean_abs_shap(trained, name, Xt[:80], batch_size=25, backend="shap")
    native, used_native = compute_mean_abs_shap(trained, name, Xt[:80], batch_size=25, backend="native")
    assert used_native == "tree-native" and used_shap == "tree"
    np.testing.assert_allclose(native, via_shap, rtol=1e-4, atol=1e-6)


def test_native_backend_rejected_for_other_models(data):
    X, y, Xt, _ = data
    trained = train_model(build_model("logistic_regression", input_dim=X.shape[1], seed=0), X, y,
                          model_name="logistic_regression")
    with pytest.raises(Exception):
        compute_mean_abs_shap(trained, "logistic_regression", Xt[:10], X_background=X[:20], backend="native")


# ===== Chạy 1 tổ hợp trọn các seed =====

def _run(data, model, tech, seeds, **kw):
    X, y, Xt, yt = data
    return rs.run_rerun_combo(model, tech, X, y, Xt, yt, FEATURES, [], seeds, shap_batch=64, **kw)


def test_deterministic_combo_gives_cies_one_and_flag(data):
    # Logistic Regression + class_weighting: không có ngẫu nhiên → 10 model giống hệt → CIES = 1 (không phải lỗi)
    r = _run(data, "logistic_regression", "class_weighting", [0, 1, 2, 3])
    assert r["complete"] and r["cies_metrics"]["n_runs"] == 4
    assert r["cies_metrics"]["cies_score"] == pytest.approx(1.0)
    assert r["diagnostics"]["identical_mean_abs_shap"] and r["diagnostics"]["identical_ranks"]
    assert r["pr_auc_summary"]["std"] == pytest.approx(0.0, abs=1e-12)
    assert r["config"]["n_eval_shap"] == len(data[3])  # SHAP trên TOÀN BỘ test


def test_random_forest_smote_with_per_seed_cache(data, tmp_path):
    params = {"n_estimators": 15, "max_depth": 6}
    seeds = [0, 1, 2]
    r = _run(data, "random_forest", "smote", seeds, cache_root=tmp_path, params=params)
    assert r["complete"] and r["pr_auc_summary"]["n"] == 3
    assert 0.0 <= r["cies_metrics"]["cies_score"] <= 1.0
    assert not r["diagnostics"]["identical_mean_abs_shap"]
    files = [rs.seed_cache_dir(tmp_path, s) / "train_encoded_smote.parquet" for s in seeds]
    assert all(f.exists() for f in files)  # mỗi seed 1 thư mục riêng, không đè nhau
    mtimes = [f.stat().st_mtime_ns for f in files]
    # Lần chạy lại dùng đúng cache (không tính lại), kết quả y hệt (tất định theo seed)
    r2 = _run(data, "random_forest", "smote", seeds, cache_root=tmp_path, params=params)
    assert [f.stat().st_mtime_ns for f in files] == mtimes
    assert r2["cies_metrics"]["cies_score"] == pytest.approx(r["cies_metrics"]["cies_score"])
    assert r2["config"]["data_fingerprint"] == r["config"]["data_fingerprint"]
    # Các seed cho resample khác nhau
    import pandas as pd
    d0 = pd.read_parquet(files[0])
    d1 = pd.read_parquet(files[1])
    assert not d0.equals(d1)


def test_combo_needs_two_good_seeds(data):
    with pytest.raises(RuntimeError, match="cần ít nhất 2"):
        _run(data, "logistic_regression", "class_weighting", [0])


def test_build_resample_cache_skips_class_weighting(data, tmp_path):
    X, y, _, _ = data
    r = rs.build_resample_cache("class_weighting", X, y, 0, FEATURES, [], tmp_path)
    assert "skipped" in r and not list(tmp_path.iterdir())
    r = rs.build_resample_cache("smote", X, y, 3, FEATURES, [], tmp_path)
    assert (tmp_path / "seed03" / "train_encoded_smote.parquet").exists() and r["n_fraud_after"] == r["n_after"] - r["n_fraud_after"]


def test_pilot_timing_runs(data):
    X, y, Xt, yt = data
    r = rs.pilot_timing("logistic_regression", X, y, Xt, yt, pilot_rows=60, n_runs_total=50)
    assert r["pilot_rows"] == pytest.approx(60, abs=1)
    assert r["est_seconds_per_run"] >= 0 and r["est_hours_for_model"] >= 0
    assert r["n_eval_shap"] == len(yt) and r["est_seconds_shap_eval"] == r["est_seconds_shap_full_test"]


def test_pilot_estimates_the_eval_sample_not_full_test(data):
    # Hồi quy: pilot từng ước SHAP trên TOÀN BỘ test trong khi train của Sparkov dùng mẫu 20% → giờ ước cao ~5 lần.
    X, y, Xt, yt = data
    r = rs.pilot_timing("logistic_regression", X, y, Xt, yt, pilot_rows=60, eval_rows=60)
    assert r["n_eval_shap"] == 60
    assert r["est_seconds_shap_eval"] <= r["est_seconds_shap_full_test"]
    big = rs.pilot_timing("logistic_regression", X, y, Xt, yt, pilot_rows=60, eval_rows=10 ** 9)
    assert big["n_eval_shap"] == len(yt)  # không vượt số dòng test


# ===== Gộp kết quả =====

def _fake(combo_model, tech, fp="aaa", complete=True):
    return {"combination": f"{combo_model}__{tech}", "model_name": combo_model, "imbalance_technique": tech,
            "complete": complete, "cies_metrics": {"cies_score": 0.9},
            "config": {"data_fingerprint": fp}}


def test_merge_ok_and_sorted():
    out = rs.merge_rerun_results({"a.json": [_fake("xgboost", "smote")], "b.json": [_fake("logistic_regression", "adasyn")]})
    assert out["problems"] == []
    assert [r["model_name"] for r in out["results"]] == ["logistic_regression", "xgboost"]


def test_merge_flags_duplicates_fingerprint_and_incomplete():
    out = rs.merge_rerun_results({
        "a.json": [_fake("xgboost", "smote"), _fake("xgboost", "adasyn", complete=False)],
        "b.json": [_fake("xgboost", "smote"), _fake("catboost", "smote", fp="bbb")],
    })
    text = " | ".join(out["problems"])
    assert "xgboost__smote" in text and "chưa đủ seed" in text and "vân tay" in text
    assert len([r for r in out["results"] if r["combination"] == "xgboost__smote"]) == 1


def test_merge_ignores_error_records():
    out = rs.merge_rerun_results({"a.json": [{"combination": "x__y", "error": "boom"}, _fake("xgboost", "smote")]})
    assert len(out["results"]) == 1


# ===== ULB đối xử ngang Sparkov =====

def test_ulb_loader_params_and_paths(tmp_path):
    import pandas as pd

    from src.config import ULB_FEATURE_COLS, ULB_TARGET_COL

    r = np.random.RandomState(0)

    def mk(n):
        d = pd.DataFrame(r.randn(n, 29), columns=list(ULB_FEATURE_COLS))
        d["Time"] = np.arange(n)
        d[ULB_TARGET_COL] = (r.rand(n) < 0.1).astype(int)
        return d

    mk(200).to_parquet(tmp_path / "ulb_train.parquet")
    mk(80).to_parquet(tmp_path / "ulb_test.parquet")
    Xtr, ytr, Xte, yte, names, groups = rs.load_encoded_data(tmp_path, "ulb")
    assert names == list(ULB_FEATURE_COLS) and "Time" not in names  # bỏ Time như notebook 05
    assert Xtr.shape == (200, 29) and Xte.shape == (80, 29) and groups == []
    with pytest.raises(ValueError):
        rs.load_encoded_data(tmp_path, "khac")
    # Sparkov dùng best_params.json (None = build_model tự nạp); ULB nạp best_params_ulb.json
    assert rs.dataset_params("sparkov", "xgboost") is None
    assert rs.dataset_params("ulb", "xgboost")
    assert rs.dataset_results_dir("ulb") != rs.dataset_results_dir("sparkov")
    assert rs.dataset_cache_root("ulb").name == "ulb" and rs.dataset_cache_root("sparkov").name != "ulb"
    # Hồi quy: --cache-root cho ULB từng dùng thẳng gốc → chung thư mục với Sparkov, cache hai dataset ghi đè nhau
    assert rs.dataset_cache_root("ulb", tmp_path) == tmp_path / "ulb"
    assert rs.dataset_cache_root("sparkov", tmp_path) == tmp_path


def test_parallel_tree_shap_matches_serial(data):
    X, y, Xt, _ = data
    trained = train_model(build_model("random_forest", input_dim=X.shape[1], params={"n_estimators": 15, "max_depth": 6}, seed=0),
                          X, y, model_name="random_forest")
    serial, used1 = compute_mean_abs_shap(trained, "random_forest", Xt[:120], batch_size=40, n_jobs=1)
    para, used2 = compute_mean_abs_shap(trained, "random_forest", Xt[:120], batch_size=40, n_jobs=2)
    assert used1 == used2 == "tree"
    np.testing.assert_allclose(para, serial, rtol=1e-9, atol=1e-12)
