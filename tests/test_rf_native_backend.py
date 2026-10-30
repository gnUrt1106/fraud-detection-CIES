"""
`compute_mean_abs_shap(..., backend="native")` cho Random Forest (TreeSHAP của XGBoost, src/explainability/rf_gpu_shap.py)
phải cho cùng mean|SHAP| như backend "shap" (shap.TreeExplainer), còn "auto" vẫn là shap.TreeExplainer.
"""

import numpy as np
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from src.explainability.shap_utils import compute_mean_abs_shap  # noqa: E402


def _rf():
    rng = np.random.RandomState(0)
    X = np.hstack([rng.randn(2500, 5), (rng.rand(2500, 3) < 0.3).astype(float)])
    y = (rng.rand(2500) < 1 / (1 + np.exp(-(1.5 * X[:, 0] - 2.0 * X[:, 5] - 5.0)))).astype(int)
    rf = RandomForestClassifier(n_estimators=20, max_depth=8, min_samples_leaf=2, class_weight={0: 1.0, 1: 20.0},
                                random_state=1, n_jobs=2).fit(X, y)
    return rf, X[:333]


def test_native_matches_shap_backend_and_ignores_n_jobs():
    rf, X = _rf()
    ref, used_ref = compute_mean_abs_shap(rf, "random_forest", X, batch_size=100, backend="shap", n_jobs=1)
    nat, used_nat = compute_mean_abs_shap(rf, "random_forest", X, batch_size=77, backend="native", n_jobs=4)
    assert used_ref == "tree"
    assert used_nat in ("tree-xgb-cpu", "tree-xgb-gpu")
    assert np.allclose(nat, ref, rtol=1e-5, atol=1e-9)


def test_auto_backend_for_random_forest_is_unchanged():
    rf, X = _rf()
    _, used = compute_mean_abs_shap(rf, "random_forest", X[:50], backend="auto", n_jobs=1)
    assert used == "tree"


def test_force_cpu_env_gives_cpu_explainer(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    rf, X = _rf()
    _, used = compute_mean_abs_shap(rf, "random_forest", X[:50], backend="native")
    assert used == "tree-xgb-cpu"


def test_native_still_rejected_for_other_models():
    rf, X = _rf()
    with pytest.raises(ValueError):
        compute_mean_abs_shap(rf, "logistic_regression", X[:10], X_background=X[:50], backend="native")
