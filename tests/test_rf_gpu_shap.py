"""
Test src/explainability/rf_gpu_shap.py: SHAP của Random Forest tính bằng XGBoost (đường CPU của cùng mã; GPU dùng đúng
các Booster này, chỉ khác device) phải khớp shap.TreeExplainer — giá trị, bias và tính cộng dồn.
"""

import numpy as np
import pytest

shap = pytest.importorskip("shap")
xgb = pytest.importorskip("xgboost")
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from src.explainability.rf_gpu_shap import RFXgbShap, float32_strict_upper, rf_to_boosters  # noqa: E402
from src.explainability.shap_utils import _class1_2d  # noqa: E402


def _data(n=3000, seed=0):
    rng = np.random.RandomState(seed)
    cont = rng.randn(n, 5) * np.array([1.0, 10.0, 0.01, 100.0, 1.0])
    binary = (rng.rand(n, 3) < 0.3).astype(float)  # cột 0/1 (giống one-hot) — nhiều giá trị trùng
    ints = rng.randint(0, 24, size=(n, 1)).astype(float)  # giờ trong ngày
    X = np.hstack([cont, binary, ints])
    logit = 1.5 * X[:, 0] + 0.05 * X[:, 1] - 2.0 * X[:, 5] + 0.1 * X[:, 8] - 6.0
    y = (rng.rand(n) < 1 / (1 + np.exp(-logit))).astype(int)
    assert 20 < y.sum() < n // 2
    return X, y


def _fit(**kw):
    X, y = _data()
    params = dict(n_estimators=24, max_depth=8, min_samples_split=6, min_samples_leaf=2,
                  class_weight={0: 1.0, 1: 15.0}, random_state=3, n_jobs=2)
    params.update(kw)
    rf = RandomForestClassifier(**params).fit(X, y)
    return rf, X[:400]


def _reference(rf, Xe):
    expl = shap.TreeExplainer(rf)
    return _class1_2d(expl.shap_values(Xe)), float(np.asarray(expl.expected_value).ravel()[-1])


def test_float32_threshold_equivalence():
    rng = np.random.RandomState(1)
    a = rng.randn(2000).astype(np.float32) * np.float32(50)
    b = np.nextafter(a, np.float32(np.inf)).astype(np.float32)
    thr = np.concatenate([
        (a.astype(np.float64) + b.astype(np.float64)) / 2.0,      # trung điểm hai float32 liền kề
        a.astype(np.float64), b.astype(np.float64),                  # đúng một giá trị float32
        rng.randn(2000) * 3.0,                                       # ngưỡng bất kỳ
    ])
    s = float32_strict_upper(thr)
    base = thr.astype(np.float32)
    for x in (np.nextafter(base, np.float32(-np.inf)), base, np.nextafter(base, np.float32(np.inf)),
              np.nextafter(np.nextafter(base, np.float32(np.inf)), np.float32(np.inf))):
        assert np.array_equal(x.astype(np.float64) <= thr, x < s)


@pytest.mark.parametrize("chunk,row_batch", [(7, 150), (1, 400), (24, 64)])
def test_matches_shap_tree_explainer(chunk, row_batch):
    rf, Xe = _fit()
    ref, ref_bias = _reference(rf, Xe)
    fn = RFXgbShap(rf, device="cpu", chunk_trees=chunk, row_batch=row_batch)
    got = fn(Xe)
    assert got.shape == ref.shape
    assert np.max(np.abs(got - ref)) < 1e-6
    proba = rf.predict_proba(Xe)[:, 1]
    assert np.max(np.abs(got.sum(axis=1) + fn.bias(Xe) - proba)) < 1e-5  # cộng dồn: bias + Σφ = P(fraud)
    assert abs(float(fn.bias(Xe[:1])[0]) - ref_bias) < 1e-6
    assert np.array_equal(np.argsort(-np.abs(got).mean(axis=0)), np.argsort(-np.abs(ref).mean(axis=0)))


def test_chunk_size_does_not_change_result():
    rf, Xe = _fit(n_estimators=20)
    a = RFXgbShap(rf, device="cpu", chunk_trees=3)(Xe)
    b = RFXgbShap(rf, device="cpu", chunk_trees=20)(Xe)
    assert np.max(np.abs(a - b)) < 1e-6


@pytest.mark.parametrize("kw", [
    dict(bootstrap=False, class_weight=None),
    dict(max_depth=None, min_samples_leaf=1, n_estimators=12),                 # cây sâu
    dict(class_weight="balanced_subsample", max_samples=0.6),
    dict(max_features=0.5, min_samples_leaf=5, class_weight={0: 1.0, 1: 120.0}),
])
def test_other_forest_settings(kw):
    rf, Xe = _fit(**kw)
    ref, _ = _reference(rf, Xe)
    got = RFXgbShap(rf, device="cpu", chunk_trees=5)(Xe)
    assert np.max(np.abs(got - ref)) < 1e-6


def test_rejects_non_binary_classes():
    X, y = _data()
    y3 = np.where(y == 1, 2, 0)
    rf = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y3)
    with pytest.raises(ValueError):
        rf_to_boosters(rf, 3)
