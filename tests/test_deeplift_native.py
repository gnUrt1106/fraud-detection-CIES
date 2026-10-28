"""
DeepLIFT native (GPU/CPU, theo lô) phải TRÙNG shap.DeepExplainer đến sai số float32 — bằng chứng để dùng thay vòng lặp từng dòng
của shap khi tính SHAP ANN trên toàn bộ test. Chạy trong subprocess: torch không được nạp chung process với xgboost.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COMMON = """
import numpy as np, torch
from src.models.train import build_model, train_model, scale_ann_input
from src.explainability.shap_utils import _make_shap_batch_fn, compute_mean_abs_shap

def make_model(n=3000, d=21, seed=0, dropout=0.2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)); X[:, 3] = np.round(X[:, 3]); X[:, 5] = (X[:, 5] > 0).astype(float)
    y = ((X[:, 0] * 1.5 + X[:, 1] - 0.7 * X[:, 4] + 0.3 * rng.normal(size=n)) > 1.4).astype(int)
    m = build_model("ann", input_dim=d, class_weights=None, params={"lr": 0.002, "dropout": dropout, "epochs": 3, "batch_size": 512}, seed=seed)
    return train_model(m, X, y, model_name="ann"), X, rng
"""


def _run(code: str) -> None:
    r = subprocess.run([sys.executable, "-c", COMMON + textwrap.dedent(code)], cwd=ROOT,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, (r.stdout[-2000:] + "\n" + r.stderr[-3000:])


def test_native_matches_shap_deepexplainer_and_is_additive():
    _run("""
    trained, X, rng = make_model()
    bg = X[rng.choice(len(X), 300, replace=False)]
    ev = X[:60]
    ref, n1 = _make_shap_batch_fn(trained, "ann", bg, "auto")
    nat, n2 = _make_shap_batch_fn(trained, "ann", bg, "native")
    assert (n1, n2) == ("deep", "deep-native")
    v1, v2 = ref(ev), nat(ev)
    assert v1.shape == v2.shape == (60, 21)
    assert np.abs(v1 - v2).max() <= 1e-4 * np.abs(v1).max(), np.abs(v1 - v2).max()
    m1, m2 = np.abs(v1).mean(0), np.abs(v2).mean(0)
    assert np.allclose(m1, m2, rtol=1e-4) and np.array_equal(np.argsort(-m1), np.argsort(-m2))
    ann = trained["model"]; ann.eval()
    with torch.no_grad():
        fx = ann(torch.FloatTensor(scale_ann_input(trained, ev))).numpy().ravel()
        fb = ann(torch.FloatTensor(scale_ann_input(trained, bg))).numpy().ravel().mean()
    assert np.abs(v2.sum(1) - (fx - fb)).max() < 1e-4        # tính đầy đủ: tổng SHAP = f(x) - E[f(nền)]
    """)


def test_rows_equal_to_background_rows_take_the_small_delta_branch_like_shap():
    # x trùng hệt một dòng nền → Δ đầu vào = 0 ở mọi ReLU của cặp đó: phải đi nhánh gradient như shap.nonlinear_1d
    _run("""
    trained, X, rng = make_model(seed=3)
    bg = X[:200]
    ev = np.vstack([X[:25], X[1000:1025]])
    ref, _ = _make_shap_batch_fn(trained, "ann", bg, "auto")
    nat, _ = _make_shap_batch_fn(trained, "ann", bg, "native")
    v1, v2 = ref(ev), nat(ev)
    assert np.abs(v1 - v2).max() <= 1e-4 * np.abs(v1).max(), np.abs(v1 - v2).max()
    """)


def test_result_does_not_depend_on_chunk_size_and_batch_size():
    _run("""
    from src.explainability.deeplift_native import make_deeplift_fn
    trained, X, rng = make_model(seed=5)
    bg = torch.FloatTensor(scale_ann_input(trained, X[:300])); ev = torch.FloatTensor(scale_ann_input(trained, X[500:560]))
    a = make_deeplift_fn(trained["model"], bg, chunk=7)(ev)
    b = make_deeplift_fn(trained["model"], bg, chunk=60)(ev)
    assert np.abs(a - b).max() < 1e-5
    m_small, used1 = compute_mean_abs_shap(trained, "ann", X[500:560], X_background=X[:300], batch_size=13, backend="native", n_jobs=1)
    m_big, used2 = compute_mean_abs_shap(trained, "ann", X[500:560], X_background=X[:300], batch_size=60, backend="native", n_jobs=4)
    assert used1 == used2 == "deep-native"          # n_jobs không chuyển sang đường tiến trình song song khi backend native
    assert np.allclose(m_small, m_big, rtol=1e-4)
    """)


def test_unsupported_architecture_raises_instead_of_silently_falling_back():
    _run("""
    import torch.nn as nn
    from src.explainability.deeplift_native import make_deeplift_fn
    bad = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
    try:
        make_deeplift_fn(bad if False else type("W", (), {"network": bad, "eval": lambda self: None})(), torch.zeros(5, 4))
    except ValueError as e:
        assert "Tanh" in str(e)
    else:
        raise AssertionError("phải báo lỗi với lớp chưa hỗ trợ")
    """)
