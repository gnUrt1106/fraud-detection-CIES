"""
ANN: SHAP DeepExplainer chia cho nhiều tiến trình (compute_mean_abs_shap(..., n_jobs>1)) phải cho cùng kết quả
với chạy tuần tự. Chạy trong subprocess: torch không được nạp chung process với xgboost.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_parallel_deep_shap_matches_serial():
    code = textwrap.dedent("""
    import numpy as np
    from src.models.train import build_model, train_model
    from src.explainability.shap_utils import compute_mean_abs_shap

    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 8))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=600) * 0.5 > 1).astype(int)
    m = build_model("ann", input_dim=8, params={"lr": 1e-3, "dropout": 0.1, "epochs": 3, "batch_size": 128}, seed=0)
    m = train_model(m, X, y, model_name="ann")
    bg = X[:50]
    ser, e1 = compute_mean_abs_shap(m, "ann", X[:200], X_background=bg)
    for jobs, batch in [(2, 60), (3, 10_000), (-1, 33)]:
        par, e2 = compute_mean_abs_shap(m, "ann", X[:200], X_background=bg, n_jobs=jobs, batch_size=batch)
        assert e1 == e2 == "deep"
        assert np.allclose(ser, par, rtol=1e-4, atol=1e-7), (jobs, ser, par)
    try:
        compute_mean_abs_shap(m, "ann", X[:20], X_background=None, n_jobs=2)
        raise SystemExit("thiếu nền phải báo lỗi")
    except ValueError:
        pass
    """)
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + "\n" + r.stderr[-3000:]
