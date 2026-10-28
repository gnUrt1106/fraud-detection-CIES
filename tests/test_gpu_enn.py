"""
Test bước ENN trên GPU (chạy bằng torch CPU, không cần GPU thật): kết quả phải TRÙNG imblearn.

Mọi thứ chạy trong subprocess: torch không được nạp chung process với xgboost (xem src/utils/isolation.py).
"""

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COMMON = """
import numpy as np
def make(n=3000, seed=0, discrete=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12))
    y = (rng.random(n) < 0.06).astype(np.int64)
    X[y == 1] += 0.8                      # lớp thiểu số lẫn một phần vào lớp đa số → ENN có việc để làm
    if discrete:                          # nhiều giá trị rời rạc/trùng lặp như dữ liệu thật (giờ, thứ, one-hot)
        X[:, :4] = np.round(X[:, :4])
    return X, y
"""


def _run(code: str) -> None:
    r = subprocess.run(
        [sys.executable, "-c", COMMON + textwrap.dedent(code)],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    assert r.returncode == 0, (r.stdout[-2000:] + "\n" + r.stderr[-3000:])


def test_enn_keep_indices_matches_imblearn_and_ignores_chunking():
    _run("""
    from imblearn.under_sampling import EditedNearestNeighbours
    from src.imbalance.gpu_enn import enn_keep_indices
    X, y = make()
    enn = EditedNearestNeighbours(sampling_strategy="all", n_neighbors=3, kind_sel="all")
    enn.fit_resample(X, y)
    ref = enn.sample_indices_
    assert 0 < len(ref) < len(y)
    for qc, dc in [(1024, 262144), (97, 501), (3000, 3000)]:
        got = enn_keep_indices(X, y, device="cpu", dtype="float64", q_chunk=qc, db_chunk=dc)
        assert np.array_equal(got, ref), (qc, dc, len(got), len(ref))
    """)


def test_smote_enn_gpu_matches_imblearn_smoteenn():
    _run("""
    from imblearn.combine import SMOTEENN
    from src.imbalance.gpu_enn import smote_enn_gpu
    X, y = make(n=4000, seed=3)
    X_ref, y_ref = SMOTEENN(random_state=7).fit_resample(X, y)
    X_g, y_g, _ = smote_enn_gpu(X, y, seed=7, device="cpu", dtype="float64", q_chunk=500, db_chunk=2000)
    assert X_g.shape == X_ref.shape and np.array_equal(y_g, y_ref)
    assert np.array_equal(X_g, X_ref)
    """)


def test_discrete_features_with_ties_agree_with_imblearn():
    # Dữ liệu có nhiều điểm trùng/hoà khoảng cách: chỉ được lệch ở các điểm hoà khoảng cách khác nhãn (hiếm).
    _run("""
    from imblearn.under_sampling import EditedNearestNeighbours
    from src.imbalance.gpu_enn import enn_keep_indices
    X, y = make(n=6000, seed=5, discrete=True)
    enn = EditedNearestNeighbours(sampling_strategy="all", n_neighbors=3, kind_sel="all")
    enn.fit_resample(X, y)
    got = enn_keep_indices(X, y, device="cpu", dtype="float64")
    diff = len(np.setxor1d(got, enn.sample_indices_))
    assert diff <= 0.002 * len(y), diff
    print("khac nhau:", diff, "/", len(y))
    """)


def test_written_cache_is_accepted_as_cache_hit_by_apply_imbalance_cached(tmp_path):
    _run(f"""
    import pathlib
    import src.imbalance.resamplers as rs
    from src.imbalance.gpu_enn import smote_enn_gpu, write_resample_cache
    from src.config import TARGET_COL
    X, y = make(n=2500, seed=9)
    names = [f"f{{i}}" for i in range(X.shape[1])]
    X_g, y_g, _ = smote_enn_gpu(X, y, seed=11, device="cpu")
    fp = rs._resample_fingerprint("smote_enn", X, y, 11, None)
    d = pathlib.Path({str(tmp_path)!r})
    write_resample_cache(d / "train_encoded_smote_enn.parquet", X_g, y_g, names, fp, TARGET_COL)

    def boom(*a, **k):
        raise AssertionError("cache miss: apply_imbalance bị gọi")
    rs.apply_imbalance = boom
    X_c, y_c, _ = rs.apply_imbalance_cached("smote_enn", X, y, seed=11, cache_dir=d, feature_names=names)
    assert np.array_equal(X_c, X_g) and np.array_equal(y_c, y_g)
    """)
