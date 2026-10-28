"""
gpu_enn.py — Bước ENN của SMOTE-ENN chạy trên GPU (ứng viên thay cho CPU để rút ngắn cache SMOTE-ENN).

SMOTE-ENN của imblearn = SMOTE(random_state=seed) rồi EditedNearestNeighbours(sampling_strategy="all",
n_neighbors=3, kind_sel="all"). Bước tốn thời gian là ENN: với mọi dòng tìm 3 láng giềng gần nhất (không tính
chính nó) trong toàn bộ tập sau SMOTE (~2,6 triệu dòng trên Sparkov), dòng nào có ÍT NHẤT 1 láng giềng khác
nhãn thì bị xoá. Ở đây SMOTE giữ nguyên trên CPU (nhanh, ~20 giây), chỉ phần tìm láng giềng chuyển sang GPU,
tìm CHÍNH XÁC (brute-force theo từng khối, không xấp xỉ), mặc định float64 để khớp bản CPU của scikit-learn.

Module này KHÔNG thay đổi mã đang dùng: `resamplers.py` vẫn là đường chuẩn. Chỉ dùng kết quả GPU khi
`scripts/gpu_smote_enn.py check` xác nhận trùng khít với bản CPU (xem script đó).

Torch được nạp lười, bên trong hàm: module này không được dùng chung process với xgboost (xem
src/utils/isolation.py); script `gpu_smote_enn.py` không nạp xgboost nên an toàn.
"""

import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DTYPES = ("float64", "float32")


def resolve_device(device: str = "auto"):
    """'auto' → cuda nếu có, nếu không báo lỗi (không âm thầm lùi về CPU). 'cpu' chỉ để kiểm thử."""
    import torch

    if device == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("torch không thấy CUDA (có thể đang cài bản torch CPU). Dùng --device cpu chỉ để thử.")
        return torch.device("cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch không thấy CUDA.")
    return torch.device(device)


def device_description(device) -> str:
    import torch

    if device.type == "cuda":
        return f"cuda: {torch.cuda.get_device_name(device)}"
    return "cpu"


def knn_indices(
    X: np.ndarray,
    k: int,
    device="auto",
    dtype: str = "float64",
    start: int = 0,
    stop: Optional[int] = None,
    q_chunk: int = 1024,
    db_chunk: int = 262_144,
    progress: Optional[Callable[[int, int, float], None]] = None,
) -> np.ndarray:
    """
    Chỉ số k láng giềng gần nhất (Euclid, tìm chính xác) của các dòng X[start:stop] trong TOÀN BỘ X, sắp tăng dần
    theo khoảng cách. Cột 0 thường là chính dòng đó (khoảng cách 0), giống `NearestNeighbors.kneighbors(X)` của
    scikit-learn khi truy vấn đúng tập đã fit. Hoà khoảng cách: thứ tự không xác định (như bản CPU).

    Trả mảng int64 (stop - start, k). Tự giảm q_chunk khi hết VRAM.
    """
    import torch

    if dtype not in DTYPES:
        raise ValueError(f"dtype phải là một trong {DTYPES}")
    dev = resolve_device(device) if not hasattr(device, "type") else device
    tdtype = torch.float64 if dtype == "float64" else torch.float32
    n = X.shape[0]
    stop = n if stop is None else min(stop, n)
    if not (0 <= start < stop):
        raise ValueError("start/stop không hợp lệ")
    if k > n:
        raise ValueError("k lớn hơn số dòng")

    X = np.asarray(X, dtype=np.float64)
    # Dịch gốc về trung bình: không đổi khoảng cách, giảm sai số làm tròn của công thức |a|²+|b|²-2a·b.
    Xt = torch.from_numpy(X - X.mean(axis=0, keepdims=True)).to(device=dev, dtype=tdtype)
    sq = (Xt * Xt).sum(dim=1)

    out = np.empty((stop - start, k), dtype=np.int64)
    t0 = time.time()
    qs = start
    while qs < stop:
        qe = min(qs + q_chunk, stop)
        try:
            Q, qn = Xt[qs:qe], sq[qs:qe]
            best_d = best_i = None
            for ds in range(0, n, db_chunk):
                de = min(ds + db_chunk, n)
                D = torch.mm(Q, Xt[ds:de].T)
                D.mul_(-2.0).add_(qn[:, None]).add_(sq[ds:de][None, :]).clamp_(min=0)
                v, i = torch.topk(D, min(k, de - ds), dim=1, largest=False)
                del D
                i += ds
                if best_d is None:
                    best_d, best_i = v, i
                else:
                    cd, ci = torch.cat([best_d, v], 1), torch.cat([best_i, i], 1)
                    best_d, j = torch.topk(cd, k, dim=1, largest=False)
                    best_i = torch.gather(ci, 1, j)
            out[qs - start:qe - start] = best_i.cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if q_chunk <= 32:
                raise
            q_chunk //= 2
            logger.warning(f"Hết VRAM → giảm q_chunk còn {q_chunk} và thử lại")
            continue
        qs = qe
        if progress is not None:
            progress(qs - start, stop - start, time.time() - t0)
    return out


def enn_keep_indices(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 3,
    **knn_kwargs,
) -> np.ndarray:
    """
    Chỉ số các dòng GIỮ LẠI của EditedNearestNeighbours(sampling_strategy="all", n_neighbors=3, kind_sel="all"),
    đúng thứ tự imblearn trả (gom theo lớp tăng dần, trong lớp tăng dần theo chỉ số) — tương đương `sample_indices_`.
    """
    y = np.asarray(y)
    nn = knn_indices(X, n_neighbors + 1, **knn_kwargs)[:, 1:]  # bỏ láng giềng đầu (chính nó), như imblearn
    keep = (y[nn] == y[:, None]).all(axis=1)
    return np.concatenate([np.flatnonzero((y == c) & keep) for c in np.unique(y)])


def smote_enn_gpu(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    onehot_groups: Optional[Sequence[Sequence[int]]] = None,
    **knn_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SMOTE-ENN như `apply_imbalance("smote_enn")` nhưng ENN trên GPU. Trả (X_res, y_res, chỉ_số_giữ_lại_trong_tập_sau_SMOTE).
    """
    from imblearn.over_sampling import SMOTE

    from src.config import SNAP_SYNTHETIC_ONEHOT
    from src.imbalance.resamplers import snap_onehot_groups

    X_s, y_s = SMOTE(sampling_strategy="auto", random_state=seed).fit_resample(X, y)
    idx = enn_keep_indices(X_s, y_s, **knn_kwargs)
    X_r, y_r = X_s[idx], y_s[idx]
    if onehot_groups and SNAP_SYNTHETIC_ONEHOT:
        X_r = snap_onehot_groups(X_r, onehot_groups)
    return X_r, y_r, idx


def write_resample_cache(
    path: Path,
    X_res: np.ndarray,
    y_res: np.ndarray,
    feature_names: Sequence[str],
    fingerprint: str,
    target_col: str,
) -> None:
    """Ghi file cache đúng định dạng của `apply_imbalance_cached` (parquet + dấu vân tay trong metadata), ghi nguyên tử."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.imbalance.resamplers import FINGERPRINT_KEY

    df = pd.DataFrame(X_res, columns=list(feature_names))
    df[target_col] = y_res
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), FINGERPRINT_KEY: fingerprint.encode()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
