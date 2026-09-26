"""
resamplers.py — Wrapper cho 5 kỹ thuật xử lý mất cân bằng.

CHÍNH XÁC 5 kỹ thuật — KHÔNG thêm, KHÔNG bớt (spec mục 4.1).

Danh sách CẤM (spec mục 4.2):
    - GAN-based oversampling (SMOTE-GAN, GANified-SMOTE...)
    - Focal loss / cost-sensitive nâng cao
    - Random undersampling đơn thuần

class_weighting là algorithm-level — KHÔNG resample dữ liệu,
truyền class_weight vào model.fit.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Sequence

import imblearn
import sklearn
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN
from sklearn.utils.class_weight import compute_class_weight

from src.config import SEED, ONEHOT_COLS, SNAP_SYNTHETIC_ONEHOT, TARGET_COL

FINGERPRINT_KEY = b"cies_resample_fingerprint"


def get_resampler(technique: str, seed: int = SEED):
    """
    Trả về resampler instance cho 1 kỹ thuật.

    Args:
        technique: Tên kỹ thuật (smote, smote_enn, adasyn, borderline_smote, class_weighting)
        seed: Random seed

    Returns:
        Resampler instance hoặc None (cho class_weighting)

    Raises:
        ValueError: Nếu technique không nằm trong 5 kỹ thuật đã chốt
    """
    resamplers = {
        "smote": SMOTE(random_state=seed),
        "smote_enn": SMOTEENN(random_state=seed),
        "adasyn": ADASYN(random_state=seed),
        "borderline_smote": BorderlineSMOTE(random_state=seed),
        "class_weighting": None,  # Xử lý riêng — không resample
    }

    if technique not in resamplers:
        raise ValueError(
            f"Kỹ thuật '{technique}' không nằm trong 5 kỹ thuật đã chốt. "
            f"Chỉ được dùng: {list(resamplers.keys())}"
        )

    return resamplers[technique]


def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Tính class weights dùng compute_class_weight('balanced').

    Dùng cho kỹ thuật class_weighting — truyền vào model thay vì resample.

    Args:
        y: Array target labels

    Returns:
        Dict mapping class → weight (e.g., {0: 0.51, 1: 96.3})
    """
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def onehot_groups_from_columns(
    columns: Sequence[str],
    onehot_cols: Sequence[str] = tuple(ONEHOT_COLS),
) -> List[List[int]]:
    """
    Chỉ số cột của từng nhóm one-hot, suy ra từ tên cột dạng `<cột gốc>_<category>`
    (do encode_train sinh ra). Nhóm không có cột nào bị bỏ qua (vd. ULB không có one-hot).
    """
    groups = []
    for col in onehot_cols:
        idx = [i for i, name in enumerate(columns) if str(name).startswith(f"{col}_")]
        if idx:
            groups.append(idx)
    return groups


def snap_onehot_groups(X: np.ndarray, onehot_groups: Sequence[Sequence[int]]) -> np.ndarray:
    """
    Ép mỗi nhóm one-hot về đúng 1 category (argmax, hoà thì lấy cột đầu — tất định).
    Áp cho MỌI dòng: dòng thật vốn đã là one-hot hợp lệ nên là no-op, chỉ dòng tổng hợp
    (giá trị phân số do nội suy) bị thay đổi. Nhờ vậy không cần biết dòng nào là tổng hợp
    (SMOTE-ENN xoá bớt dòng nên vị trí biên giữa dòng thật/tổng hợp không còn rõ).
    """
    X = np.array(X, dtype=float, copy=True)
    for cols in onehot_groups:
        cols = list(cols)
        pick = X[:, cols].argmax(axis=1)
        block = np.zeros((len(X), len(cols)))
        block[np.arange(len(X)), pick] = 1.0
        X[:, cols] = block
    return X


def apply_imbalance(
    technique: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
    onehot_groups: Optional[Sequence[Sequence[int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[int, float]]]:
    """
    Áp dụng kỹ thuật xử lý mất cân bằng.

    PHÂN NHÁNH RIÊNG cho class_weighting:
    - 4 kỹ thuật resampling: resample data, trả class_weights=None
    - class_weighting: giữ nguyên data, trả class_weights dict

    Args:
        technique: Tên kỹ thuật
        X: Feature matrix
        y: Target array
        seed: Random seed
        onehot_groups: Chỉ số cột của từng nhóm one-hot (xem onehot_groups_from_columns). Nếu
            có và config.SNAP_SYNTHETIC_ONEHOT=True, dòng tổng hợp được ép về one-hot hợp lệ.
            Không ảnh hưởng class_weighting (không sinh dòng mới).

    Returns:
        X_resampled: Feature matrix sau resample (hoặc giữ nguyên nếu class_weighting)
        y_resampled: Target array sau resample (hoặc giữ nguyên)
        class_weights: Dict class weights (chỉ cho class_weighting, None cho resamplers)
    """
    if technique == "class_weighting":
        # KHÔNG resample — trả nguyên data + class weights
        class_weights = get_class_weights(y)
        return X, y, class_weights

    # 4 kỹ thuật resampling
    resampler = get_resampler(technique, seed=seed)
    X_resampled, y_resampled = resampler.fit_resample(X, y)

    if onehot_groups and SNAP_SYNTHETIC_ONEHOT:
        X_resampled = snap_onehot_groups(X_resampled, onehot_groups)

    return X_resampled, y_resampled, None


def _resample_fingerprint(technique, X, y, seed, onehot_groups) -> str:
    h = hashlib.sha1()
    for part in (technique, seed, SNAP_SYNTHETIC_ONEHOT, onehot_groups, imblearn.__version__, sklearn.__version__,
                 X.shape, X.dtype, y.dtype):
        h.update(repr(part).encode())
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()[:16]


def apply_imbalance_cached(
    technique: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
    onehot_groups: Optional[Sequence[Sequence[int]]] = None,
    cache_dir: Optional[Path] = None,
    feature_names: Optional[Sequence[str]] = None,
    target_col: str = TARGET_COL,
    name_prefix: str = "train_encoded_",
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[int, float]]]:
    """
    `apply_imbalance`, nhưng lưu kết quả resample thành `cache_dir/<name_prefix><kỹ thuật>.parquet`
    (có tên cột, đọc/upload được như 1 dataset bình thường) để model khác dùng lại. Resample chỉ phụ
    thuộc dữ liệu + kỹ thuật + seed (không phụ thuộc model), nên benchmark 5 model × 5 kỹ thuật không
    phải tính lại cùng 1 phép SMOTE-ENN (~2,5 giờ trên Sparkov) cho từng model.

    Metadata của file chứa dấu vân tay của X, y, kỹ thuật, seed, SNAP_SYNTHETIC_ONEHOT, nhóm one-hot
    và version imblearn + scikit-learn (thuật toán láng giềng gần); khác dấu vân tay (dữ liệu hay cấu hình đã đổi) thì tính lại và ghi đè — không
    bao giờ dùng nhầm bản cũ.
    """
    if cache_dir is None or technique == "class_weighting":
        return apply_imbalance(technique, X, y, seed=seed, onehot_groups=onehot_groups)

    import pyarrow as pa
    import pyarrow.parquet as pq

    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(X.shape[1])]
    path = Path(cache_dir) / f"{name_prefix}{technique}.parquet"
    fingerprint = _resample_fingerprint(technique, X, y, seed, onehot_groups)
    if path.exists() and (pq.read_schema(path).metadata or {}).get(FINGERPRINT_KEY) == fingerprint.encode():
        df = pq.read_table(path).to_pandas()
        return df[names].to_numpy(dtype=X.dtype), df[target_col].to_numpy(dtype=y.dtype), None

    X_res, y_res, _ = apply_imbalance(technique, X, y, seed=seed, onehot_groups=onehot_groups)
    df = pd.DataFrame(X_res, columns=names)
    df[target_col] = y_res
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), FINGERPRINT_KEY: fingerprint.encode()})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return X_res, y_res, None
