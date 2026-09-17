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

import numpy as np
from typing import Tuple, Optional, Dict, Any

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN
from sklearn.utils.class_weight import compute_class_weight

from src.config import SEED


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


def apply_imbalance(
    technique: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
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

    return X_resampled, y_resampled, None
