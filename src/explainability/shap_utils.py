"""
shap_utils.py — SHAP explainability utilities.

Mapping explainer bắt buộc theo model (spec mục 6.1):
    - Logistic Regression → shap.LinearExplainer (exact)
    - Random Forest       → shap.TreeExplainer (exact)
    - XGBoost             → shap.TreeExplainer (exact)
    - CatBoost            → shap.TreeExplainer (exact)
    - ANN                 → shap.KernelExplainer hoặc shap.DeepExplainer (xấp xỉ)

ANN dùng explainer xấp xỉ — ghi log riêng, không so sánh thô với TreeSHAP.
"""

import numpy as np
import shap
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Mapping model_name → explainer type
EXPLAINER_MAP = {
    "logistic_regression": "linear",
    "random_forest": "tree",
    "xgboost": "tree",
    "catboost": "tree",
    "ann": "kernel",  # Hoặc "deep" — mặc định kernel cho safety
}


def compute_shap(
    model: Any,
    model_name: str,
    X_eval: np.ndarray,
    X_background: Optional[np.ndarray] = None,
    max_background_samples: int = 100,
) -> np.ndarray:
    """
    Tính SHAP values cho tập eval CỐ ĐỊNH.

    Args:
        model: Trained model (sklearn/xgb/catboost hoặc dict cho ANN)
        model_name: Tên model để chọn explainer
        X_eval: Feature matrix cần giải thích (CỐ ĐỊNH qua các CIES runs)
        X_background: Background data cho KernelExplainer (subsample từ train)
        max_background_samples: Số samples tối đa cho background (KernelExplainer)

    Returns:
        SHAP values array, shape (n_samples, n_features)
    """
    explainer_type = EXPLAINER_MAP.get(model_name)

    if explainer_type is None:
        raise ValueError(
            f"Model '{model_name}' không có explainer mapping. "
            f"Chỉ hỗ trợ: {list(EXPLAINER_MAP.keys())}"
        )

    if explainer_type == "linear":
        return _compute_shap_linear(model, X_eval, X_background)
    elif explainer_type == "tree":
        return _compute_shap_tree(model, X_eval)
    elif explainer_type == "kernel":
        return _compute_shap_kernel(
            model, X_eval, X_background, max_background_samples
        )
    elif explainer_type == "deep":
        return _compute_shap_deep(model, X_eval, X_background)
    else:
        raise ValueError(f"Explainer type '{explainer_type}' không hỗ trợ")


def _compute_shap_linear(
    model: Any, X_eval: np.ndarray, X_background: Optional[np.ndarray]
) -> np.ndarray:
    """SHAP cho Logistic Regression — LinearExplainer (exact)."""
    # LR giờ được bọc {"model", "scaler"} (StandardScaler, xem train.py) — cần
    # unwrap và scale X_eval/X_background CÙNG scaler đã fit lúc train, nếu
    # không LinearExplainer sẽ đọc coef_ ở không gian đã scale nhưng nhận input
    # ở không gian gốc, ra SHAP values sai lệch hoàn toàn.
    if isinstance(model, dict) and "scaler" in model:
        scaler = model["scaler"]
        model = model["model"]
        X_eval = scaler.transform(X_eval)
        if X_background is not None:
            X_background = scaler.transform(X_background)

    if X_background is None:
        # Dùng X_eval làm background nếu không có — không lý tưởng nhưng chấp nhận
        X_background = X_eval

    masker = shap.maskers.Independent(X_background)
    explainer = shap.LinearExplainer(model, masker)
    shap_values = explainer.shap_values(X_eval)

    # LinearExplainer có thể trả list (binary classification)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Lấy class 1 (fraud)

    return np.array(shap_values)


def _compute_shap_tree(model: Any, X_eval: np.ndarray) -> np.ndarray:
    """SHAP cho RF/XGBoost/CatBoost — TreeExplainer (exact)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_eval)

    # TreeExplainer trả về khác dạng tuỳ model/version shap cho binary
    # classification — PHẢI luôn quy về đúng shape (n_samples, n_features)
    # ứng với class 1 (fraud), nếu không các bước sau (rank, stability
    # metric) sẽ vỡ shape một cách ÂM THẦM hoặc lỗi khó hiểu:
    #   - list (API cũ, một số version/model): [array_class0, array_class1]
    #   - ndarray 3 chiều (RandomForestClassifier trên shap>=0.45 — đã xác
    #     nhận tái hiện với shap 0.52.0): (n_samples, n_features, n_classes)
    #   - ndarray 2 chiều (XGBoost/CatBoost — binary objective, đã sẵn 1 output):
    #     (n_samples, n_features), dùng thẳng
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Lấy class 1 (fraud)
    else:
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # Lấy class 1 (fraud)

    return np.array(shap_values)


def _compute_shap_kernel(
    model: Any,
    X_eval: np.ndarray,
    X_background: Optional[np.ndarray],
    max_background_samples: int = 100,
) -> np.ndarray:
    """
    SHAP cho ANN — KernelExplainer (XẤP XỈ).

    Log cảnh báo rằng đây là xấp xỉ, không so sánh thô với TreeSHAP.
    """
    logger.warning(
        "ANN dùng KernelExplainer (xấp xỉ) — kết quả SHAP có nhiễu riêng, "
        "không so sánh trực tiếp với TreeSHAP của các model tree-based."
    )

    import torch

    # Extract predict function từ ANN dict
    if isinstance(model, dict) and "model" in model:
        ann = model["model"]
        device = model.get("device", torch.device("cpu"))
        ann.eval()

        def predict_fn(X):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(device)
                logits = ann(X_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
            return probs
    else:
        raise ValueError("ANN model phải là dict từ build_model()")

    # Background data
    if X_background is None:
        X_background = X_eval

    # Subsample background để giảm compute
    if len(X_background) > max_background_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_background), max_background_samples, replace=False)
        X_background = X_background[idx]

    background_summary = shap.kmeans(X_background, min(10, len(X_background)))
    explainer = shap.KernelExplainer(predict_fn, background_summary)
    shap_values = explainer.shap_values(X_eval, nsamples="auto")

    return np.array(shap_values)


def _compute_shap_deep(
    model: Any,
    X_eval: np.ndarray,
    X_background: Optional[np.ndarray],
) -> np.ndarray:
    """SHAP cho ANN — DeepExplainer (thay thế cho KernelExplainer nếu cần)."""
    import torch

    if isinstance(model, dict) and "model" in model:
        ann = model["model"]
        device = model.get("device", torch.device("cpu"))
    else:
        raise ValueError("ANN model phải là dict từ build_model()")

    if X_background is None:
        X_background = X_eval[:100]

    X_bg_tensor = torch.FloatTensor(X_background).to(device)
    X_eval_tensor = torch.FloatTensor(X_eval).to(device)

    explainer = shap.DeepExplainer(ann, X_bg_tensor)
    shap_values = explainer.shap_values(X_eval_tensor)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    return np.array(shap_values)
