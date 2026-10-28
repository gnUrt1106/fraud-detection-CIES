"""
shap_utils.py — SHAP explainability utilities.

Mapping explainer theo model:
    - Logistic Regression → shap.LinearExplainer (exact)
    - Random Forest       → shap.TreeExplainer (exact)
    - XGBoost             → shap.TreeExplainer (exact)
    - CatBoost            → shap.TreeExplainer (exact)
    - ANN                 → shap.DeepExplainer (xấp xỉ, tất định); KernelExplainer làm fallback

ANN dùng explainer xấp xỉ — ghi log riêng, không so sánh thô với TreeSHAP.
"""

import os
import numpy as np
import shap
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


# Mapping model_name → explainer type
EXPLAINER_MAP = {
    "logistic_regression": "linear",
    "random_forest": "tree",
    "xgboost": "tree",
    "catboost": "tree",
    # DeepExplainer thay KernelExplainer: đo trên ANN thật, Kernel chạy lặp 2 lần cùng
    # model chỉ khớp nhau Spearman ~0.85 (nhiễu lấy mẫu riêng — sẽ làm điểm CIES của ANN
    # thấp GIẢ TẠO vì CIES đo đúng độ dao động thứ hạng), chậm hơn ~5x, còn Deep
    # tất định, cộng dồn đúng (sai số 0) và cùng đơn vị log-odds với LR/XGBoost.
    "ann": "deep",
}


def compute_shap(
    model: Any,
    model_name: str,
    X_eval: np.ndarray,
    X_background: Optional[np.ndarray] = None,
    return_explainer: bool = False,
) -> Any:
    """
    Tính SHAP values cho tập eval CỐ ĐỊNH.

    Args:
        model: Trained model (sklearn/xgb/catboost hoặc dict cho ANN)
        model_name: Tên model để chọn explainer
        X_eval: Feature matrix cần giải thích (CỐ ĐỊNH qua các CIES runs)
        X_background: Dữ liệu nền — mốc so sánh của SHAP. BẮT BUỘC cho LR (LinearExplainer) và ANN
            (DeepExplainer); TreeExplainer không dùng. Không tự lấy X_eval làm nền: mốc sai mà không báo gì.
        return_explainer: True → trả thêm tên explainer THỰC SỰ đã dùng ("linear" | "tree" |
            "deep" | "kernel"). ANN có thể lùi từ "deep" về "kernel" khi DeepExplainer lỗi — CIES
            ghi lại tên này để biết điểm của ANN tính bằng explainer nào (Kernel có nhiễu lấy mẫu
            riêng, làm CIES thấp giả tạo).

    Returns:
        SHAP values array, shape (n_samples, n_features); hoặc (values, tên explainer) nếu
        return_explainer=True
    """
    explainer_type = EXPLAINER_MAP.get(model_name)

    if explainer_type is None:
        raise ValueError(
            f"Model '{model_name}' không có explainer mapping. "
            f"Chỉ hỗ trợ: {list(EXPLAINER_MAP.keys())}"
        )

    if explainer_type in ("linear", "deep") and X_background is None:
        raise ValueError(f"'{model_name}' cần X_background (dữ liệu nền SHAP) — không tự lấy X_eval làm nền")

    used = explainer_type
    if explainer_type == "linear":
        values = _compute_shap_linear(model, X_eval, X_background)
    elif explainer_type == "tree":
        values = _compute_shap_tree(model, X_eval)
    else:  # "deep"
        values, used = _compute_shap_deep(model, X_eval, X_background)
    return (values, used) if return_explainer else values


def _compute_shap_linear(model: Any, X_eval: np.ndarray, X_background: np.ndarray) -> np.ndarray:
    """SHAP cho Logistic Regression — LinearExplainer (exact)."""
    from src.models.train import model_kind

    # LR giờ được bọc {"kind": "lr", "model", "scaler"} (StandardScaler, xem train.py) — cần
    # unwrap và scale X_eval/X_background CÙNG scaler đã fit lúc train, nếu
    # không LinearExplainer sẽ đọc coef_ ở không gian đã scale nhưng nhận input
    # ở không gian gốc, ra SHAP values sai lệch hoàn toàn.
    if model_kind(model) == "lr":
        scaler = model["scaler"]
        model = model["model"]
        X_eval = scaler.transform(X_eval)
        X_background = scaler.transform(X_background)

    # max_samples = toàn bộ nền: mặc định Independent chỉ giữ tối đa 100 dòng (tự lấy mẫu con), nên nền
    # lớn hơn bị cắt âm thầm và trung bình nền — mốc của SHAP tuyến tính — đổi theo mẫu con.
    masker = shap.maskers.Independent(X_background, max_samples=len(X_background))
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
    X_background: np.ndarray,
    max_background_samples: int = 100,
) -> np.ndarray:
    """
    SHAP cho ANN — KernelExplainer (XẤP XỈ), chỉ dùng khi DeepExplainer lỗi.

    Log cảnh báo rằng đây là xấp xỉ, không so sánh thô với TreeSHAP.
    """
    logger.warning(
        "ANN dùng KernelExplainer (xấp xỉ) — kết quả SHAP có nhiễu riêng, "
        "không so sánh trực tiếp với TreeSHAP của các model tree-based."
    )

    import torch
    from src.models.train import model_kind

    # Extract predict function từ ANN dict
    if model_kind(model) != "ann":
        raise ValueError("ANN model phải là dict từ build_model()")
    ann = model["model"]
    device = model.get("device", torch.device("cpu"))
    ann.eval()

    def predict_fn(X):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            logits = ann(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    # ANN được train trên input đã scale (xem train.py::_train_ann) — giải thích trong
    # không gian đã scale, giống LR.
    from src.models.train import scale_ann_input

    X_eval = scale_ann_input(model, X_eval)
    X_background = scale_ann_input(model, X_background)

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
    X_background: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """
    SHAP cho ANN — DeepExplainer (xấp xỉ dựa trên DeepLIFT, tất định). Trả (values, "deep"), hoặc
    (values, "kernel") nếu đã phải lùi về KernelExplainer.

    Giải thích logit (đầu ra trước sigmoid) — cùng thang log-odds với LR/XGBoost.
    Nếu DeepExplainer lỗi (vd. version shap/torch không hỗ trợ 1 layer), tự lùi về
    KernelExplainer và ghi log cảnh báo thay vì làm hỏng cả run CIES.
    """
    import torch
    from src.models.train import model_kind

    if model_kind(model) != "ann":
        raise ValueError("ANN model phải là dict từ build_model()")
    ann = model["model"]
    device = model.get("device", torch.device("cpu"))
    ann.eval()  # BatchNorm/Dropout phải ở chế độ eval, nếu không SHAP không tất định

    from src.models.train import scale_ann_input

    try:
        X_bg_tensor = torch.FloatTensor(np.asarray(scale_ann_input(model, X_background))).to(device)
        X_eval_tensor = torch.FloatTensor(np.asarray(scale_ann_input(model, X_eval))).to(device)
        explainer = shap.DeepExplainer(ann, X_bg_tensor)
        shap_values = explainer.shap_values(X_eval_tensor)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"DeepExplainer lỗi ({type(e).__name__}: {e}) — lùi về KernelExplainer.")
        return _compute_shap_kernel(model, X_eval, X_background), "kernel"

    # shap>=0.4x trả ndarray (n_samples, n_features, n_outputs=1) thay vì list;
    # bản cũ trả list [array]. Luôn quy về (n_samples, n_features).
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    return shap_values, "deep"


# ===== mean|SHAP| trên tập LỚN (chia đợt) — dùng cho chế độ chạy lại theo seed =====
# Chế độ rerun (src/explainability/rerun_stability.py) tính SHAP trên TOÀN BỘ tập test (hàng trăm
# nghìn dòng). CIES chỉ cần mean|SHAP| từng feature, nên không giữ ma trận (n_dòng x n_feature):
# tính từng đợt rồi cộng dồn |SHAP|. Explainer chỉ dựng MỘT lần cho cả tập (dựng lại mỗi đợt thì
# Random Forest/LinearExplainer tốn thời gian vô ích). Kết quả không phụ thuộc kích thước đợt:
# SHAP của mỗi dòng độc lập với dòng khác khi nền cố định.

SHAP_BACKENDS = ("auto", "shap", "native")


def _class1_2d(shap_values: Any) -> np.ndarray:
    """Quy mọi dạng đầu ra SHAP (list / 3 chiều / 2 chiều) về (n_samples, n_features) của class fraud."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def _make_shap_batch_fn(
    model: Any,
    model_name: str,
    X_background: Optional[np.ndarray],
    backend: str,
):
    """
    Dựng explainer MỘT lần, trả (hàm_đợt, tên_explainer). `hàm_đợt(X)` trả SHAP (n, n_features).

    backend:
        "shap"   — shap.TreeExplainer cho mọi cây (giống compute_shap).
        "native" — XGBoost `pred_contribs` / CatBoost `ShapValues`: cùng thuật toán TreeSHAP chính xác
                   (đã so khớp với shap.TreeExplainer trong tests), nhưng chạy bằng mã gốc của thư viện
                   (đa luồng, XGBoost chạy được trên GPU) — nhanh hơn rất nhiều trên tập lớn.
        "auto"   — "native" cho xgboost/catboost, "shap" cho phần còn lại.
    Random Forest: "auto"/"shap" = shap.TreeExplainer (đơn luồng, chậm); "native" = chuyển rừng sang XGBoost rồi dùng
        TreeSHAP chính xác của XGBoost, GPU nếu máy có (src/explainability/rf_gpu_shap.py) — khớp shap.TreeExplainer đến
        sai số float32 (có test). Tên explainer ghi trong kết quả: "tree-xgb-gpu" hoặc "tree-xgb-cpu".
    ANN: "auto"/"shap" = shap.DeepExplainer (lặp từng dòng, rất chậm); "native" = DeepLIFT Rescale theo lô trên GPU
        (src/explainability/deeplift_native.py) — cùng thuật toán, khớp DeepExplainer đến sai số float32 (có test),
        nhanh hơn hàng trăm lần. Tên explainer ghi trong kết quả: "deep-native".
    """
    if backend not in SHAP_BACKENDS:
        raise ValueError(f"backend phải là một trong {SHAP_BACKENDS}, nhận '{backend}'")
    explainer_type = EXPLAINER_MAP.get(model_name)
    if explainer_type is None:
        raise ValueError(
            f"Model '{model_name}' không có explainer mapping. Chỉ hỗ trợ: {list(EXPLAINER_MAP.keys())}"
        )
    if explainer_type in ("linear", "deep") and X_background is None:
        raise ValueError(f"'{model_name}' cần X_background (dữ liệu nền SHAP) — không tự lấy X_eval làm nền")
    if backend == "native" and model_name not in ("xgboost", "catboost", "ann", "random_forest"):
        raise ValueError(f"backend 'native' chỉ hỗ trợ xgboost/catboost/ann/random_forest, không phải '{model_name}'")

    if explainer_type == "linear":
        from src.models.train import model_kind

        scaler = None
        sk_model = model
        if model_kind(model) == "lr":
            scaler = model["scaler"]
            sk_model = model["model"]
        bg = scaler.transform(X_background) if scaler is not None else X_background
        # max_samples = toàn bộ nền (xem _compute_shap_linear): mặc định Independent cắt âm thầm còn 100 dòng
        masker = shap.maskers.Independent(bg, max_samples=len(bg))
        explainer = shap.LinearExplainer(sk_model, masker)

        def linear_fn(X: np.ndarray) -> np.ndarray:
            Xs = scaler.transform(X) if scaler is not None else X
            return _class1_2d(explainer.shap_values(Xs))

        return linear_fn, "linear"

    if explainer_type == "tree":
        use_native = backend == "native" or (backend == "auto" and model_name in ("xgboost", "catboost"))
        if use_native and model_name == "xgboost":
            import xgboost as xgb

            booster = model.get_booster()

            def xgb_fn(X: np.ndarray) -> np.ndarray:
                # Cột cuối của pred_contribs là bias (giá trị nền) — bỏ đi, giống shap_values.
                return booster.predict(xgb.DMatrix(np.asarray(X)), pred_contribs=True)[:, :-1]

            return xgb_fn, "tree-native"
        if use_native and model_name == "catboost":
            from catboost import Pool

            def cb_fn(X: np.ndarray) -> np.ndarray:
                return np.asarray(model.get_feature_importance(Pool(np.asarray(X)), type="ShapValues"))[:, :-1]

            return cb_fn, "tree-native"

        if backend == "native" and model_name == "random_forest":
            # Rừng sklearn → XGBoost (từng cây, ngưỡng, cover giữ nguyên) → TreeSHAP chính xác của XGBoost, GPU nếu có.
            from src.explainability.rf_gpu_shap import make_rf_gpu_shap_fn

            return make_rf_gpu_shap_fn(model, device="auto")

        tree_explainer = shap.TreeExplainer(model)

        def tree_fn(X: np.ndarray) -> np.ndarray:
            return _class1_2d(tree_explainer.shap_values(X))

        return tree_fn, "tree"

    # ANN — DeepExplainer. Khác compute_shap: KHÔNG lùi âm thầm về KernelExplainer (nhiễu lấy mẫu riêng
    # làm CIES thấp giả tạo) — ở chế độ rerun, DeepExplainer lỗi thì dừng và báo lỗi.
    import torch
    from src.models.train import model_kind, scale_ann_input

    if model_kind(model) != "ann":
        raise ValueError("ANN model phải là dict từ build_model()")
    ann = model["model"]
    device = model.get("device", torch.device("cpu"))
    ann.eval()  # BatchNorm/Dropout ở chế độ eval, nếu không SHAP không tất định
    bg_tensor = torch.FloatTensor(np.asarray(scale_ann_input(model, X_background))).to(device)
    if backend == "native":
        from src.explainability.deeplift_native import make_deeplift_fn

        native = make_deeplift_fn(ann, bg_tensor)

        def native_fn(X: np.ndarray) -> np.ndarray:
            return native(torch.FloatTensor(np.asarray(scale_ann_input(model, X))).to(device))

        return native_fn, "deep-native"
    deep = shap.DeepExplainer(ann, bg_tensor)

    def deep_fn(X: np.ndarray) -> np.ndarray:
        x_tensor = torch.FloatTensor(np.asarray(scale_ann_input(model, X))).to(device)
        values = deep.shap_values(x_tensor)
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 0]
        return values

    return deep_fn, "deep"


def _uses_shap_tree_explainer(model_name: str, backend: str) -> bool:
    """True nếu đường tính là shap.TreeExplainer (Random Forest, hoặc xgboost/catboost khi ép backend 'shap')."""
    if EXPLAINER_MAP.get(model_name) != "tree":
        return False
    return not (backend == "native" or (backend == "auto" and model_name in ("xgboost", "catboost")))


def _is_cpu_ann(model: Any) -> bool:
    """True nếu model là dict ANN đang nằm trên CPU (ANN trên GPU không chia tiến trình)."""
    return isinstance(model, dict) and getattr(model.get("device"), "type", "cpu") == "cpu"


def _deep_abs_sum(model: Any, X_background: np.ndarray, X: np.ndarray, torch_threads: int) -> np.ndarray:
    """Tổng |SHAP| theo cột của 1 đợt, bằng shap.DeepExplainer (chạy trong tiến trình con của joblib)."""
    import torch

    torch.set_num_threads(torch_threads)
    shap_fn, _ = _make_shap_batch_fn(model, "ann", X_background, "auto")
    return np.abs(np.asarray(shap_fn(X), dtype=np.float64)).sum(axis=0)


def _tree_abs_sum(model: Any, X: np.ndarray) -> np.ndarray:
    """Tổng |SHAP| theo cột của 1 đợt, bằng shap.TreeExplainer (chạy trong tiến trình con của joblib)."""
    values = _class1_2d(shap.TreeExplainer(model).shap_values(X))
    return np.abs(np.asarray(values, dtype=np.float64)).sum(axis=0)


def compute_mean_abs_shap(
    model: Any,
    model_name: str,
    X_eval: np.ndarray,
    X_background: Optional[np.ndarray] = None,
    batch_size: int = 10_000,
    backend: str = "auto",
    n_jobs: int = 1,
) -> Tuple[np.ndarray, str]:
    """
    mean|SHAP| từng feature trên TOÀN BỘ `X_eval`, tính theo từng đợt `batch_size` dòng.

    n_jobs: chỉ dùng cho đường `shap.TreeExplainer` (Random Forest). TreeSHAP của `shap` chạy ĐƠN LUỒNG và giữ GIL
        (đã đo: dùng luồng không nhanh hơn), nên chia các đợt cho nhiều TIẾN TRÌNH (joblib/loky). Kết quả không đổi
        (tổng |SHAP| cộng được theo đợt, chỉ khác thứ tự cộng số thực). Các explainer khác bỏ qua tham số này
        (XGBoost/CatBoost native đã đa luồng, Logistic Regression vector hoá).
        ANN trên CPU cũng dùng tham số này: DeepExplainer của thư viện shap lặp TỪNG dòng (mỗi dòng một lượt lan truyền
        ngược trên toàn bộ dữ liệu nền) nên bị giới hạn bởi độ trễ, không bởi số lõi; chia các đợt cho nhiều tiến
        trình (mỗi tiến trình ít luồng torch hơn). SHAP từng dòng độc lập nên kết quả giống đến sai số làm tròn
        (thứ tự cộng số thực và số luồng torch khác nhau); ANN trên GPU bỏ qua tham số này.

    Returns:
        (vector mean|SHAP| shape (n_features,), tên explainer thực sự dùng)
    """
    if batch_size < 1:
        raise ValueError(f"batch_size phải >= 1, nhận {batch_size}")
    n = len(X_eval)
    if n == 0:
        raise ValueError("X_eval rỗng")
    if n_jobs != 1 and _uses_shap_tree_explainer(model_name, backend):
        from joblib import Parallel, delayed

        n_workers = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
        # Đợt nhỏ nhất vẫn >= 1 dòng; chia đều cho các tiến trình, mỗi tiến trình dựng explainer 1 lần/đợt.
        bounds = list(range(0, n, batch_size))
        parts = Parallel(n_jobs=n_workers)(
            delayed(_tree_abs_sum)(model, X_eval[a:a + batch_size]) for a in bounds
        )
        return np.sum(parts, axis=0) / n, "tree"

    if n_jobs != 1 and EXPLAINER_MAP.get(model_name) == "deep" and _is_cpu_ann(model) and backend != "native":
        from joblib import Parallel, delayed

        if X_background is None:
            raise ValueError(f"'{model_name}' cần X_background (dữ liệu nền SHAP) — không tự lấy X_eval làm nền")
        n_workers = max(1, min(n_jobs if n_jobs > 0 else (os.cpu_count() or 1), n))
        threads = max(1, (os.cpu_count() or 1) // n_workers)
        # Chia nhỏ hơn số tiến trình (~4 phần mỗi tiến trình) để các tiến trình xong gần cùng lúc.
        rows = max(1, min(batch_size, -(-n // (n_workers * 4))))
        parts = Parallel(n_jobs=n_workers)(
            delayed(_deep_abs_sum)(model, X_background, X_eval[a:a + rows], threads) for a in range(0, n, rows)
        )
        return np.sum(parts, axis=0) / n, "deep"

    shap_fn, used = _make_shap_batch_fn(model, model_name, X_background, backend)

    total = None
    for start in range(0, n, batch_size):
        chunk = X_eval[start:start + batch_size]
        values = np.asarray(shap_fn(chunk), dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != len(chunk):
            raise ValueError(f"SHAP đợt {start}: shape {values.shape}, cần ({len(chunk)}, n_features)")
        part = np.abs(values).sum(axis=0)
        total = part if total is None else total + part
    return total / n, used
