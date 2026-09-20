"""
train.py — Build và train 5 models cho fraud detection.

RÀNG BUỘC QUAN TRỌNG (spec mục 5.2):
    CatBoost PHẢI dùng chung bộ feature đã encode sẵn — KHÔNG dùng
    cơ chế Ordered Target Statistics nội tại (KHÔNG truyền cat_features).

Nếu CatBoost dùng encoding riêng, sự khác biệt CIES giữa CatBoost và
các model khác không tách bạch được — phá vỡ RQ3.
"""

import numpy as np
from typing import Optional, Dict, Any

from src.config import SEED


# ===== Lazy model-library imports =====
# KHÔNG import sklearn/xgboost/catboost/torch ở top-level của module này.
# Lý do: torch và xgboost cùng nạp vào 1 process có thể segfault hoặc treo
# (hang) vô thời hạn do xung đột OpenMP runtime (đã xác nhận tái hiện được
# trên môi trường dev của project — xem src/utils/isolation.py). Vì
# notebook 03/04 lặp qua cả 5 model trong cùng 1 kernel, và mỗi tổ hợp
# (model × technique) được cô lập trong 1 subprocess riêng (run_isolated),
# mỗi thư viện model chỉ nên import bên trong nhánh xử lý model tương ứng
# của build_model() — đảm bảo 1 subprocess chỉ bao giờ nạp ĐÚNG 1 thư viện
# model, không phụ thuộc model nào chạy trước/sau trong vòng lặp.


# ===== ANN Definition (lazy) =====

_FraudDetectorANN = None


def _get_ann_class():
    """Lazy-define và cache class FraudDetectorANN — chỉ import torch khi gọi lần đầu."""
    global _FraudDetectorANN
    if _FraudDetectorANN is None:
        import torch
        import torch.nn as nn

        class FraudDetectorANN(nn.Module):
            """
            Simple feedforward neural network cho fraud detection.

            Architecture: Input → 128 → ReLU → Dropout → 64 → ReLU → Dropout → 32 → ReLU → 1 → Sigmoid
            """

            def __init__(self, input_dim: int, dropout: float = 0.3):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(dropout),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(dropout),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.network(x)

        _FraudDetectorANN = FraudDetectorANN
    return _FraudDetectorANN


_has_gpu_cache: Optional[bool] = None


def _has_gpu() -> bool:
    """
    Kiểm tra có GPU NVIDIA khả dụng không, qua `nvidia-smi` (subprocess).

    CỐ Ý không dùng `torch.cuda.is_available()` ở đây: gọi nó sẽ import torch
    vào process hiện tại, đúng thứ mà toàn bộ cơ chế lazy-import ở đầu file
    này tránh — nếu process đó sau đó (hoặc trước đó) cũng build XGBoost,
    torch + xgboost cùng nạp 1 process có thể segfault/treo do xung đột
    OpenMP (xem src/utils/isolation.py).
    """
    global _has_gpu_cache
    if _has_gpu_cache is None:
        import shutil
        import subprocess

        if shutil.which("nvidia-smi") is None:
            _has_gpu_cache = False
        else:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5,
                )
                _has_gpu_cache = result.returncode == 0 and "GPU" in result.stdout
            except Exception:
                _has_gpu_cache = False
    return _has_gpu_cache


# ===== Model Builder =====

def build_model(
    model_name: str,
    input_dim: Optional[int] = None,
    class_weights: Optional[Dict[int, float]] = None,
    params: Optional[Dict[str, Any]] = None,
    seed: int = SEED,
) -> Any:
    """
    Tạo model instance (chưa train).

    Args:
        model_name: Tên model (logistic_regression, random_forest, xgboost, catboost, ann)
        input_dim: Số features — BẮT BUỘC cho ANN
        class_weights: Dict class weights (từ class_weighting technique). None nếu dùng resampling.
        params: Dict hyperparameters tùy chỉnh (từ Optuna). Nếu None, tự động nạp từ results/best_params.json nếu có.
        seed: Random seed

    Returns:
        Model instance (sklearn/xgb/catboost) hoặc dict chứa PyTorch model + metadata cho ANN
    """
    # Nếu params không truyền trực tiếp, thử nạp từ file best_params.json
    if params is None:
        try:
            from src.models.tune import load_best_params
            params = load_best_params(model_name)
        except Exception:
            params = None

    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        lr_params = {
            "max_iter": 1000,
            "random_state": seed,
            "class_weight": class_weights,
            "solver": "lbfgs",
        }
        if params:
            lr_params.update(params)
        # StandardScaler — feature chưa scale (amt, lat/long, hour, target-encode
        # probability...) chênh lệch scale lớn khiến lbfgs khó hội tụ trong
        # max_iter (ConvergenceWarning). Chỉ LR cần: cây (RF/XGBoost/CatBoost)
        # bất biến với scaling, ANN đã có BatchNorm ngay sau input.
        return {"model": LogisticRegression(**lr_params), "scaler": StandardScaler()}

    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        rf_params = {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 10,
            "random_state": seed,
            "class_weight": class_weights,
            "n_jobs": -1,
        }
        if params:
            rf_params.update(params)
        return RandomForestClassifier(**rf_params)

    elif model_name == "xgboost":
        from xgboost import XGBClassifier

        xgb_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": seed,
            "eval_metric": "aucpr",
            "use_label_encoder": False,
            "n_jobs": -1,
        }
        if _has_gpu():
            xgb_params["device"] = "cuda"
            xgb_params["tree_method"] = "hist"
        if class_weights is not None:
            # scale_pos_weight = trọng số lớp dương / trọng số lớp âm (= n_neg/n_pos
            # với class_weight='balanced'). Từng bị tính NGƯỢC (w0/w1 ≈ 0.0017
            # thay vì ≈ 577), khiến fraud bị GIẢM trọng số: PR-AUC XGBoost +
            # class_weighting trên ULB tụt 0.877 → 0.704.
            xgb_params["scale_pos_weight"] = (
                class_weights.get(1, 1.0) / class_weights.get(0, 1.0)
                if class_weights.get(0, 1.0) > 0
                else 1.0
            )
        if params:
            xgb_params.update(params)
        return XGBClassifier(**xgb_params)

    elif model_name == "catboost":
        from catboost import CatBoostClassifier

        # RÀNG BUỘC (spec 5.2): KHÔNG truyền cat_features — dùng feature đã encode chung
        cb_params = {
            "iterations": 200,
            "depth": 6,
            "learning_rate": 0.1,
            "random_seed": seed,
            "eval_metric": "PRAUC",
            "verbose": 0,
        }
        if _has_gpu():
            cb_params["task_type"] = "GPU"
        if class_weights is not None:
            cb_params["class_weights"] = [
                class_weights.get(0, 1.0),
                class_weights.get(1, 1.0),
            ]
        if params:
            cb_params.update(params)
        # Đảm bảo cat_features không bao giờ xuất hiện
        cb_params.pop("cat_features", None)
        return CatBoostClassifier(**cb_params)

    elif model_name == "ann":
        if input_dim is None:
            raise ValueError("input_dim bắt buộc cho ANN")

        dropout = 0.3
        lr = 1e-3
        epochs = 30
        batch_size = 1024
        if params:
            dropout = params.get("dropout", dropout)
            lr = params.get("lr", lr)
            epochs = params.get("epochs", epochs)
            batch_size = params.get("batch_size", batch_size)

        import torch
        ann_cls = _get_ann_class()
        torch.manual_seed(seed)
        model = ann_cls(input_dim=input_dim, dropout=dropout)
        return {
            "model": model,
            "class_weights": class_weights,
            "seed": seed,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
        }

    else:
        raise ValueError(
            f"Model '{model_name}' không hợp lệ. "
            f"Chọn từ: logistic_regression, random_forest, xgboost, catboost, ann"
        )


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
) -> Any:
    """
    Train model trên dữ liệu đã encode + resample.

    Args:
        model: Model instance từ build_model()
        X_train: Feature matrix (đã encode, đã resample nếu cần)
        y_train: Target array
        model_name: Tên model (cần cho ANN logic)
        epochs: Số epochs cho ANN (None = lấy từ model dict)
        batch_size: Batch size cho ANN (None = lấy từ model dict)
        learning_rate: Learning rate cho ANN (None = lấy từ model dict)

    Returns:
        Trained model
    """
    # --- Logistic Regression (với StandardScaler) ---
    if isinstance(model, dict) and "scaler" in model:
        X_train_scaled = model["scaler"].fit_transform(X_train)
        model["model"].fit(X_train_scaled, y_train)
        return model

    # --- ANN (PyTorch) ---
    if isinstance(model, dict) and "model" in model:
        eff_epochs = epochs if epochs is not None else model.get("epochs", 30)
        eff_batch = batch_size if batch_size is not None else model.get("batch_size", 1024)
        eff_lr = learning_rate if learning_rate is not None else model.get("lr", 1e-3)
        return _train_ann(
            model, X_train, y_train,
            epochs=eff_epochs, batch_size=eff_batch, learning_rate=eff_lr,
        )

    # --- Sklearn / XGBoost / CatBoost ---
    model.fit(X_train, y_train)
    return model


def _train_ann(
    model_dict: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 30,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
) -> Dict:
    """Train PyTorch ANN model."""
    import torch
    import torch.nn as nn

    from sklearn.preprocessing import StandardScaler

    model = model_dict["model"]
    class_weights = model_dict.get("class_weights")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chuẩn hoá input: feature thô có thang cực khác nhau (Sparkov: unix_time ~1e9,
    # city_pop ~1e6, target-encoded ~1e-2; ULB: Time/Amount ~1e3-1e5). Không scale thì
    # lớp Linear đầu nhận giá trị khổng lồ và train gần như hỏng — đo trên Sparkov:
    # PR-AUC 0.16 (không scale) vs 0.74 (có scale). BatchNorm nằm SAU lớp Linear đầu nên
    # không cứu được. Scaler lưu vào model_dict để predict/SHAP dùng lại đúng.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model_dict["input_scaler"] = scaler
    model = model.to(device)

    # Loss function với class weight
    if class_weights is not None:
        pos_weight = torch.tensor(
            [class_weights.get(1, 1.0) / class_weights.get(0, 1.0)],
            dtype=torch.float32,
        ).to(device)
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)

    # Training loop
    model.train()
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    # BatchNorm1d (train mode) báo lỗi nếu batch cuối chỉ có 1 mẫu — xảy ra khi
    # len(dataset) % batch_size == 1 (vd. 513 mẫu, batch 512). Bỏ batch lẻ đó.
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=(len(dataset) % batch_size == 1),
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    model.eval()
    model_dict["model"] = model
    model_dict["device"] = device
    return model_dict


def scale_ann_input(model: Any, X: np.ndarray) -> np.ndarray:
    """Áp scaler đã fit lúc train lên X cho ANN (no-op nếu model chưa có scaler)."""
    scaler = model.get("input_scaler") if isinstance(model, dict) else None
    return scaler.transform(X) if scaler is not None else X


def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Predict probabilities cho class 1 (fraud).

    Wrapper thống nhất cho cả sklearn/xgb/catboost và PyTorch ANN.

    Args:
        model: Trained model
        X: Feature matrix

    Returns:
        Array of fraud probabilities (shape: [n_samples])
    """
    # --- Logistic Regression (với StandardScaler) ---
    if isinstance(model, dict) and "scaler" in model:
        X_scaled = model["scaler"].transform(X)
        return model["model"].predict_proba(X_scaled)[:, 1]

    # --- ANN ---
    if isinstance(model, dict) and "model" in model:
        import torch

        ann = model["model"]
        device = model.get("device", torch.device("cpu"))
        ann.eval()
        X = scale_ann_input(model, X)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            logits = ann(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    # --- Sklearn / XGBoost / CatBoost ---
    return model.predict_proba(X)[:, 1]


def train_and_evaluate_combo(
    model_name: str,
    technique: str,
    X_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = SEED,
    onehot_groups: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Chạy trọn 1 tổ hợp (model × imbalance technique): imbalance → train →
    predict → evaluate trên test set (không resample).

    Hàm top-level, picklable — dùng làm target cho
    src.utils.isolation.run_isolated() để cô lập mỗi tổ hợp trong 1
    subprocess riêng, tránh xung đột torch/xgboost khi lặp qua nhiều
    model trong cùng 1 process (xem notebooks/03_train_models.ipynb).

    Args:
        model_name: Tên model
        technique: Tên kỹ thuật imbalance
        X_train_raw, y_train_raw: Train set đã encode (chưa resample)
        X_test, y_test: Test set đã encode (không resample)
        seed: Random seed
        onehot_groups: Chỉ số cột từng nhóm one-hot (resamplers.onehot_groups_from_columns)

    Returns:
        Dict: {"model", "imbalance_technique", **metrics}
    """
    from src.imbalance.resamplers import apply_imbalance
    from src.evaluation.metrics import evaluate_model

    X_res, y_res, class_weights = apply_imbalance(
        technique, X_train_raw, y_train_raw, seed=seed, onehot_groups=onehot_groups,
    )
    model = build_model(
        model_name=model_name,
        input_dim=X_res.shape[1],
        class_weights=class_weights,
        seed=seed,
    )
    trained_model = train_model(model, X_res, y_res, model_name=model_name)
    y_proba = predict_proba(trained_model, X_test)
    metrics = evaluate_model(y_test, y_proba)
    return {"model": model_name, "imbalance_technique": technique, **metrics}
