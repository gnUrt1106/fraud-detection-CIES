"""
train.py — Build và train 5 models cho fraud detection.

RÀNG BUỘC QUAN TRỌNG (spec mục 5.2):
    CatBoost PHẢI dùng chung bộ feature đã encode sẵn — KHÔNG dùng
    cơ chế Ordered Target Statistics nội tại (KHÔNG truyền cat_features).

Nếu CatBoost dùng encoding riêng, sự khác biệt CIES giữa CatBoost và
các model khác không tách bạch được — phá vỡ RQ3.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.config import SEED


# ===== ANN Definition =====

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
        lr_params = {
            "max_iter": 1000,
            "random_state": seed,
            "class_weight": class_weights,
            "solver": "lbfgs",
        }
        if params:
            lr_params.update(params)
        return LogisticRegression(**lr_params)

    elif model_name == "random_forest":
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
        xgb_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": seed,
            "eval_metric": "aucpr",
            "use_label_encoder": False,
            "n_jobs": -1,
        }
        if class_weights is not None:
            xgb_params["scale_pos_weight"] = (
                class_weights.get(0, 1.0) / class_weights.get(1, 1.0)
                if class_weights.get(1, 1.0) > 0
                else 1.0
            )
        if params:
            xgb_params.update(params)
        return XGBClassifier(**xgb_params)

    elif model_name == "catboost":
        # RÀNG BUỘC (spec 5.2): KHÔNG truyền cat_features — dùng feature đã encode chung
        cb_params = {
            "iterations": 200,
            "depth": 6,
            "learning_rate": 0.1,
            "random_seed": seed,
            "eval_metric": "PRAUC",
            "verbose": 0,
        }
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

        torch.manual_seed(seed)
        model = FraudDetectorANN(input_dim=input_dim, dropout=dropout)
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
    model = model_dict["model"]
    class_weights = model_dict.get("class_weights")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
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
    # --- ANN ---
    if isinstance(model, dict) and "model" in model:
        ann = model["model"]
        device = model.get("device", torch.device("cpu"))
        ann.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            logits = ann(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    # --- Sklearn / XGBoost / CatBoost ---
    return model.predict_proba(X)[:, 1]
