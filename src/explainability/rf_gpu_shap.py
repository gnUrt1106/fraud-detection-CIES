"""
rf_gpu_shap.py — TreeSHAP CHÍNH XÁC cho Random Forest (scikit-learn) bằng XGBoost, chạy được trên GPU.

Ý tưởng: SHAP của Random Forest là tổng theo cây (rừng = trung bình xác suất lớp 1 của các cây), và
XGBoost đã có TreeSHAP chính xác trên GPU (GPUTreeShap) — nhanh hơn `shap.TreeExplainer` (đơn luồng) rất nhiều.
Nên ta chuyển TỪNG CÂY của rừng sang định dạng cây XGBoost rồi để XGBoost tính `pred_contribs`:

  * cấu trúc cây, đặc trưng tách giữ nguyên;
  * ngưỡng: sklearn đi trái khi `x <= thr`, XGBoost đi trái khi `x < s` → chọn s = float32 liền sau float32 lớn nhất <= thr,
    nên với MỌI x float32 hai điều kiện tương đương (sklearn cũng ép dữ liệu về float32 trước khi dự đoán);
  * cover của nút = `weighted_n_node_samples` — đúng thứ `shap.TreeExplainer` dùng khi không có dữ liệu nền
    (đã gồm trọng số bootstrap và class_weight);
  * giá trị lá = P(lớp 1 | lá) / số_cây → tổng các cây = predict_proba[:, 1] của rừng (hàm mục tiêu 'reg:squarederror',
    base_score = 0, không có hàm liên kết) — cùng đơn vị (xác suất) với SHAP của shap.TreeExplainer.

Rừng được chia thành các Booster theo NHÓM CÂY (SHAP cộng được theo cây) để giới hạn bộ nhớ GPU; hết VRAM thì tự
giảm cỡ nhóm. Kết quả khớp shap.TreeExplainer đến sai số float32 (có test: tests/test_rf_gpu_shap.py).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
import time
from typing import Any, Callable, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_INT32_MAX = 2147483647


def float32_strict_upper(thr: np.ndarray) -> np.ndarray:
    """
    Với ngưỡng sklearn `thr` (float64, đi trái khi x <= thr) trả s (float32) sao cho với mọi x float32:
        (x <= thr)  <=>  (x < s)
    s = float32 liền sau float32 lớn nhất không vượt quá thr.
    """
    thr = np.asarray(thr, dtype=np.float64)
    t32 = thr.astype(np.float32)
    # astype làm tròn gần nhất: nếu làm tròn LÊN quá thr thì lùi một bậc để a32 <= thr
    a32 = np.where(t32.astype(np.float64) > thr, np.nextafter(t32, np.float32(-np.inf)), t32).astype(np.float32)
    return np.nextafter(a32, np.float32(np.inf)).astype(np.float32)


def _nvidia_gpu_visible() -> bool:
    """Có GPU NVIDIA nhìn thấy được không (nvidia-smi -L) — XGBoost KHÔNG tự báo lỗi mà lặng lẽ lùi về CPU."""
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
        return r.returncode == 0 and "GPU" in r.stdout
    except Exception:
        return False


def resolve_device(device: str = "auto") -> str:
    """
    "auto" → "cuda" nếu máy có GPU NVIDIA và bản xgboost có CUDA, ngược lại "cpu".
    Tôn trọng --force-cpu của run_rerun (nó đặt CUDA_VISIBLE_DEVICES=-1) và KHÔNG import torch
    (torch và xgboost không được chung tiến trình ở dự án này).
    """
    if device != "auto":
        return device
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "-1" or not _nvidia_gpu_visible():
        return "cpu"
    try:
        import xgboost as xgb

        if not xgb.build_info().get("USE_CUDA"):
            return "cpu"
        rng = np.random.RandomState(0)
        X = rng.rand(64, 3).astype(np.float32)
        booster = xgb.train({"max_depth": 2, "device": "cpu", "verbosity": 0, "nthread": 1},
                            xgb.DMatrix(X, label=rng.rand(64)), num_boost_round=2)
        booster.set_param({"device": "cuda"})
        booster.predict(xgb.DMatrix(X), pred_contribs=True)
        return "cuda"
    except Exception as exc:
        logger.info(f"XGBoost không dùng được GPU ({str(exc).splitlines()[0][:120]}) — dùng CPU")
        return "cpu"


def _make_template(n_features: int, n_trees: int) -> dict:
    """JSON mô hình XGBoost nhỏ do CHÍNH phiên bản xgboost đang cài sinh ra — để mọi trường bắt buộc đúng định dạng."""
    import xgboost as xgb

    rng = np.random.RandomState(0)
    X = rng.rand(256, n_features).astype(np.float32)
    y = rng.rand(256)
    params = {
        "objective": "reg:squarederror", "max_depth": 1, "base_score": 0.0, "eta": 1.0, "min_child_weight": 0,
        "tree_method": "hist", "device": "cpu", "nthread": 1, "lambda": 0.0, "verbosity": 0,
    }
    booster = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=n_trees)
    tpl = json.loads(bytes(booster.save_raw(raw_format="json")).decode("utf-8"))
    got = len(tpl["learner"]["gradient_booster"]["model"]["trees"])
    if got != n_trees:
        raise RuntimeError(f"Mẫu XGBoost có {got} cây, cần {n_trees}")
    return tpl


def _tree_json(template_tree: dict, tree: Any, n_total_trees: int, tree_id: int, n_features: int) -> dict:
    """Một cây sklearn (`estimator.tree_`) → một cây XGBoost (dict JSON), theo khung của `template_tree`."""
    left = np.asarray(tree.children_left, dtype=np.int64)
    right = np.asarray(tree.children_right, dtype=np.int64)
    n = int(tree.node_count)
    internal = left != -1
    value = np.asarray(tree.value, dtype=np.float64)[:, 0, :]
    if value.shape[1] != 2:
        raise ValueError(f"Cần bài toán 2 lớp, cây có {value.shape[1]} lớp")
    p1 = value[:, 1] / value.sum(axis=1)
    leaf_val = (p1 / n_total_trees).astype(np.float32)

    split_cond = leaf_val.copy()
    split_cond[internal] = float32_strict_upper(np.asarray(tree.threshold, dtype=np.float64)[internal])
    parents = np.full(n, _INT32_MAX, dtype=np.int64)
    ids = np.nonzero(internal)[0]
    parents[left[internal]] = ids
    parents[right[internal]] = ids
    split_idx = np.where(internal, np.asarray(tree.feature, dtype=np.int64), 0)
    if split_idx.max(initial=0) >= n_features:
        raise ValueError("Chỉ số đặc trưng vượt số đặc trưng")
    cover = np.asarray(tree.weighted_n_node_samples, dtype=np.float64).astype(np.float32)

    t = copy.deepcopy(template_tree)
    t["id"] = int(tree_id)
    t["tree_param"]["num_nodes"] = str(n)
    t["tree_param"]["num_deleted"] = "0"
    t["tree_param"]["num_feature"] = str(n_features)
    t["base_weights"] = np.where(internal, np.float32(0.0), leaf_val).astype(np.float32).tolist()
    t["default_left"] = [0] * n
    t["left_children"] = left.tolist()
    t["right_children"] = right.tolist()
    t["loss_changes"] = [0.0] * n
    t["parents"] = parents.tolist()
    t["split_conditions"] = split_cond.tolist()
    t["split_indices"] = split_idx.tolist()
    t["split_type"] = [0] * n
    t["sum_hessian"] = cover.tolist()
    for key in ("categories", "categories_nodes", "categories_segments", "categories_sizes"):
        if key in t:
            t[key] = []
    return t


def rf_to_boosters(rf: Any, chunk_trees: int = 50) -> List[Any]:
    """Chia rừng sklearn thành các xgboost.Booster (mỗi cái `chunk_trees` cây), giá trị lá đã chia cho TỔNG số cây."""
    import xgboost as xgb

    classes = list(getattr(rf, "classes_", [0, 1]))
    if classes != [0, 1]:
        raise ValueError(f"Chỉ hỗ trợ lớp [0, 1], rừng có {classes}")
    estimators = list(rf.estimators_)
    total = len(estimators)
    n_features = int(rf.n_features_in_)
    chunk_trees = max(1, int(chunk_trees))
    boosters: List[Any] = []
    for start in range(0, total, chunk_trees):
        group = estimators[start:start + chunk_trees]
        tpl = _make_template(n_features, len(group))
        model = tpl["learner"]["gradient_booster"]["model"]
        proto = model["trees"][0]
        model["trees"] = [
            _tree_json(proto, est.tree_, total, i, n_features) for i, est in enumerate(group)  # id = vị trí TRONG Booster
        ]
        raw = json.dumps(tpl).encode("utf-8")
        booster = xgb.Booster()
        booster.load_model(bytearray(raw))
        boosters.append(booster)
    return boosters


class RFXgbShap:
    """
    SHAP (lớp 1, đơn vị xác suất) của một RandomForestClassifier, tính bằng XGBoost.
    `__call__(X)` trả ma trận có dấu (n, n_features) float64 — giống `shap.TreeExplainer(rf).shap_values(X)[..., 1]`.
    """

    def __init__(self, rf: Any, device: str = "cuda", chunk_trees: int = 50, row_batch: int = 20_000):
        self._rf = rf
        self.device = resolve_device(device)
        self.chunk_trees = chunk_trees
        self.row_batch = max(1, int(row_batch))
        self.n_features = int(rf.n_features_in_)
        self.n_trees = len(rf.estimators_)
        t = time.time()
        self.boosters = rf_to_boosters(rf, chunk_trees)
        self.seconds_convert = time.time() - t
        self._apply_device()

    def _apply_device(self) -> None:
        for b in self.boosters:
            b.set_param({"device": self.device})

    def _shrink(self) -> bool:
        """Hết VRAM: giảm nhóm cây (và đợt dòng) một nửa, dựng lại Booster. False nếu không giảm được nữa."""
        if self.chunk_trees <= 5 and self.row_batch <= 2000:
            return False
        self.chunk_trees = max(5, self.chunk_trees // 2)
        self.row_batch = max(2000, self.row_batch // 2)
        logger.warning(f"Hết bộ nhớ GPU — giảm còn {self.chunk_trees} cây/nhóm, {self.row_batch} dòng/đợt")
        self.boosters = rf_to_boosters(self._rf, self.chunk_trees)
        self._apply_device()
        return True

    def _contribs(self, X32: np.ndarray) -> np.ndarray:
        import xgboost as xgb

        dm = xgb.DMatrix(X32)
        acc = None
        for b in self.boosters:
            part = b.predict(dm, pred_contribs=True, validate_features=False)[:, :-1]
            acc = part.astype(np.float64) if acc is None else acc + part
        return acc

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X32 = np.ascontiguousarray(np.asarray(X), dtype=np.float32)
        out = np.empty((len(X32), self.n_features), dtype=np.float64)
        start = 0
        while start < len(X32):
            stop = min(start + self.row_batch, len(X32))
            try:
                out[start:stop] = self._contribs(X32[start:stop])
            except Exception as exc:  # xgboost.core.XGBoostError, CUDA OOM...
                msg = str(exc).lower()
                if self.device != "cpu" and ("memory" in msg or "alloc" in msg) and self._shrink():
                    continue
                raise
            start = stop
        return out

    def bias(self, X: np.ndarray) -> np.ndarray:
        """Cột bias (giá trị kỳ vọng) — chỉ để kiểm tra: bias + tổng SHAP = predict_proba[:, 1]."""
        import xgboost as xgb

        X32 = np.ascontiguousarray(np.asarray(X), dtype=np.float32)
        dm = xgb.DMatrix(X32)
        total = None
        for b in self.boosters:
            col = b.predict(dm, pred_contribs=True, validate_features=False)[:, -1].astype(np.float64)
            total = col if total is None else total + col
        return total


def make_rf_gpu_shap_fn(rf: Any, device: str = "auto", chunk_trees: int = 50,
                        row_batch: int = 20_000) -> Tuple[Callable[[np.ndarray], np.ndarray], str]:
    """(hàm_đợt, tên_explainer) cho shap_utils: hàm_đợt(X) → SHAP có dấu (n, n_features).
    Tên explainer ghi trong kết quả: "tree-xgb-gpu" hoặc "tree-xgb-cpu" (theo thiết bị thực sự dùng)."""
    fn = RFXgbShap(rf, device=device, chunk_trees=chunk_trees, row_batch=row_batch)
    logger.info(f"SHAP Random Forest qua XGBoost trên {fn.device}: {fn.n_trees} cây → {len(fn.boosters)} nhóm "
                f"(chuyển đổi {fn.seconds_convert:.1f}s)")
    return fn, ("tree-xgb-cpu" if fn.device == "cpu" else "tree-xgb-gpu")
