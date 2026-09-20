"""
cies.py — Thuật toán tính CIES (Credibility Index via Explanation Stability).

CIES đo độ ổn định của SHAP values qua nhiều lần bootstrap resample train.
Sử dụng rank-weighted distance function — phạt nặng hơn khi top features
thay đổi thứ hạng, nhẹ hơn khi features ít quan trọng thay đổi.

Pseudo-code bắt buộc (spec mục 7.1):
    1. Bootstrap resample train (KHÔNG đổi eval set)
    2. Encoding fit lại trên resample (KHÔNG dùng lại encoding cũ)
    3. Imbalance handling
    4. Train model
    5. SHAP trên tập eval CỐ ĐỊNH
    → Lặp N_RUNS lần, tính stability metric

RÀNG BUỘC KHÔNG ĐƯỢC VI PHẠM:
    - df_test_fixed_eval phải CỐ ĐỊNH qua toàn bộ N_RUNS
    - Encoding phải fit lại từ đầu ở mỗi run
    - Ghi log seed của từng run
"""

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from scipy.stats import spearmanr

from src.config import (
    SEED, N_RUNS, TARGET_COL, RESULTS_DIR,
    ONEHOT_COLS, TARGET_ENCODE_COLS,
)
from src.data.encoding import encode_train, encode_test
from src.imbalance.resamplers import apply_imbalance
from src.models.train import build_model, train_model, predict_proba
from src.explainability.shap_utils import compute_shap

logger = logging.getLogger(__name__)


# ===== CIES Computation =====

def compute_rank_weighted_distance(
    ranks_a: np.ndarray,
    ranks_b: np.ndarray,
) -> float:
    """
    Tính rank-weighted distance giữa 2 SHAP ranking vectors.

    Weight = 1/rank — top features có weight cao hơn,
    phạt nặng khi top features thay đổi thứ hạng.

    Args:
        ranks_a: Ranking vector lần A (1-indexed, rank 1 = quan trọng nhất)
        ranks_b: Ranking vector lần B

    Returns:
        Weighted distance (0 = identical rankings, cao hơn = kém ổn định)
    """
    ranks_a = np.asarray(ranks_a, dtype=float)
    ranks_b = np.asarray(ranks_b, dtype=float)
    n = len(ranks_a)

    # Trọng số THEO THỨ HẠNG của từng feature (không theo vị trí cột): feature nào
    # từng lọt top ở ít nhất 1 trong 2 lần chạy (min rank nhỏ) thì thay đổi của nó
    # bị phạt nặng. Bản cũ gán w_i = 1/i theo chỉ số cột i, nên kết quả phụ thuộc
    # vào THỨ TỰ CỘT feature — cùng 1 cú đổi chỗ top-2 cho khoảng cách 0.122 nếu 2
    # feature nằm ở cột 0-1 nhưng 0.030 nếu nằm ở cột 4-5.
    weights = 1.0 / np.minimum(ranks_a, ranks_b)
    weights = weights / weights.sum()  # Normalize

    # Khoảng cách rank có trọng số
    rank_diff = np.abs(ranks_a - ranks_b)
    # Normalize bằng max possible diff
    max_diff = n - 1
    if max_diff > 0:
        normalized_diff = rank_diff / max_diff
    else:
        normalized_diff = np.zeros_like(rank_diff, dtype=float)

    distance = np.sum(weights * normalized_diff)
    return float(distance)


def shap_to_ranks(shap_values: np.ndarray) -> np.ndarray:
    """
    Chuyển SHAP values thành ranking vector.

    Ranking theo absolute SHAP value trung bình qua các samples.
    Rank 1 = feature quan trọng nhất.

    Args:
        shap_values: SHAP matrix, shape (n_samples, n_features)

    Returns:
        Ranking vector, shape (n_features,), 1-indexed
    """
    # Mean absolute SHAP value per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    # Hoà hạng (vd. các feature SHAP = 0 chưa từng được dùng) chia đều hạng trung bình,
    # KHÔNG xếp theo thứ tự cột. Xếp theo cột thì khi tập feature bằng 0 khác nhau giữa
    # 2 run, thứ tự cột quyết định feature nào "hơn" — tạo chênh lệch thứ hạng giả chỉ do
    # cách sắp cột. (Feature bằng 0 ở MỌI run vẫn giữ hạng cố định và không đóng góp
    # khoảng cách; trọng số 1/rank ở đuôi bảng vốn đã rất nhỏ.)
    from scipy.stats import rankdata

    return rankdata(-mean_abs_shap, method="average")


def compute_stability_metric(
    shap_values_runs: List[np.ndarray],
) -> Dict[str, float]:
    """
    Tính CIES score từ danh sách SHAP values qua N runs.

    CIES = 1 - mean(rank_weighted_distance) over all pairs
    Score 1.0 = hoàn toàn ổn định, 0.0 = không ổn định.

    Thêm: Spearman rank correlation trung bình giữa các runs.

    Args:
        shap_values_runs: List of SHAP value arrays, mỗi array shape (n_eval, n_features)

    Returns:
        Dict chứa:
            - cies_score: 1 - mean pairwise rank-weighted distance
            - mean_rank_distance: Mean pairwise rank-weighted distance
            - std_rank_distance: Std of pairwise distances
            - mean_spearman: Mean pairwise Spearman correlation
            - n_runs: Số runs thực tế
    """
    n_runs = len(shap_values_runs)
    if n_runs < 2:
        logger.warning("Cần ít nhất 2 runs để tính CIES. Trả về 0.")
        return {
            "cies_score": 0.0,
            "mean_rank_distance": 0.0,
            "std_rank_distance": 0.0,
            "mean_spearman": 0.0,
            "n_runs": n_runs,
        }

    # Chuyển SHAP values → rankings
    rankings = [shap_to_ranks(sv) for sv in shap_values_runs]

    # Tính pairwise rank-weighted distances
    distances = []
    spearman_corrs = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            dist = compute_rank_weighted_distance(rankings[i], rankings[j])
            distances.append(dist)
            # Spearman correlation giữa mean |SHAP| vectors
            mean_shap_i = np.mean(np.abs(shap_values_runs[i]), axis=0)
            mean_shap_j = np.mean(np.abs(shap_values_runs[j]), axis=0)
            if np.std(mean_shap_i) > 0 and np.std(mean_shap_j) > 0:
                corr, _ = spearmanr(mean_shap_i, mean_shap_j)
                spearman_corrs.append(corr)

    mean_distance = float(np.mean(distances))
    std_distance = float(np.std(distances))
    cies_score = 1.0 - mean_distance  # 1 = ổn định, 0 = không ổn định

    mean_spearman = float(np.mean(spearman_corrs)) if spearman_corrs else 0.0

    return {
        "cies_score": cies_score,
        "mean_rank_distance": mean_distance,
        "std_rank_distance": std_distance,
        "mean_spearman": mean_spearman,
        "n_runs": n_runs,
    }


def compute_feature_level_cies(
    shap_values_runs: List[np.ndarray],
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Tính CIES ở cấp feature — đo stability cho từng feature riêng.

    Chỉ dùng cho dataset chính (Sparkov), không cần cho dataset phụ (ULB).

    Args:
        shap_values_runs: List of SHAP arrays qua N runs
        feature_names: Tên features (optional)

    Returns:
        DataFrame với cột: feature, mean_abs_shap, shap_cv (coefficient of variation),
        mean_rank, rank_std
    """
    n_runs = len(shap_values_runs)
    n_features = shap_values_runs[0].shape[1]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Mean |SHAP| per feature per run
    mean_abs_per_run = np.array([
        np.mean(np.abs(sv), axis=0) for sv in shap_values_runs
    ])  # shape: (n_runs, n_features)

    # Rankings per run
    rankings_per_run = np.array([
        shap_to_ranks(sv) for sv in shap_values_runs
    ])  # shape: (n_runs, n_features)

    results = []
    for f_idx in range(n_features):
        shap_vals = mean_abs_per_run[:, f_idx]
        ranks = rankings_per_run[:, f_idx]

        mean_shap = float(np.mean(shap_vals))
        std_shap = float(np.std(shap_vals))
        cv = std_shap / mean_shap if mean_shap > 0 else float("inf")

        results.append({
            "feature": feature_names[f_idx],
            "mean_abs_shap": mean_shap,
            "shap_std": std_shap,
            "shap_cv": cv,
            "mean_rank": float(np.mean(ranks)),
            "rank_std": float(np.std(ranks)),
        })

    df = pd.DataFrame(results).sort_values("mean_abs_shap", ascending=False)
    return df.reset_index(drop=True)


# ===== CIES Experiment Runner =====

def run_cies_experiment(
    model_name: str,
    imbalance_technique: str,
    df_train: pd.DataFrame,
    df_test_fixed_eval: pd.DataFrame,
    target_col: str = TARGET_COL,
    onehot_cols: Optional[list] = None,
    target_encode_cols: Optional[list] = None,
    n_runs: int = N_RUNS,
    feature_level: bool = False,
) -> Dict[str, Any]:
    """
    Chạy thí nghiệm CIES cho 1 tổ hợp (model × imbalance technique).

    Theo pseudo-code bắt buộc (spec mục 7.1):
        Mỗi run: bootstrap resample → re-encode → resample → train → SHAP trên eval cố định

    Args:
        model_name: Tên model
        imbalance_technique: Tên kỹ thuật imbalance
        df_train: DataFrame train (raw, chưa encode)
        df_test_fixed_eval: DataFrame eval CỐ ĐỊNH (raw, chưa encode)
        target_col: Tên cột target
        onehot_cols: Cột one-hot (mặc định theo config)
        target_encode_cols: Cột target encoding (mặc định theo config)
        n_runs: Số lần lặp
        feature_level: Tính feature-level CIES (chỉ cho Sparkov)

    Returns:
        Dict chứa:
            - cies_metrics: CIES scores (aggregate)
            - feature_cies: DataFrame feature-level (nếu feature_level=True)
            - run_logs: List log mỗi run (seed, metrics, ...)
    """
    if onehot_cols is None:
        onehot_cols = ONEHOT_COLS
    if target_encode_cols is None:
        target_encode_cols = TARGET_ENCODE_COLS

    # Trích xuất categories cố định từ df_train ban đầu để các bootstrap sample đồng nhất
    fixed_categories = {
        col: sorted(df_train[col].dropna().unique().tolist())
        for col in onehot_cols
        if col in df_train.columns
    }

    shap_values_runs = []
    run_logs = []
    last_feature_names = None

    combination_name = f"{model_name}__{imbalance_technique}"
    logger.info(f"=== CIES Experiment: {combination_name} ({n_runs} runs) ===")

    for run_idx in range(n_runs):
        seed = run_idx  # Spec: dùng range(N_RUNS) làm seeds
        logger.info(f"  Run {run_idx + 1}/{n_runs} (seed={seed})")

        try:
            # 1. Bootstrap resample train (KHÔNG đổi df_test_fixed_eval)
            # Bootstrap CÓ dòng trùng. Giữ id dòng gốc làm `groups` để target encoding
            # (K-fold out-of-fold) không tách bản sao của 1 dòng ra 2 fold khác nhau — nếu
            # tách, nhãn của chính dòng validation nằm trong thống kê fold train (rò rỉ).
            # Rút mẫu vị trí giống hệt trước đây (sample() độc lập với nhãn index).
            base = df_train.reset_index(drop=True)
            sampled = base.sample(frac=1.0, replace=True, random_state=seed)
            row_groups = sampled.index.to_numpy()
            train_resample = sampled.reset_index(drop=True)

            # 2. Encoding — FIT LẠI trên resample này
            encoded_train, encoding_maps = encode_train(
                train_resample, target_col,
                onehot_cols=onehot_cols,
                target_encode_cols=target_encode_cols,
                random_state=seed,
                fixed_categories=fixed_categories,
                groups=row_groups,
            )

            # Encode eval set dùng mapping từ resample hiện tại
            encoded_eval = encode_test(
                df_test_fixed_eval, encoding_maps,
                target_encode_cols=target_encode_cols,
            )

            # Tách X, y và đảm bảo thứ tự feature cột khớp tuyệt đối
            y_train = encoded_train[target_col].values
            train_feature_cols = [c for c in encoded_train.columns if c != target_col]
            for col in train_feature_cols:
                if col not in encoded_eval.columns:
                    encoded_eval[col] = 0.0

            X_train = encoded_train[train_feature_cols].values
            X_eval = encoded_eval[train_feature_cols].values
            last_feature_names = train_feature_cols

            # 3. Imbalance handling
            X_res, y_res, class_weights = apply_imbalance(
                imbalance_technique, X_train, y_train, seed=seed
            )

            # 4. Train model
            model = build_model(
                model_name,
                input_dim=X_res.shape[1],
                class_weights=class_weights,
                seed=seed,
            )
            trained_model = train_model(model, X_res, y_res, model_name=model_name)

            # 5. SHAP trên tập eval CỐ ĐỊNH
            X_background = X_train[:min(200, len(X_train))]
            shap_vals = compute_shap(
                trained_model, model_name, X_eval,
                X_background=X_background,
            )
            # Model không học được gì (dự đoán hằng số) → SHAP toàn 0 → mọi thứ hạng
            # hoà nhau → CIES = 1.0 GIẢ (ổn định tuyệt đối vì không có gì để dao
            # động). Loại run này khỏi phép đo thay vì để nó kéo điểm lên.
            if np.allclose(shap_vals, 0.0):
                raise ValueError("SHAP toàn 0 — model không học được (dự đoán hằng số), bỏ run này")
            shap_values_runs.append(shap_vals)

            run_logs.append({
                "run_idx": run_idx,
                "seed": seed,
                "n_train_resampled": len(train_resample),
                "n_train_after_imbalance": len(X_res),
                "n_eval": len(X_eval),
                "n_features": X_eval.shape[1],
                # mean|SHAP| từng feature của run này — đủ để vẽ lại độ ổn định
                # thứ hạng feature qua các run (src/visualization/explain.py)
                # mà không cần lưu nguyên ma trận SHAP (n_eval x n_features).
                "mean_abs_shap": np.mean(np.abs(shap_vals), axis=0).tolist(),
                "status": "success",
            })

        except Exception as e:
            logger.error(f"  Run {run_idx} failed: {e}")
            run_logs.append({
                "run_idx": run_idx,
                "seed": seed,
                "status": "failed",
                "error": str(e),
            })
            continue

    # Tính CIES
    if len(shap_values_runs) < 2:
        logger.error(f"Chỉ có {len(shap_values_runs)} runs thành công, cần ít nhất 2.")
        # Đủ khoá như nhánh thành công — notebook 04/05 truy cập cies_metrics["mean_spearman"]
        # ngay sau mỗi tổ hợp; thiếu khoá thì KeyError làm sập cả vòng lặp 9-25 tổ hợp.
        cies_metrics = {
            "cies_score": 0.0, "mean_rank_distance": 0.0, "std_rank_distance": 0.0,
            "mean_spearman": 0.0, "n_runs": len(shap_values_runs),
        }
    else:
        cies_metrics = compute_stability_metric(shap_values_runs)

    result = {
        "combination": combination_name,
        "model_name": model_name,
        "imbalance_technique": imbalance_technique,
        "cies_metrics": cies_metrics,
        "feature_names": list(last_feature_names) if last_feature_names else None,
        "run_logs": run_logs,
    }

    # Feature-level CIES (chỉ cho Sparkov)
    if feature_level and len(shap_values_runs) >= 2:
        result["feature_cies"] = compute_feature_level_cies(
            shap_values_runs, last_feature_names
        )

    return result


def run_cies_experiment_isolated(
    model_name: str,
    imbalance_technique: str,
    df_train: pd.DataFrame,
    df_test_fixed_eval: pd.DataFrame,
    target_col: str = TARGET_COL,
    onehot_cols: Optional[list] = None,
    target_encode_cols: Optional[list] = None,
    n_runs: int = N_RUNS,
    feature_level: bool = False,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Wrapper chạy run_cies_experiment() trong 1 subprocess riêng biệt.

    BẮT BUỘC dùng hàm này (thay vì gọi run_cies_experiment() trực tiếp)
    khi lặp qua NHIỀU model trong cùng 1 kernel/script (vd. vòng lặp
    `for model_name in [...]` ở notebooks/04_cies_experiment.ipynb) —
    tránh torch (ANN) và xgboost cùng tồn tại trong 1 process, có thể
    segfault hoặc treo (hang) do xung đột OpenMP runtime. Xem
    src/utils/isolation.py để biết chi tiết.

    Tham số giống hệt run_cies_experiment(), cộng thêm `timeout` (giây,
    None = dùng mặc định của run_isolated()).
    """
    from src.utils.isolation import run_isolated, DEFAULT_TIMEOUT_SECONDS

    return run_isolated(
        run_cies_experiment,
        model_name=model_name,
        imbalance_technique=imbalance_technique,
        df_train=df_train,
        df_test_fixed_eval=df_test_fixed_eval,
        target_col=target_col,
        onehot_cols=onehot_cols,
        target_encode_cols=target_encode_cols,
        n_runs=n_runs,
        feature_level=feature_level,
        timeout=timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS,
    )


def save_cies_results(
    results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    filename: str = "cies_results.json",
):
    """
    Lưu kết quả CIES ra file JSON.

    Args:
        results: List kết quả từ run_cies_experiment()
        output_path: Thư mục output (mặc định RESULTS_DIR)
        filename: Tên file
    """
    if output_path is None:
        output_path = RESULTS_DIR

    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename

    # Convert non-serializable types
    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        return str(obj)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, default=_serialize, indent=2, ensure_ascii=False)

    logger.info(f"CIES results saved to: {filepath}")
