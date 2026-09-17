"""
metrics.py — Evaluation metrics cho fraud detection.

Metric chính: PR-AUC (spec mục 8)
    - KHÔNG dùng ROC-AUC làm metric quyết định (có thể báo cáo tham khảo)
    - Báo cáo kèm: Precision@Recall cố định, F1, F2
    - Test set giữ nguyên phân phối gốc — KHÔNG resample test
"""

import numpy as np
from typing import Dict
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)


def evaluate_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    recall_targets: tuple = (0.5, 0.7, 0.8),
) -> Dict[str, float]:
    """
    Đánh giá model với đầy đủ metrics theo spec.

    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities cho class 1 (fraud)
        threshold: Threshold cho binary classification (mặc định 0.5)
        recall_targets: Các mốc Recall cố định để tính Precision@Recall

    Returns:
        Dict chứa tất cả metrics:
            - pr_auc: PR-AUC (metric chính)
            - roc_auc: ROC-AUC (tham khảo, KHÔNG dùng quyết định)
            - f1: F1-score
            - f2: F2-score (coi trọng recall hơn precision)
            - precision_at_recall_X: Precision tại các mốc Recall cố định
    """
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {}

    # === PR-AUC — METRIC CHÍNH ===
    metrics["pr_auc"] = average_precision_score(y_true, y_proba)

    # === ROC-AUC — chỉ tham khảo ===
    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    # === F1, F2 ===
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["f2"] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    # === Precision@Recall cố định ===
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    for target_recall in recall_targets:
        # Tìm precision cao nhất tại recall >= target
        valid_mask = recall_curve >= target_recall
        if valid_mask.any():
            metrics[f"precision_at_recall_{target_recall}"] = float(
                precision_curve[valid_mask].max()
            )
        else:
            metrics[f"precision_at_recall_{target_recall}"] = 0.0

    # === Confusion matrix stats ===
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["false_positives"] = int(fp)
    metrics["true_negatives"] = int(tn)
    metrics["false_negatives"] = int(fn)

    return metrics


def format_metrics_table(
    results: Dict[str, Dict[str, float]],
    primary_metric: str = "pr_auc",
) -> str:
    """
    Format kết quả metrics thành bảng markdown.

    Args:
        results: Dict[combination_name] → Dict[metric_name] → value
        primary_metric: Metric chính để sort (mặc định pr_auc)

    Returns:
        Bảng markdown string
    """
    if not results:
        return "Chưa có kết quả."

    # Sort by primary metric descending
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get(primary_metric, 0),
        reverse=True,
    )

    # Header
    display_metrics = ["pr_auc", "f1", "f2", "roc_auc"]
    header = "| Combination | " + " | ".join(m.upper() for m in display_metrics) + " |"
    separator = "|---|" + "|".join(["---"] * len(display_metrics)) + "|"

    rows = [header, separator]
    for name, m in sorted_results:
        values = " | ".join(f"{m.get(metric, 0):.4f}" for metric in display_metrics)
        rows.append(f"| {name} | {values} |")

    return "\n".join(rows)
