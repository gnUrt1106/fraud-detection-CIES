"""
explain.py — Biểu đồ SHAP và CIES (độ ổn định giải thích).

Nhóm 1 (SHAP, 1 model đã train): beeswarm, dependence.
Nhóm 2 (so sánh giữa model): mean|SHAP| chuẩn hoá, độ đồng thuận thứ hạng.
Nhóm 3 (CIES): heatmap model × kỹ thuật, độ ổn định thứ hạng feature qua các
bootstrap run, ma trận Spearman giữa các run, so sánh 2 dataset, trade-off với PR-AUC.

Dữ liệu CIES lấy từ file JSON do `merge_cies_result()` (hoặc `save_cies_results()`) ghi; các biểu đồ theo run
cần trường `mean_abs_shap` trong `run_logs` (chỉ có ở kết quả chạy SAU khi
`run_cies_experiment` bắt đầu lưu trường này).
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata, spearmanr

sns.set_theme(style="whitegrid", context="notebook")

C_A = "#2f6fed"   # Sparkov
C_B = "#0e9bab"   # ULB
C_ACCENT = "#e4572e"
C_NEUTRAL = "#9aa0a6"


def _finish(fig, save_path: Optional[Path]):
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ===== Nhóm 1: SHAP của 1 model =====

def plot_shap_beeswarm(shap_values, X_eval, feature_names: Sequence[str], title: str = "",
                       max_display: int = 12, save_path: Optional[Path] = None):
    """Beeswarm: mỗi chấm là 1 mẫu; màu = giá trị feature, vị trí = đóng góp SHAP."""
    import shap

    plt.figure(figsize=(9, 0.42 * max_display + 1.8))
    shap.summary_plot(np.asarray(shap_values), np.asarray(X_eval), feature_names=list(feature_names),
                      max_display=max_display, show=False)
    fig = plt.gcf()
    if title:
        fig.axes[0].set_title(title)
    return _finish(fig, save_path)


def plot_shap_dependence(shap_values, X_eval, feature_names: Sequence[str], feature: str,
                         save_path: Optional[Path] = None):
    """Giá trị feature vs SHAP: cho thấy chiều và ngưỡng tác động (vd. tiền lớn → SHAP tăng vọt)."""
    names = list(feature_names)
    j = names.index(feature)
    x = np.asarray(X_eval)[:, j]
    s = np.asarray(shap_values)[:, j]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(x, s, c=s, cmap="coolwarm", s=18, alpha=0.8)
    ax.axhline(0, color=C_NEUTRAL, lw=1)
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP({feature})")
    ax.set_title(f"Dependence: {feature}")
    fig.colorbar(sc, ax=ax, label="SHAP")
    return _finish(fig, save_path)


# ===== Nhóm 2: so sánh giữa model =====

def _share(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / v.sum() * 100


def plot_shap_bar_by_model(mean_abs_by_model: Dict[str, np.ndarray], feature_names: Sequence[str],
                           top_n: int = 10, save_path: Optional[Path] = None):
    """% đóng góp mean|SHAP| của từng feature theo model — các model có 'nhìn' cùng feature không?"""
    names = list(feature_names)
    df = pd.DataFrame({m: _share(v) for m, v in mean_abs_by_model.items()}, index=names)
    top = df.mean(axis=1).sort_values(ascending=False).head(top_n).index
    d = df.loc[top].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.5 * top_n + 1.5))
    d.plot.barh(ax=ax, width=0.8)
    ax.set_xlabel("% tổng mean|SHAP| của model")
    ax.set_title(f"Top {top_n} feature: mức độ quan trọng theo từng model")
    ax.legend(title="Model", fontsize=8)
    return _finish(fig, save_path)


def plot_model_rank_agreement(mean_abs_by_model: Dict[str, np.ndarray], save_path: Optional[Path] = None):
    """Spearman giữa thứ hạng feature của các model — model nào 'đồng ý' với nhau về feature quan trọng."""
    models = list(mean_abs_by_model)
    n = len(models)
    m = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = spearmanr(mean_abs_by_model[models[i]], mean_abs_by_model[models[j]])[0]
            m[i, j] = m[j, i] = r
    fig, ax = plt.subplots(figsize=(0.9 * n + 3, 0.8 * n + 2.2))
    sns.heatmap(pd.DataFrame(m, index=models, columns=models), annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Đồng thuận thứ hạng feature giữa các model (Spearman)")
    return _finish(fig, save_path)


def plot_condition_agreement(
    mean_abs_by_condition: Dict[str, np.ndarray],
    title: str = "Đồng thuận thứ hạng feature",
    save_path: Optional[Path] = None,
):
    """
    Ma trận đồng thuận thứ hạng feature giữa các điều kiện (vd. 5 kỹ thuật imbalance của cùng
    1 model) — dùng rank-weighted distance của CIES (phạt nặng hơn khi TOP feature bất đồng),
    KHÔNG phải Spearman như `plot_model_rank_agreement` (Spearman coi mọi hạng ngang nhau).
    """
    from src.explainability.cies import compute_condition_agreement_matrix

    m = compute_condition_agreement_matrix(mean_abs_by_condition)
    fig, ax = plt.subplots(figsize=(0.9 * len(m) + 3, 0.8 * len(m) + 2.2))
    sns.heatmap(m, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title(title)
    return _finish(fig, save_path)


# ===== Nhóm 3: CIES =====

def cies_results_to_df(results: List[dict]) -> pd.DataFrame:
    """Bảng tóm tắt (bỏ qua các tổ hợp lỗi)."""
    rows = []
    for r in results:
        if "cies_metrics" not in r:
            continue
        m = r["cies_metrics"]
        rows.append({
            "model": r["model_name"], "technique": r["imbalance_technique"],
            "cies_score": m.get("cies_score"), "mean_rank_distance": m.get("mean_rank_distance"),
            "mean_spearman": m.get("mean_spearman"), "n_runs": m.get("n_runs"),
        })
    return pd.DataFrame(rows)


def plot_cies_heatmap(df: pd.DataFrame, metric: str = "cies_score", title: str = "CIES score (cao = giải thích ổn định)",
                      save_path: Optional[Path] = None):
    pt = df.pivot(index="model", columns="technique", values=metric)
    fig, ax = plt.subplots(figsize=(1.6 * pt.shape[1] + 3, 0.8 * pt.shape[0] + 2.2))
    sns.heatmap(pt, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, linewidths=0.5,
                cbar_kws={"label": metric})
    ax.set_title(title)
    ax.set_xlabel("Kỹ thuật imbalance")
    ax.set_ylabel("")
    plt.setp(ax.get_yticklabels(), rotation=0)
    return _finish(fig, save_path)


def _runs_matrix(result: dict):
    """(n_runs_ok × n_features) mean|SHAP| và tên feature; None nếu kết quả cũ không có dữ liệu theo run."""
    rows = [r["mean_abs_shap"] for r in result.get("run_logs", []) if r.get("status") == "success" and "mean_abs_shap" in r]
    if len(rows) < 2:
        return None, None
    names = result.get("feature_names") or [f"f{i}" for i in range(len(rows[0]))]
    return np.asarray(rows, dtype=float), list(names)


def _ranks(mat: np.ndarray) -> np.ndarray:
    return np.vstack([rankdata(-row, method="average") for row in mat])


def plot_rank_stability(result: dict, top_k: int = 10, save_path: Optional[Path] = None):
    """Phân phối THỨ HẠNG mỗi feature qua các bootstrap run (rank 1 = quan trọng nhất). Hộp hẹp = ổn định."""
    mat, names = _runs_matrix(result)
    if mat is None:
        raise ValueError("Kết quả không có mean_abs_shap theo run — chạy lại CIES bằng code mới.")
    ranks = _ranks(mat)
    med = np.median(ranks, axis=0)
    order = np.argsort(med)[:top_k]
    data = pd.DataFrame(ranks[:, order], columns=[names[i] for i in order]).melt(var_name="feature", value_name="rank")
    fig, ax = plt.subplots(figsize=(9, 0.5 * top_k + 1.8))
    sns.boxplot(data=data, y="feature", x="rank", orient="h", color="#cfe0ff", fliersize=0, ax=ax)
    sns.stripplot(data=data, y="feature", x="rank", orient="h", color=C_A, size=4, alpha=0.6, ax=ax)
    ax.invert_xaxis()
    ax.set_xlabel("Thứ hạng qua các run (1 = quan trọng nhất; trục đảo, phải = tốt)")
    ax.set_ylabel("")
    ax.set_title(f"Độ ổn định thứ hạng — {result.get('combination', '')}  (CIES {result['cies_metrics']['cies_score']:.3f})")
    return _finish(fig, save_path)


def plot_topk_frequency(result: dict, k: int = 5, save_path: Optional[Path] = None):
    """Tỷ lệ số run mà mỗi feature nằm trong top-k — trực quan hơn 'thứ hạng trung bình'."""
    mat, names = _runs_matrix(result)
    if mat is None:
        raise ValueError("Kết quả không có mean_abs_shap theo run.")
    ranks = _ranks(mat)
    freq = pd.Series((ranks <= k).mean(axis=0) * 100, index=names).sort_values(ascending=False)
    freq = freq[freq > 0].head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(freq) + 1.6))
    ax.barh(freq.index, freq.values, color=[C_ACCENT if v < 100 else C_A for v in freq.values])
    ax.set_xlim(0, 105)
    ax.set_xlabel(f"% run mà feature nằm trong top-{k}")
    ax.set_title(f"Top-{k} có ổn định không? — {result.get('combination', '')}")
    for y, v in enumerate(freq.values):
        ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=8)
    return _finish(fig, save_path)


def plot_pairwise_spearman(result: dict, save_path: Optional[Path] = None):
    """Ma trận Spearman giữa mọi cặp run: vùng nhạt = cặp run cho giải thích khác nhau."""
    mat, _ = _runs_matrix(result)
    if mat is None:
        raise ValueError("Kết quả không có mean_abs_shap theo run.")
    n = len(mat)
    m = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = spearmanr(mat[i], mat[j])[0]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(m, cmap="RdYlGn", vmin=0, vmax=1, ax=ax, square=True, cbar_kws={"label": "Spearman"})
    iu = np.triu_indices(n, 1)
    ax.set_title(f"Spearman giữa các run — {result.get('combination', '')}\n(TB {m[iu].mean():.3f}, thấp nhất {m[iu].min():.3f})")
    ax.set_xlabel("Run")
    ax.set_ylabel("Run")
    return _finish(fig, save_path)


def plot_cies_dataset_compare(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str = "Sparkov", name_b: str = "ULB",
                              save_path: Optional[Path] = None):
    """Dumbbell: cies_score của cùng model × kỹ thuật ở 2 dataset — đường ngắn = giải thích ổn định nhất quán."""
    m = df_a.merge(df_b, on=["model", "technique"], suffixes=(f"_{name_a}", f"_{name_b}"))
    if m.empty:
        raise ValueError("Không có tổ hợp chung giữa 2 dataset.")
    m["label"] = m["model"] + " × " + m["technique"]
    m = m.sort_values(f"cies_score_{name_a}")
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(m) + 1.8))
    y = np.arange(len(m))
    ax.hlines(y, m[f"cies_score_{name_a}"], m[f"cies_score_{name_b}"], color=C_NEUTRAL, lw=2)
    ax.scatter(m[f"cies_score_{name_a}"], y, color=C_A, s=70, label=name_a, zorder=3)
    ax.scatter(m[f"cies_score_{name_b}"], y, color=C_B, s=70, label=name_b, zorder=3)
    ax.set_yticks(y, m["label"])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("CIES score")
    ax.set_title("CIES có nhất quán giữa 2 dataset?")
    ax.legend(loc="lower right")
    return _finish(fig, save_path)


def plot_cies_vs_prauc(df_cies: pd.DataFrame, df_bench: pd.DataFrame, save_path: Optional[Path] = None):
    """Trade-off: hiệu năng (PR-AUC) vs độ tin cậy giải thích (CIES). Góc phải-trên = tốt cả hai."""
    # benchmark (notebook 03) đặt tên cột là "imbalance_technique". Nếu không đổi tên, việc
    # ghép rơi về chỉ theo "model" → tích Descartes: mỗi điểm CIES bị ghép với PR-AUC của
    # MỌI kỹ thuật của model đó.
    df_bench = df_bench.rename(columns={"imbalance_technique": "technique"})
    m = df_cies.merge(df_bench, on=["model", "technique"])
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sns.scatterplot(data=m, x="pr_auc", y="cies_score", hue="model", style="technique", s=140, ax=ax)
    ax.set_xlabel("PR-AUC (hiệu năng)")
    ax.set_ylabel("CIES score (ổn định giải thích)")
    ax.set_title("Hiệu năng vs độ tin cậy giải thích")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    return _finish(fig, save_path)
