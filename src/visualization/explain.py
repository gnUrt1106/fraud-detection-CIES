"""
explain.py — Biểu đồ SHAP và CIES (độ ổn định giải thích).

Nhóm 1 (SHAP, 1 model đã train): beeswarm, dependence.
Nhóm 2 (so sánh giữa model): mean|SHAP| chuẩn hoá, độ đồng thuận thứ hạng.
Nhóm 3 (CIES): heatmap model × kỹ thuật, thứ hạng kỹ thuật trong từng model, độ ổn định thứ hạng feature
qua các bootstrap run, CIES theo cỡ mẫu, trade-off với PR-AUC, đồng thuận feature giữa các kỹ thuật.
Nhóm 4 (benchmark): PR-AUC theo kỹ thuật (độ lệch trong model), số báo nhầm ở ngưỡng 0,5.

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

from src.config import MODEL_NAMES, IMBALANCE_TECHNIQUES

sns.set_theme(style="whitegrid", context="notebook")

C_A = "#2f6fed"
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
    np.random.seed(0)  # summary_plot xê dịch chấm ngẫu nhiên — cố định để ảnh giống hệt giữa các lần chạy
    shap.summary_plot(np.asarray(shap_values), np.asarray(X_eval), feature_names=list(feature_names),
                      max_display=max_display, show=False)
    fig = plt.gcf()
    if title:
        fig.axes[0].set_title(title)
    return _finish(fig, save_path)


def plot_shap_dependence(shap_values, X_eval, feature_names: Sequence[str], feature: str,
                         color_by: Optional[str] = None, log_x: bool = False, save_path: Optional[Path] = None):
    """Giá trị feature vs SHAP: chiều và ngưỡng tác động. `color_by` tô màu theo 1 feature khác để thấy
    tương tác (tô theo chính SHAP thì trùng thông tin với trục tung)."""
    names = list(feature_names)
    X = np.asarray(X_eval)
    x = X[:, names.index(feature)]
    s = np.asarray(shap_values)[:, names.index(feature)]
    fig, ax = plt.subplots(figsize=(8, 5))
    if color_by is None:
        ax.scatter(x, s, color=C_A, s=22, alpha=0.7)
    else:
        sc = ax.scatter(x, s, c=X[:, names.index(color_by)], cmap="viridis", s=22, alpha=0.8)
        fig.colorbar(sc, ax=ax, label=color_by)
    if log_x:
        ax.set_xscale("log")
    ax.axhline(0, color=C_NEUTRAL, lw=1)
    ax.set_xlabel(feature + (" (thang log)" if log_x else ""))
    ax.set_ylabel(f"SHAP({feature})")
    ax.set_title(f"SHAP của {feature}" + (f", tô màu theo {color_by}" if color_by else ""))
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
    # Thứ tự theo config (như các file kết quả), không theo bảng chữ cái; tên lạ xếp cuối
    pt = df.pivot(index="model", columns="technique", values=metric)
    order = lambda names, ref: [n for n in ref if n in names] + sorted(n for n in names if n not in ref)
    pt = pt.loc[order(pt.index, MODEL_NAMES), order(pt.columns, IMBALANCE_TECHNIQUES)]
    pt = pt.rename(index=lambda m: LABEL.get(m, m), columns=lambda t: LABEL.get(t, t))
    # Thang màu theo khoảng giá trị THẬT: CIES nằm trong ~0,84–0,97, thang 0–1 tô mọi ô cùng 1 màu
    lo, hi = np.floor(np.nanmin(pt.values) * 100) / 100, np.ceil(np.nanmax(pt.values) * 100) / 100
    fig, ax = plt.subplots(figsize=(1.7 * pt.shape[1] + 3, 0.75 * pt.shape[0] + 2.2))
    sns.heatmap(pt, annot=True, fmt=".3f", cmap="RdYlGn", vmin=lo, vmax=hi, ax=ax, linewidths=0.5,
                annot_kws={"fontsize": 12}, cbar_kws={"label": f"{metric} (thang {lo:.2f}–{hi:.2f})"})
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
    np.random.seed(0)  # stripplot xê dịch chấm ngẫu nhiên — cố định để ảnh giống hệt giữa các lần chạy
    sns.stripplot(data=data, y="feature", x="rank", orient="h", color=C_A, size=4, alpha=0.6, ax=ax)
    ax.invert_xaxis()
    ax.set_xlabel("Thứ hạng qua các run (1 = quan trọng nhất; trục đảo, phải = tốt)")
    ax.set_ylabel("")
    ax.set_title(f"Độ ổn định thứ hạng — {result.get('combination', '')}  (CIES {result['cies_metrics']['cies_score']:.3f})")
    return _finish(fig, save_path)


TECH_MARKERS = {"smote": "o", "smote_enn": "s", "adasyn": "^", "borderline_smote": "D", "class_weighting": "*"}
MODEL_COLORS = dict(zip(MODEL_NAMES, sns.color_palette("tab10", len(MODEL_NAMES))))
LABEL = {"logistic_regression": "LR", "random_forest": "RF", "xgboost": "XGBoost", "catboost": "CatBoost", "ann": "ANN",
         "smote": "SMOTE", "smote_enn": "SMOTE-ENN", "adasyn": "ADASYN", "borderline_smote": "Borderline-SMOTE",
         "class_weighting": "Class weighting"}


def _ordered(names, ref):
    return [n for n in ref if n in set(names)] + sorted(n for n in set(names) if n not in ref)


def _model_tech_legend(fig, models, techniques):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=MODEL_COLORS.get(m, "grey"), marker="o", ls="", ms=9, label=LABEL.get(m, m)) for m in models]
    handles += [Line2D([], [], color="black", marker=TECH_MARKERS.get(t, "o"), ls="", ms=9, mfc="none", label=LABEL.get(t, t))
                for t in techniques]
    # Dưới đáy hình (ngoài vùng vẽ): savefig(bbox_inches="tight") tự nới khung chứa nó, không đè nhãn trục
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))


def plot_cies_vs_prauc(df_cies: pd.DataFrame, df_bench: pd.DataFrame, save_path: Optional[Path] = None,
                       ax=None, title: str = "Hiệu năng vs độ ổn định giải thích", exclude_models: Sequence[str] = ()):
    """Mỗi điểm = 1 tổ hợp model × kỹ thuật: PR-AUC (test) vs CIES. Màu = model, hình = kỹ thuật."""
    # benchmark đặt tên cột "imbalance_technique": phải đổi tên, nếu không ghép chỉ theo "model"
    # → tích Descartes (mỗi điểm CIES bị ghép với PR-AUC của MỌI kỹ thuật của model đó).
    m = df_cies.merge(df_bench.rename(columns={"imbalance_technique": "technique"}), on=["model", "technique"])
    m = m[~m["model"].isin(exclude_models)]
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 6))
    for _, r in m.iterrows():
        star = r["technique"] == "class_weighting"
        ax.scatter(r["pr_auc"], r["cies_score"], color=MODEL_COLORS.get(r["model"], "grey"),
                   marker=TECH_MARKERS.get(r["technique"], "o"), s=260 if star else 110, edgecolor="white", linewidth=0.6, zorder=3)
    rho = spearmanr(m["pr_auc"], m["cies_score"])[0] if len(m) > 2 else float("nan")
    ax.set_xlabel("PR-AUC trên test (hiệu năng)")
    ax.set_ylabel("CIES (ổn định giải thích)")
    ax.set_title(f"{title}\nSpearman(PR-AUC, CIES) = {rho:.2f}")
    if own:
        _model_tech_legend(ax.figure, _ordered(m["model"], MODEL_NAMES), _ordered(m["technique"], IMBALANCE_TECHNIQUES))
        return _finish(ax.figure, save_path)
    return ax


def plot_cies_vs_prauc_by_dataset(panels: Dict[str, tuple], exclude: Optional[Dict[str, Sequence[str]]] = None,
                                  save_path: Optional[Path] = None):
    """1 panel / dataset: {tên: (df_cies, df_bench)}. `exclude` bỏ model khỏi 1 panel (vd. LR Sparkov có PR-AUC
    ~0,1 kéo giãn trục, dồn 20 điểm còn lại vào 1 góc) — ghi rõ trong tiêu đề."""
    exclude = exclude or {}
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 6.2))
    axes = np.atleast_1d(axes)
    models, techs = set(), set()
    for ax, (name, (dc, db)) in zip(axes, panels.items()):
        ex = list(exclude.get(name, ()))
        note = f" (không vẽ {', '.join(LABEL.get(e, e) for e in ex)})" if ex else ""
        plot_cies_vs_prauc(dc, db, ax=ax, title=name + note, exclude_models=ex)
        models |= set(dc["model"]) - set(ex)
        techs |= set(dc["technique"])
    _model_tech_legend(fig, _ordered(models, MODEL_NAMES), _ordered(techs, IMBALANCE_TECHNIQUES))
    return _finish(fig, save_path)


def plot_technique_ranks(df_by_dataset: Dict[str, pd.DataFrame], metric: str = "cies_score",
                         save_path: Optional[Path] = None):
    """Thứ hạng CIES của 5 kỹ thuật TRONG từng model (1 = ổn định nhất, 5 = kém nhất), mỗi dataset 1 panel.
    So thứ hạng thay vì mức CIES: mức CIES phụ thuộc cỡ mẫu nên không so được giữa 2 dataset."""
    fig, axes = plt.subplots(1, len(df_by_dataset), figsize=(7.6 * len(df_by_dataset), 4.8))
    axes = np.atleast_1d(axes)
    for ax, (name, df) in zip(axes, df_by_dataset.items()):
        pt = df.pivot(index="model", columns="technique", values=metric)
        pt = pt.loc[_ordered(pt.index, MODEL_NAMES), _ordered(pt.columns, IMBALANCE_TECHNIQUES)]
        ranks = pt.rank(axis=1, ascending=False, method="min")
        sns.heatmap(ranks, annot=pt.map(lambda v: f"{v:.3f}"), fmt="", cmap="RdYlGn_r", vmin=1, vmax=len(pt.columns),
                    cbar=False, linewidths=1, ax=ax, annot_kws={"fontsize": 11})
        for (i, j), rk in np.ndenumerate(ranks.values):
            ax.text(j + 0.5, i + 0.22, f"hạng {int(rk)}", ha="center", va="center", fontsize=9, color="black", alpha=0.8)
        ax.set_title(f"{name}", fontsize=13)
        ax.set_xticks(np.arange(len(pt.columns)) + 0.5, [LABEL.get(t, t) for t in pt.columns], rotation=20, ha="right", fontsize=11)
        ax.set_yticks(np.arange(len(pt.index)) + 0.5, [LABEL.get(m, m) for m in pt.index], rotation=0, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Kỹ thuật nào cho giải thích ổn định nhất trong từng model? (xanh = hạng 1, đỏ = hạng 5; số = CIES)", fontsize=13)
    return _finish(fig, save_path)


def plot_cies_vs_sample_size(rows: List[dict], dataset_frauds: Optional[Dict[str, int]] = None,
                             save_path: Optional[Path] = None):
    """CIES theo số fraud trong mẫu bootstrap (kiểm tra độ nhạy). Đường dọc = số fraud mà mỗi dataset thực dùng."""
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for (m, t), g in df.groupby(["model", "technique"], sort=False):
        g = g.sort_values("n_fraud")
        ax.plot(g["n_fraud"], g["cies_score"], marker=TECH_MARKERS.get(t, "o"), ms=9, lw=2.2,
                color=MODEL_COLORS.get(m, "grey"), label=f"{LABEL.get(m, m)} × {LABEL.get(t, t)}")
        last = g.iloc[-1]
        ax.annotate(f"{last['cies_score']:.3f}", (last["n_fraud"], last["cies_score"]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=10)
    for name, n in (dataset_frauds or {}).items():
        ax.axvline(n, color=C_NEUTRAL, ls="--", lw=1.2)
        ax.text(n, ax.get_ylim()[0], f" {name}\n {n} fraud", va="bottom", ha="left", fontsize=10, color="dimgray")
    ax.set_xlabel("Số fraud trong mẫu bootstrap")
    ax.set_ylabel("CIES")
    ax.set_title("CIES tăng theo cỡ mẫu → chỉ so CIES trong cùng dataset")
    ax.legend(loc="upper left", fontsize=10)
    return _finish(fig, save_path)


def plot_prauc_by_technique(bench_by_dataset: Dict[str, pd.DataFrame], save_path: Optional[Path] = None):
    """PR-AUC từng kỹ thuật so với trung bình 5 kỹ thuật của CÙNG model (điểm % chênh), mỗi dataset 1 panel.
    Mức PR-AUC giữa các model chênh nhau rất xa (LR Sparkov ~0,1 so với ~0,87) — vẽ độ lệch để thấy tác động
    của kỹ thuật; nhãn trục tung ghi PR-AUC tốt nhất của model."""
    fig, axes = plt.subplots(1, len(bench_by_dataset), figsize=(7.2 * len(bench_by_dataset), 5), sharey=False)
    axes = np.atleast_1d(axes)
    techs = set()
    for ax, (name, b) in zip(axes, bench_by_dataset.items()):
        models = _ordered(b["model"], MODEL_NAMES)[::-1]
        for y, m in enumerate(models):
            g = b[b["model"] == m].set_index("imbalance_technique")["pr_auc"]
            ax.hlines(y, (g - g.mean()).min() * 100, (g - g.mean()).max() * 100, color="lightgrey", lw=6, zorder=1)
            for t, v in g.items():
                techs.add(t)
                ax.scatter((v - g.mean()) * 100, y, marker=TECH_MARKERS.get(t, "o"), s=260 if t == "class_weighting" else 120,
                           color=MODEL_COLORS.get(m, "grey"),
                           edgecolor="black", linewidth=0.6, zorder=3)
        ax.axvline(0, color=C_NEUTRAL, lw=1)
        ax.set_yticks(range(len(models)),
                      [f"{LABEL.get(m, m)}\n(tốt nhất {b[b['model'] == m]['pr_auc'].max():.3f})" for m in models])
        ax.set_xlabel("PR-AUC − trung bình 5 kỹ thuật của model (điểm %)")
        ax.set_title(name, fontsize=13)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color="black", marker=TECH_MARKERS.get(t, "o"), ls="", ms=9, mfc="none", label=LABEL.get(t, t))
                        for t in _ordered(techs, IMBALANCE_TECHNIQUES)], loc="upper center", ncol=5, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Kỹ thuật imbalance kéo PR-AUC lên/xuống bao nhiêu trong từng model?", fontsize=13)
    return _finish(fig, save_path)


def plot_false_positives(bench: pd.DataFrame, title: str = "Sparkov", save_path: Optional[Path] = None):
    """Số giao dịch thật bị báo là fraud ở ngưỡng 0,5 (thang log), theo model × kỹ thuật."""
    models = _ordered(bench["model"], MODEL_NAMES)
    techs = _ordered(bench["imbalance_technique"], IMBALANCE_TECHNIQUES)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    w = 0.8 / len(techs)
    palette = dict(zip(techs, ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", C_ACCENT]))
    for j, t in enumerate(techs):
        vals = [bench[(bench.model == m) & (bench.imbalance_technique == t)]["false_positives"].iloc[0] for m in models]
        ax.bar(np.arange(len(models)) + (j - (len(techs) - 1) / 2) * w, vals, w, label=LABEL.get(t, t), color=palette.get(t))
    ax.set_yscale("log")
    ax.set_xticks(range(len(models)), [LABEL.get(m, m) for m in models], fontsize=11)
    ax.set_ylabel("Số giao dịch thật bị báo nhầm (log)")
    ax.set_title(f"{title}: báo nhầm ở ngưỡng 0,5 — class weighting (cam) cao nhất ở 4/5 model")
    ax.legend(ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    return _finish(fig, save_path)


def plot_technique_agreement_summary(agreement: Dict[str, Dict[str, pd.DataFrame]], save_path: Optional[Path] = None):
    """Tóm tắt ma trận đồng thuận thứ hạng feature giữa 5 kỹ thuật (`compute_condition_agreement_matrix`):
    với mỗi model, khoảng giá trị giữa các kỹ thuật resample với nhau so với giữa class weighting và chúng.
    agreement = {dataset: {model: ma trận 5×5}}."""
    fig, axes = plt.subplots(1, len(agreement), figsize=(7.2 * len(agreement), 4.4), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, by_model) in zip(axes, agreement.items()):
        models = _ordered(by_model, MODEL_NAMES)[::-1]
        for y, m in enumerate(models):
            A = by_model[m]
            res = [t for t in A.index if t != "class_weighting"]
            rr = A.loc[res, res].values[~np.eye(len(res), dtype=bool)]
            cw = A.loc["class_weighting", res].values
            for vals, dy, col, lab in [(rr, 0.15, C_A, "giữa các kỹ thuật resample"), (cw, -0.15, C_ACCENT, "class weighting ↔ resample")]:
                ax.hlines(y + dy, vals.min(), vals.max(), color=col, lw=7, alpha=0.8, label=lab if y == 0 else None)
        ax.set_yticks(range(len(models)), [LABEL.get(m, m) for m in models], fontsize=11)
        ax.set_xlabel("Đồng thuận thứ hạng feature (1 = giống hệt)")
        ax.set_title(name, fontsize=13)
    axes[0].legend(loc="lower left", fontsize=10)
    fig.suptitle("Đổi sang class weighting làm đổi feature quan trọng nhiều hơn đổi giữa các kỹ thuật resample (trừ LR Sparkov)",
                 fontsize=13)
    return _finish(fig, save_path)
