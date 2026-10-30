"""
plot_rerun.py — Biểu đồ cho kết quả chạy lại theo seed (thiết kế chính thức), đọc từ bảng tổng hợp của `run_rerun.py merge`.

    python scripts/plot_rerun.py                 # → reports/figures/rerun_cies_heatmap.png, rerun_prauc_vs_cies.png

Hai hình:
    1. Heatmap CIES 5 model × 5 kỹ thuật cho Sparkov và ULB (cùng một thang màu), viền cam + số đậm = kỹ thuật có CIES
       thấp nhất trong từng model.
    2. PR-AUC (trung bình 10 seed, thanh ngang = ±1 độ lệch chuẩn) theo CIES, mỗi dataset một ô. Logistic Regression
       không vẽ vì PR-AUC của nó (Sparkov 0,10–0,12) kéo giãn trục; hệ số Spearman được ghi cả khi có và khi bỏ LR.

Notebook 06 và các hình khác trong reports/figures là của thiết kế bootstrap (thiết kế đầu tiên).
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.config import FIGURES_DIR, IMBALANCE_TECHNIQUES, MODEL_NAMES, RESULTS_DIR  # noqa: E402

MODEL_LABELS = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost",
                "catboost": "CatBoost", "ann": "ANN"}
TECH_LABELS = {"smote": "SMOTE", "smote_enn": "SMOTE-ENN", "adasyn": "ADASYN", "borderline_smote": "Borderline-\nSMOTE",
               "class_weighting": "Class\nWeighting"}
TECH_SHORT = {"smote": "S", "smote_enn": "S-E", "adasyn": "A", "borderline_smote": "B", "class_weighting": "CW"}

# Thang tuần tự một màu (xanh, nhạt → đậm) và bộ màu phân loại 4 slot đã kiểm bằng validator (all-pairs: đạt, CVD
# ở dải 6–8 nên luôn kèm mã thứ hai: hình dạng điểm + nhãn tên model trực tiếp).
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
            "#1c5cab", "#184f95", "#104281", "#0d366b"]
MODEL_COLORS = {"random_forest": "#2a78d6", "xgboost": "#eda100", "catboost": "#008300", "ann": "#e34948"}
MODEL_MARKERS = {"random_forest": "o", "xgboost": "s", "catboost": "D", "ann": "^"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MARK_LOW = "#eb6834"  # viền ô thấp nhất trong model: màu nhấn khác hẳn thang xanh, kèm chữ đậm (không chỉ dựa vào màu)


def load_summary(path: Path) -> dict:
    """{(model, kỹ thuật): {"cies", "pr_mean", "pr_std"}} từ cies_rerun_summary[_ulb].json."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for r in data["results"]:
        out[(r["model_name"], r["imbalance_technique"])] = {
            "cies": float(r["cies_metrics"]["cies_score"]),
            "pr_mean": float(r["pr_auc_summary"]["mean"]),
            "pr_std": float(r["pr_auc_summary"]["std"]),
        }
    return out


def _spearman(x, y) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y)[0])


def _fmt(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}".replace(".", ",").replace("-", "−")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def plot_cies_heatmap(datasets: dict, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    vals = [v["cies"] for d in datasets.values() for v in d.values()]
    vmin, vmax = np.floor(min(vals) * 100) / 100, 1.0
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 5.0), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for k_ax, (ax, (name, d)) in enumerate(zip(axes, datasets.items())):
        M = np.array([[d.get((m, t), {}).get("cies", np.nan) for t in IMBALANCE_TECHNIQUES] for m in MODEL_NAMES])
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i, m in enumerate(MODEL_NAMES):
            row = M[i]
            lo = int(np.nanargmin(row)) if np.isfinite(row).any() else None
            for j in range(len(IMBALANCE_TECHNIQUES)):
                v = row[j]
                if not np.isfinite(v):
                    ax.text(j, i, "—", ha="center", va="center", color=INK_2, fontsize=10)
                    continue
                dark = (v - vmin) / (vmax - vmin) > 0.55
                ax.text(j, i, _fmt(v), ha="center", va="center", fontsize=10,
                        color="white" if dark else INK, fontweight="bold" if j == lo else "normal")
            if lo is not None:  # vẽ sau lưới, hơi thụt vào trong ô để không bị đường phân cách che
                ax.add_patch(Rectangle((lo - 0.45, i - 0.45), 0.9, 0.9, fill=False, edgecolor=MARK_LOW,
                                       linewidth=2.6, zorder=5))
        ax.set_xticks(range(len(IMBALANCE_TECHNIQUES)), [TECH_LABELS[t] for t in IMBALANCE_TECHNIQUES])
        ax.set_yticks(range(len(MODEL_NAMES)), [MODEL_LABELS[m] for m in MODEL_NAMES] if k_ax == 0 else [])
        ax.set_title(name, fontsize=12, color=INK, loc="left", pad=8)
        ax.tick_params(length=0, colors=INK_2, labelsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(IMBALANCE_TECHNIQUES)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(MODEL_NAMES)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02)
    cb.set_label("CIES (1 = thứ hạng feature giống hệt ở mọi seed)", color=INK_2, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    fig.suptitle("CIES qua 10 lần chạy lại (seed 0–9, toàn bộ train) — viền cam, số đậm: kỹ thuật thấp nhất trong model",
                 x=0.02, y=1.02, ha="left", fontsize=12, color=INK)
    fig.text(0.02, 0.01, "Logistic Regression × Class Weighting = 1,000 vì model tất định theo seed. "
             "Không so mức CIES giữa hai dataset (khác tập SHAP và cách tính SHAP).", fontsize=8.5, color=INK_2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def _place_labels(fig, ax, labels, offsets) -> None:
    """Nhãn kỹ thuật cạnh từng điểm; chọn vị trí đầu tiên không chồng lên điểm hay nhãn đã đặt (tính theo pixel)."""
    fig.canvas.draw()
    to_px = ax.transData.transform
    pts = [to_px((x, y)) for _, x, y, _ in labels]
    placed = []
    scale = fig.dpi / 72.0
    for (m, x, y, t), (px, py) in zip(labels, pts):
        text = TECH_SHORT[t]
        w, h = (5.2 * len(text) + 2) * scale, 9 * scale
        best = offsets[0]
        for dx, dy in offsets:
            x0, y0 = px + dx * scale, py + dy * scale
            box = (x0, y0, x0 + w, y0 + h)
            hit_lbl = any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in placed)
            hit_pt = any(box[0] - 4 < qx < box[2] + 4 and box[1] - 4 < qy < box[3] + 4 for qx, qy in pts)
            if not hit_lbl and not hit_pt:
                best = (dx, dy)
                break
        x0, y0 = px + best[0] * scale, py + best[1] * scale
        placed.append((x0, y0, x0 + w, y0 + h))
        ax.annotate(text, (x, y), xytext=best, textcoords="offset points", fontsize=7.5, color=INK_2)


def plot_prauc_vs_cies(datasets: dict, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m for m in MODEL_NAMES if m in MODEL_COLORS]
    offsets = [(5, 4), (5, -11), (-17, 4), (-17, -11), (5, 13), (-17, 13)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.0 * len(datasets), 5.4), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (name, d) in zip(axes, datasets.items()):
        _style(ax)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        labels = []  # (model, x, y, kỹ thuật) — đặt nhãn sau khi đã có trục cuối để tránh chồng nhau
        for m in models:
            pts = [(d[(m, t)]["pr_mean"], d[(m, t)]["pr_std"], d[(m, t)]["cies"], t)
                   for t in IMBALANCE_TECHNIQUES if (m, t) in d]
            if not pts:
                continue
            x, xe, y, ts = zip(*pts)
            ax.errorbar(x, y, xerr=xe, fmt="none", ecolor=MODEL_COLORS[m], elinewidth=1.2, alpha=0.7, zorder=2)
            ax.scatter(x, y, s=60, marker=MODEL_MARKERS[m], color=MODEL_COLORS[m], edgecolor=SURFACE, linewidth=1.5,
                       zorder=3, label=MODEL_LABELS[m])
            labels += [(m, xi, yi, t) for xi, yi, t in zip(x, y, ts)]
        allk = [k for k in d]
        rho_all = _spearman([d[k]["pr_mean"] for k in allk], [d[k]["cies"] for k in allk])
        nolr = [k for k in allk if k[0] != "logistic_regression"]
        rho_nolr = _spearman([d[k]["pr_mean"] for k in nolr], [d[k]["cies"] for k in nolr])
        ax.set_title(f"{name}\nSpearman(PR-AUC, CIES) = {_fmt(rho_all, 2)} (n = {len(allk)}); "
                     f"bỏ LR {_fmt(rho_nolr, 2)} (n = {len(nolr)})", fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("PR-AUC trên toàn bộ test (TB 10 seed, thanh ngang = ±1 độ lệch chuẩn)", fontsize=9, color=INK_2)
        ax.set_ylabel("CIES", fontsize=9, color=INK_2)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.08)
        _place_labels(fig, ax, labels, offsets)
    for ax in axes:
        ax.legend(loc="lower left", fontsize=8.5, frameon=True, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK)
    fig.text(0.02, -0.03, "Không vẽ Logistic Regression (PR-AUC Sparkov 0,10–0,12 kéo giãn trục; CIES ≥ 0,996). "
             "Nhãn cạnh điểm: S = SMOTE, S-E = SMOTE-ENN, A = ADASYN, B = Borderline-SMOTE, CW = Class Weighting.",
             fontsize=8.5, color=INK_2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    a = ap.parse_args(argv)
    datasets = {}
    for name, fname in (("Sparkov", "cies_rerun_summary.json"), ("ULB", "cies_rerun_summary_ulb.json")):
        p = a.results_dir / fname
        if p.exists():
            datasets[name] = load_summary(p)
    if not datasets:
        print(f"Không có bảng tổng hợp trong {a.results_dir} — chạy `scripts/run_rerun.py merge` trước.")
        return 1
    for p in (plot_cies_heatmap(datasets, a.out_dir / "rerun_cies_heatmap.png"),
              plot_prauc_vs_cies(datasets, a.out_dir / "rerun_prauc_vs_cies.png")):
        print(f"Đã ghi {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
