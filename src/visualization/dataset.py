"""
dataset.py — Biểu đồ insight về dataset (Sparkov + ULB).

Mỗi hàm nhận DataFrame thô, trả về matplotlib Figure (và lưu PNG nếu truyền
`save_path`). Chỉ vẽ những góc nhìn có tín hiệu thật hoặc chứng minh được 1 giả
thuyết SAI (vd. khoảng cách khách–merchant) — không vẽ cho đủ số lượng.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

C_LEGIT = "#4c78a8"
C_FRAUD = "#e4572e"
C_NEUTRAL = "#9aa0a6"

sns.set_theme(style="whitegrid", context="notebook")


def _finish(fig, save_path: Optional[Path]):
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ===== Feature derivation dùng chung =====

def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def add_sparkov_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm hour, month, dist_km, gap_hours (giờ kể từ giao dịch trước của cùng thẻ)."""
    out = df.copy()
    out["ts"] = pd.to_datetime(out["trans_date_trans_time"])
    out["hour"] = out["ts"].dt.hour
    out["month"] = out["ts"].dt.month
    out["dist_km"] = haversine_km(out["lat"], out["long"], out["merch_lat"], out["merch_long"])
    out = out.sort_values(["cc_num", "ts"])
    out["gap_hours"] = out.groupby("cc_num")["ts"].diff().dt.total_seconds() / 3600
    return out


# ===== Sparkov vs ULB =====

def plot_imbalance_comparison(y_sparkov, y_ulb, save_path: Optional[Path] = None):
    """Số lượng lớp (log) + tỷ lệ fraud của 2 dataset — bối cảnh cho toàn bộ thí nghiệm imbalance."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ["Sparkov", "ULB"]
    ys = [np.asarray(y_sparkov), np.asarray(y_ulb)]
    legit = [int((y == 0).sum()) for y in ys]
    fraud = [int((y == 1).sum()) for y in ys]
    x = np.arange(2)
    axes[0].bar(x - 0.18, legit, 0.36, color=C_LEGIT, label="Hợp lệ")
    axes[0].bar(x + 0.18, fraud, 0.36, color=C_FRAUD, label="Gian lận")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("Số giao dịch (log)")
    axes[0].set_title("Số lượng mỗi lớp")
    axes[0].legend()
    for i in range(2):
        axes[0].text(i - 0.18, legit[i], f"{legit[i]:,}", ha="center", va="bottom", fontsize=8)
        axes[0].text(i + 0.18, fraud[i], f"{fraud[i]:,}", ha="center", va="bottom", fontsize=8)
    rates = [f / (f + l) * 100 for f, l in zip(fraud, legit)]
    bars = axes[1].bar(names, rates, color=[C_FRAUD, C_FRAUD], alpha=0.85)
    axes[1].set_ylabel("Tỷ lệ gian lận (%)")
    axes[1].set_title("Mức độ mất cân bằng")
    for b, r, f in zip(bars, rates, fraud):
        axes[1].text(b.get_x() + b.get_width() / 2, r, f"{r:.2f}%\n({f:,} mẫu fraud)", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylim(0, max(rates) * 1.35)
    return _finish(fig, save_path)


# ===== Sparkov =====

def plot_hour_profile(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Tỷ lệ fraud theo giờ (cột) + lưu lượng giao dịch (đường): fraud dồn vào 22h–3h dù lưu lượng đêm thấp."""
    g = df.groupby("hour")["is_fraud"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(11, 4.4))
    night = g.index.isin([22, 23, 0, 1, 2, 3])
    ax.bar(g.index, g["mean"] * 100, color=np.where(night, C_FRAUD, C_NEUTRAL))
    ax.set_xlabel("Giờ trong ngày")
    ax.set_ylabel("Tỷ lệ gian lận (%)")
    ax.set_xticks(range(24))
    ax.set_title("Gian lận tập trung vào khung 22h–3h")
    ax2 = ax.twinx()
    ax2.plot(g.index, g["size"] / 1000, color=C_LEGIT, marker="o", ms=3, lw=1.5)
    ax2.set_ylabel("Số giao dịch (nghìn)", color=C_LEGIT)
    ax2.grid(False)
    day, nt = df[~df["hour"].isin([22, 23, 0, 1, 2, 3])]["is_fraud"].mean(), df[df["hour"].isin([22, 23, 0, 1, 2, 3])]["is_fraud"].mean()
    ax.text(0.5, 0.97, f"Đêm (22–3h): {nt*100:.2f}%\nNgày: {day*100:.3f}%  (gấp {nt/day:.0f}x)",
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec=C_NEUTRAL))
    return _finish(fig, save_path)


def plot_hour_category_heatmap(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Heatmap tỷ lệ fraud (%) theo category × khung giờ — chỉ ra tổ hợp rủi ro cao (vd. shopping_net ban đêm)."""
    bins = [-1, 3, 7, 11, 15, 19, 21, 23]
    labels = ["0–3h", "4–7h", "8–11h", "12–15h", "16–19h", "20–21h", "22–23h"]
    d = df.assign(slot=pd.cut(df["hour"], bins=bins, labels=labels))
    pt = d.pivot_table(index="category", columns="slot", values="is_fraud", aggfunc="mean", observed=True) * 100
    pt = pt.loc[pt.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    sns.heatmap(pt, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "Tỷ lệ gian lận (%)"}, ax=ax, linewidths=0.4)
    ax.set_xlabel("Khung giờ")
    ax.set_ylabel("")
    ax.set_title("Category × khung giờ: rủi ro chồng lên nhau")
    fig.text(0.01, -0.01, "Ô trống = category không có giao dịch nào ở khung giờ đó (mỗi category chỉ xuất hiện ở một số khung giờ).",
             fontsize=8, color=C_NEUTRAL)
    return _finish(fig, save_path)


def plot_amount_fraud_rate(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Tỷ lệ fraud theo dải số tiền vs tỷ trọng số giao dịch — số tiền lớn hiếm nhưng gần như 1/5 là fraud."""
    edges = [0, 10, 50, 100, 200, 500, 1000, np.inf]
    labels = ["<10", "10–50", "50–100", "100–200", "200–500", "500–1000", ">1000"]
    d = df.assign(bin=pd.cut(df["amt"], edges, labels=labels))
    g = d.groupby("bin", observed=True)["is_fraud"].agg(["mean", "size"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].bar(g.index.astype(str), g["mean"] * 100, color=C_FRAUD)
    axes[0].set_ylabel("Tỷ lệ gian lận (%)")
    axes[0].set_xlabel("Số tiền (USD)")
    axes[0].set_title("Tỷ lệ gian lận theo dải số tiền")
    for i, v in enumerate(g["mean"] * 100):
        axes[0].text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    share = g["size"] / g["size"].sum() * 100
    axes[1].bar(g.index.astype(str), share, color=C_LEGIT)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("% tổng số giao dịch (log)")
    axes[1].set_xlabel("Số tiền (USD)")
    axes[1].set_title("…nhưng dải rủi ro cao rất hiếm")
    for i, v in enumerate(share):
        axes[1].text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
    return _finish(fig, save_path)


def plot_distance_myth(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Khoảng cách khách hàng–merchant: KHÔNG phân biệt được fraud (giả thuyết phổ biến nhưng sai trên dataset này)."""
    fig, ax = plt.subplots(figsize=(9, 4.3))
    for lab, col, name in [(0, C_LEGIT, "Hợp lệ"), (1, C_FRAUD, "Gian lận")]:
        s = df.loc[df["is_fraud"] == lab, "dist_km"]
        sns.kdeplot(s, ax=ax, color=col, fill=True, alpha=0.3, label=f"{name} (TB {s.mean():.1f} km)", common_norm=False, clip=(0, None))
    ax.set_xlabel("Khoảng cách khách–merchant (km)")
    ax.set_ylabel("Mật độ")
    ax.set_title("Khoảng cách địa lý KHÔNG phải tín hiệu gian lận ở Sparkov")
    ax.legend()
    return _finish(fig, save_path)


def plot_velocity(df: pd.DataFrame, save_path: Optional[Path] = None):
    """ECDF khoảng thời gian kể từ giao dịch trước của cùng thẻ: fraud đến dồn dập (burst) — tín hiệu mà 9 feature hiện tại KHÔNG có."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for lab, col, name in [(0, C_LEGIT, "Hợp lệ"), (1, C_FRAUD, "Gian lận")]:
        s = df.loc[(df["is_fraud"] == lab) & df["gap_hours"].notna(), "gap_hours"].clip(lower=1e-3)
        xs = np.sort(s.values)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        step = max(len(xs) // 4000, 1)
        ax.plot(xs[::step], ys[::step], color=col, lw=2, label=f"{name} (trung vị {np.median(xs):.2f} giờ)")
    ax.set_xscale("log")
    ax.set_xlabel("Giờ kể từ giao dịch trước của cùng thẻ (log)")
    ax.set_ylabel("Tỷ lệ tích luỹ")
    ax.set_title("Nhịp giao dịch: fraud đến dồn dập hơn hẳn")
    ax.legend(loc="lower right")
    return _finish(fig, save_path)


def plot_month_seasonality(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Tỷ lệ fraud theo tháng — tính mùa vụ (cảnh báo: tập train/test tách theo thời gian có thể lệch phân phối)."""
    g = df.groupby("month")["is_fraud"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(g.index, g["mean"] * 100, color=C_FRAUD, alpha=0.85)
    ax.axhline(df["is_fraud"].mean() * 100, color=C_NEUTRAL, ls="--", label="Trung bình cả năm")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Tháng")
    ax.set_ylabel("Tỷ lệ gian lận (%)")
    ax.set_title("Tính mùa vụ của gian lận")
    ax.legend()
    return _finish(fig, save_path)


# ===== ULB =====

def ulb_feature_separation(df: pd.DataFrame, target: str = "Class") -> pd.DataFrame:
    """Độ tách lớp mỗi feature: |chênh lệch trung bình| / độ lệch chuẩn gộp (Cohen's d)."""
    # Bỏ Time: config.ULB_FEATURE_COLS loại nó (chỉ là thứ tự giao dịch, không phải feature)
    feats = [c for c in df.columns if c not in (target, "Time")]
    f, l = df[df[target] == 1][feats], df[df[target] == 0][feats]
    pooled = np.sqrt((f.var() + l.var()) / 2)
    d = ((f.mean() - l.mean()) / pooled).rename("cohens_d")
    return d.to_frame().assign(abs_d=lambda x: x["cohens_d"].abs()).sort_values("abs_d", ascending=False)


def plot_ulb_top_features(df: pd.DataFrame, target: str = "Class", top_k: int = 4, save_path: Optional[Path] = None):
    """Cohen's d của mọi feature (cột) + phân phối 4 feature tách lớp mạnh nhất."""
    sep = ulb_feature_separation(df, target)
    top = sep.head(top_k).index.tolist()
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.6, 1])
    ax0 = fig.add_subplot(gs[0, :])
    order = sep.head(15)
    ax0.barh(order.index[::-1], order["cohens_d"][::-1], color=[C_FRAUD if v > 0 else C_LEGIT for v in order["cohens_d"][::-1]])
    ax0.set_xlabel("Cohen's d (fraud − hợp lệ), top 15 feature")
    ax0.set_title("ULB: feature nào tách fraud khỏi hợp lệ mạnh nhất (V1..V28 là PCA ẩn danh)")
    for i, name in enumerate(top):
        ax = fig.add_subplot(gs[1, i])
        lo, hi = df[name].quantile([0.005, 0.995])
        for lab, col in [(0, C_LEGIT), (1, C_FRAUD)]:
            sns.kdeplot(df.loc[df[target] == lab, name].clip(lo, hi), ax=ax, color=col, fill=True, alpha=0.3, common_norm=False,
                        label="Gian lận" if lab else "Hợp lệ")
        ax.set_title(name)
        ax.set_xlabel("")
        if i:
            ax.set_ylabel("")
        if i == 0:
            ax.legend(fontsize=8)
    return _finish(fig, save_path)


def plot_ulb_scatter(df: pd.DataFrame, target: str = "Class", save_path: Optional[Path] = None, seed: int = 42):
    """2 feature tách lớp mạnh nhất: fraud nằm gọn 1 vùng — vì sao mô hình cây/tuyến tính đều học được."""
    sep = ulb_feature_separation(df, target)
    a, b = sep.index[:2]
    legit = df[df[target] == 0].sample(min(20000, int((df[target] == 0).sum())), random_state=seed)
    fraud = df[df[target] == 1]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(legit[a], legit[b], s=4, c=C_LEGIT, alpha=0.25, label=f"Hợp lệ (mẫu {len(legit):,})")
    ax.scatter(fraud[a], fraud[b], s=14, c=C_FRAUD, alpha=0.8, label=f"Gian lận ({len(fraud):,})")
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_title(f"ULB: {a} vs {b}")
    ax.legend()
    return _finish(fig, save_path)
