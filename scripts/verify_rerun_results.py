"""
verify_rerun_results.py - Kiểm độc lập kết quả chạy lại theo seed (không cần torch/xgboost; chỉ numpy + scipy).

Đọc results/cies_rerun_summary[_ulb].json (và .csv đi kèm), TÍNH LẠI từ dữ liệu thô trong `run_logs`, rồi so với con số đã lưu:
  - PR-AUC: trung bình / độ lệch chuẩn / min / max qua 10 seed (so với `pr_auc_summary`);
  - CIES, khoảng cách thứ hạng, Spearman: tính lại từ `mean_abs_shap` từng seed bằng đúng công thức của
    src/explainability/cies.py (hạng trung bình khi hòa, trọng số 1/min hạng, chuẩn hóa bằng p-1, 45 cặp);
  - ma trận nhầm lẫn: TP+FN = số fraud của tập kiểm tra, TP+FP+TN+FN = số dòng tập kiểm tra;
  - nhất quán toàn bộ: 25 tổ hợp, 10 seed (0-9), cùng data_fingerprint, n_eval_shap = n_test, một môi trường, bảng CSV khớp JSON.

    python scripts/verify_rerun_results.py                 # cả Sparkov và ULB
    python scripts/verify_rerun_results.py --dataset ulb

Thoát với mã 0 nếu mọi kiểm tra đạt, 1 nếu có sai lệch (in rõ từng chỗ).
"""

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-9
FILES = {"sparkov": "cies_rerun_summary", "ulb": "cies_rerun_summary_ulb"}
MODELS = ["logistic_regression", "random_forest", "xgboost", "catboost", "ann"]
TECHS = ["smote", "smote_enn", "adasyn", "borderline_smote", "class_weighting"]
EXPLAINER = {"logistic_regression": "linear", "random_forest": "tree-xgb-gpu", "xgboost": "tree-native",
             "catboost": "tree-native", "ann": "deep-native"}


def _distance(ra, rb):
    w = 1.0 / np.minimum(ra, rb)
    w = w / w.sum()
    return float(np.sum(w * np.abs(ra - rb) / (len(ra) - 1)))


def recompute_cies(mean_abs_runs):
    ranks = [rankdata(-np.asarray(m, dtype=float), method="average") for m in mean_abs_runs]
    dist, sp = [], []
    for i, j in combinations(range(len(ranks)), 2):
        dist.append(_distance(ranks[i], ranks[j]))
        a, b = np.asarray(mean_abs_runs[i], float), np.asarray(mean_abs_runs[j], float)
        if a.std() > 0 and b.std() > 0:
            sp.append(spearmanr(a, b)[0])
    return {"cies_score": 1.0 - float(np.mean(dist)), "mean_rank_distance": float(np.mean(dist)),
            "std_rank_distance": float(np.std(dist)), "mean_spearman": float(np.mean(sp)) if sp else 0.0}


def verify(dataset: str, results_dir: Path):
    problems, notes = [], []
    base = results_dir / FILES[dataset]
    S = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    R = S["results"]
    if S.get("problems"):
        problems.append(f"file tổng hợp còn {len(S['problems'])} cảnh báo merge: {S['problems'][:2]}")
    keys = {(r["model_name"], r["imbalance_technique"]) for r in R}
    if len(R) != 25 or keys != {(m, t) for m in MODELS for t in TECHS}:
        problems.append(f"không đủ lưới 5x5: {len(R)} mục, {len(keys)} cặp khác nhau")
    cfg = [r["config"] for r in R]
    for field in ("data_fingerprint", "n_train", "n_test", "n_eval_shap", "n_eval_fraud"):
        vals = sorted({c[field] for c in cfg}, key=str)
        notes.append(f"{field}: {vals}")
        if len(vals) != 1:
            problems.append(f"{field} không đồng nhất: {vals}")
    if cfg[0]["n_eval_shap"] != cfg[0]["n_test"]:
        problems.append("n_eval_shap != n_test (SHAP không trên 100% tập kiểm tra)")
    envs = {json.dumps(c["environment"], sort_keys=True) for c in cfg}
    notes.append(f"môi trường: {len(envs)} loại")
    if len(envs) != 1:
        problems.append(f"{len(envs)} môi trường khác nhau trong một dataset")
    n_test, n_fraud = cfg[0]["n_test"], cfg[0]["n_eval_fraud"]
    max_dpr = max_dcies = max_dsp = 0.0
    std_ddof = {0: 0, 1: 0}
    for r in R:
        name = r["combination"]
        if not r.get("complete"):
            problems.append(f"{name}: chưa complete")
        logs = r["run_logs"]
        seeds = [x["seed"] for x in logs]
        if sorted(seeds) != list(range(10)) or any(x["status"] != "success" for x in logs):
            problems.append(f"{name}: seed/status lạ: {seeds}")
        if r["config"]["seeds"] != list(range(10)):
            problems.append(f"{name}: config.seeds = {r['config']['seeds']}")
        expl = {x["explainer"] for x in logs}
        if expl != {EXPLAINER[r["model_name"]]}:
            problems.append(f"{name}: explainer {expl}, mong đợi {EXPLAINER[r['model_name']]}")
        pr = np.array([x["metrics"]["pr_auc"] for x in logs])
        ps = r["pr_auc_summary"]
        for k, v in (("mean", pr.mean()), ("min", pr.min()), ("max", pr.max())):
            if abs(ps[k] - v) > TOL:
                problems.append(f"{name}: pr_auc {k} lưu {ps[k]} != tính lại {v}")
        for dd in (0, 1):
            if abs(ps["std"] - pr.std(ddof=dd)) < 1e-9:
                std_ddof[dd] += 1
        max_dpr = max(max_dpr, abs(ps["mean"] - pr.mean()))
        for x in logs:
            m = x["metrics"]
            tp, fp, tn, fn = (m["true_positives"], m["false_positives"], m["true_negatives"], m["false_negatives"])
            if tp + fn != n_fraud or tp + fp + tn + fn != n_test:
                problems.append(f"{name} seed {x['seed']}: ma trận nhầm lẫn không khớp (TP+FN={tp + fn}, tổng={tp + fp + tn + fn})")
        order = sorted(logs, key=lambda x: x["seed"])
        new = recompute_cies([x["mean_abs_shap"] for x in order])
        old = r["cies_metrics"]
        for k in ("cies_score", "mean_rank_distance", "std_rank_distance", "mean_spearman"):
            if abs(old[k] - new[k]) > 1e-9:
                problems.append(f"{name}: {k} lưu {old[k]:.10f} != tính lại {new[k]:.10f}")
        max_dcies = max(max_dcies, abs(old["cies_score"] - new["cies_score"]))
        max_dsp = max(max_dsp, abs(old["mean_spearman"] - new["mean_spearman"]))
        if old["n_runs"] != 10 or ps["n"] != 10:
            problems.append(f"{name}: n_runs/n = {old['n_runs']}/{ps['n']}")
    notes.append(f"độ lệch chuẩn PR-AUC khớp ddof=0 ở {std_ddof[0]}/25, ddof=1 ở {std_ddof[1]}/25 tổ hợp")
    # CSV khớp JSON
    rows = list(csv.DictReader(open(base.with_suffix(".csv"), encoding="utf-8")))
    by = {(r["model_name"], r["imbalance_technique"]): r for r in R}
    if len(rows) != 25:
        problems.append(f"CSV có {len(rows)} dòng")
    for row in rows:
        r = by.get((row["model"], row["technique"]))
        if r is None or abs(float(row["cies"]) - r["cies_metrics"]["cies_score"]) > 1e-6 \
                or abs(float(row["pr_auc_mean"]) - r["pr_auc_summary"]["mean"]) > 1e-6:
            problems.append(f"CSV lệch JSON ở {row['model']}/{row['technique']}")
    notes.append(f"max |ΔPR-AUC mean| = {max_dpr:.2e}; max |ΔCIES| = {max_dcies:.2e}; max |ΔSpearman| = {max_dsp:.2e}")
    return problems, notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["sparkov", "ulb", "all"], default="all")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    a = ap.parse_args(argv)
    rc = 0
    for ds in (["sparkov", "ulb"] if a.dataset == "all" else [a.dataset]):
        problems, notes = verify(ds, a.results_dir)
        print(f"== {ds}: {'ĐẠT' if not problems else 'CÓ SAI LỆCH'}")
        for n in notes:
            print("   ", n)
        for p in problems:
            print("   [LỖI]", p)
        rc |= 1 if problems else 0
    return rc


if __name__ == "__main__":
    sys.exit(main())
