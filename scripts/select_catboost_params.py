"""
Chọn lại tham số CatBoost từ study Optuna ĐÃ CÓ (không tune lại): trong các trial COMPLETE có CV PR-AUC không thua bản tốt nhất quá
`--epsilon`, lấy trial có chi phí SHAP chính xác ước tính thấp nhất.

Vì sao: SHAP chính xác của CatBoost (oblivious tree, 2^depth lá mỗi cây) tốn khoảng  số_cây × 2^depth × depth²  nên depth = 11 làm SHAP
trên toàn bộ 555.719 dòng test × 50 lần train rất chậm. Nhiều trial depth 8–9 có CV PR-AUC chỉ thấp hơn bản tốt nhất vài phần mười nghìn.

Quy tắc (cố định TRƯỚC khi nhìn kết quả test): ứng viên = COMPLETE và value >= best - epsilon; chọn chi phí nhỏ nhất, hoà thì value cao hơn.
Chi phí chỉ là ước lượng tương đối; số giờ thật phải đo bằng `scripts/run_rerun.py pilot --models catboost --force-cpu`.

Cách dùng (chỉ cần thư viện chuẩn):
    python scripts/select_catboost_params.py                      # in bảng, KHÔNG ghi gì
    python scripts/select_catboost_params.py --epsilon 0.0005     # quy tắc chặt hơn
    python scripts/select_catboost_params.py --trial 48 --write   # chọn tay một trial và ghi vào best_params.json
    python scripts/select_catboost_params.py --write              # ghi trial do quy tắc chọn

`--write` ghi đè mục "catboost" của file best_params và giữ bản gốc trong khoá "cv_best_original" (chỉ lưu lần đầu). Kèm khoá
"selection" để truy vết. Chọn --dataset ulb thì đọc/ghi bản ULB.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "sparkov": (ROOT / "results" / "tuning_checkpoints" / "catboost.db", ROOT / "results" / "best_params.json"),
    "ulb": (ROOT / "results" / "tuning_checkpoints_ulb" / "catboost.db", ROOT / "results" / "best_params_ulb.json"),
}
PARAM_NAMES = ("iterations", "depth", "learning_rate", "l2_leaf_reg")


def cost_proxy(depth: int, iterations: int) -> float:
    """Chi phí SHAP chính xác ước tính (đơn vị tương đối)."""
    return float(iterations) * (2 ** int(depth)) * (int(depth) ** 2)


def load_trials(db_path: Path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        direction = con.execute("select direction from study_directions").fetchone()[0]
        rows = con.execute(
            """select t.number, v.value, p.param_name, p.param_value
               from trials t join trial_values v on v.trial_id = t.trial_id
               join trial_params p on p.trial_id = t.trial_id
               where t.state = 'COMPLETE' and v.value is not null"""
        ).fetchall()
    finally:
        con.close()
    if direction != "MAXIMIZE":
        sys.exit(f"Study có hướng {direction}, script giả định MAXIMIZE (PR-AUC)")
    trials = {}
    for number, value, name, pval in rows:
        t = trials.setdefault(number, {"number": number, "value": float(value)})
        t[name] = float(pval)
    out = []
    for t in trials.values():
        if all(k in t for k in PARAM_NAMES):
            t["depth"], t["iterations"] = int(t["depth"]), int(t["iterations"])
            t["cost"] = cost_proxy(t["depth"], t["iterations"])
            out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="sparkov", choices=sorted(SOURCES))
    ap.add_argument("--epsilon", type=float, default=0.0015, help="CV PR-AUC được phép thấp hơn bản tốt nhất (mặc định 0,0015)")
    ap.add_argument("--trial", type=int, default=None, help="chọn tay một số trial thay vì theo quy tắc")
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--write", action="store_true", help="ghi vào best_params (mặc định chỉ in bảng)")
    a = ap.parse_args()

    db_path, params_path = SOURCES[a.dataset]
    db_path = a.db or db_path
    trials = load_trials(db_path)
    if not trials:
        sys.exit(f"Không có trial COMPLETE trong {db_path}")
    best = max(trials, key=lambda t: t["value"])
    pool = [t for t in trials if t["value"] >= best["value"] - a.epsilon]
    if a.trial is not None:
        chosen = next((t for t in trials if t["number"] == a.trial), None)
        if chosen is None:
            sys.exit(f"Không có trial COMPLETE số {a.trial}")
    else:
        chosen = min(pool, key=lambda t: (t["cost"], -t["value"]))

    print(f"Study: {db_path.relative_to(ROOT) if db_path.is_relative_to(ROOT) else db_path} | {len(trials)} trial COMPLETE | "
          f"tốt nhất #{best['number']} = {best['value']:.5f} | epsilon = {a.epsilon}")
    print(f"{'trial':>6} {'CV PR-AUC':>10} {'Δ vs best':>10} {'depth':>5} {'iters':>5} {'lr':>7} {'l2':>6} {'SHAP nhanh hơn':>15}")
    for t in sorted(pool, key=lambda t: t["cost"]):
        mark = "  <== chọn" if t["number"] == chosen["number"] else ("  (tốt nhất)" if t["number"] == best["number"] else "")
        print(f"#{t['number']:>5} {t['value']:>10.5f} {t['value'] - best['value']:>+10.5f} {t['depth']:>5} {t['iterations']:>5} "
              f"{t['learning_rate']:>7.4f} {t['l2_leaf_reg']:>6.2f} {best['cost'] / t['cost']:>14.1f}x{mark}")
    if chosen not in pool:
        print(f"\nCảnh báo: trial #{chosen['number']} nằm NGOÀI nhóm ứng viên (kém bản tốt nhất {best['value'] - chosen['value']:.5f}).")

    if not a.write:
        print("\nChế độ xem (chưa ghi). Thêm --write để ghi vào", params_path.relative_to(ROOT))
        return

    data = json.loads(params_path.read_text(encoding="utf-8"))
    block = data.get("catboost") or {}
    if "cv_best_original" not in block:
        block["cv_best_original"] = {k: block.get(k) for k in ("best_pr_auc", "best_params") if k in block}
    old_params = block["cv_best_original"].get("best_params", {})
    new_params = {
        "iterations": chosen["iterations"], "depth": chosen["depth"],
        "learning_rate": chosen["learning_rate"], "l2_leaf_reg": chosen["l2_leaf_reg"],
        "eval_metric": old_params.get("eval_metric", "PRAUC"), "verbose": old_params.get("verbose", 0),
    }
    block.update({
        "model_name": "catboost",
        "best_pr_auc": chosen["value"],
        "best_params": new_params,
        "selection": {
            "rule": "COMPLETE và value >= best - epsilon, chọn chi phí SHAP ước nhỏ nhất (iterations * 2**depth * depth**2)",
            "epsilon": a.epsilon, "trial_number": chosen["number"], "manual": a.trial is not None,
            "best_trial_number": best["number"], "best_value": best["value"],
            "cost_ratio_vs_best": round(best["cost"] / chosen["cost"], 2),
            "source_db": str(db_path.relative_to(ROOT)) if db_path.is_relative_to(ROOT) else str(db_path),
        },
    })
    data["catboost"] = block
    params_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nĐã ghi trial #{chosen['number']} vào {params_path.relative_to(ROOT)} (bản gốc nằm ở catboost.cv_best_original).")


if __name__ == "__main__":
    main()
