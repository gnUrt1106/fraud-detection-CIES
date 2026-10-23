"""
run_rerun.py — Chạy lại theo seed trên toàn bộ train Sparkov (thiết kế của giảng viên, không bootstrap).

Các giai đoạn (subcommand):
    pilot   Đo thời gian 1 lần train + SHAP mẫu nhỏ cho từng model → ước lượng tổng giờ. Chạy trước tiên.
    cache   Tạo dữ liệu đã resample theo (kỹ thuật, seed). Việc nặng nhất (SMOTE-ENN ~145 phút/lần) và làm được
            trên bất kỳ máy nào; chia seed cho nhiều máy rồi copy thư mục cache về máy train.
    train   Chạy các tổ hợp (model × kỹ thuật) trọn các seed trên MỘT máy, ghi results/rerun/<machine>.json.
            Chạy lại cùng lệnh sẽ bỏ qua tổ hợp đã xong (resume).
    status  Xem tiến độ: cache theo seed và kết quả theo máy.
    merge   Gộp các file results/rerun/<machine>.json → results/cies_rerun_summary.json + .csv.

Ví dụ:
    python scripts/run_rerun.py pilot --machine m2 --force-cpu
    python scripts/run_rerun.py cache --techniques smote_enn --seeds 0-3
    python scripts/run_rerun.py train --machine pc4070 --models xgboost catboost ann random_forest
    python scripts/run_rerun.py train --machine ryzen --models logistic_regression --force-cpu
    python scripts/run_rerun.py merge

Quy tắc: mỗi (model, kỹ thuật) phải chạy trọn mọi seed trên MỘT máy/thiết bị (xem rerun_stability.py).
"""

import argparse
import csv
import glob
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    IMBALANCE_TECHNIQUES, MODEL_NAMES, PROCESSED_DATA_DIR, RERUN_N_SEEDS, RERUN_RESULTS_DIR,
    RESAMPLE_CACHE_RERUN_DIR, RESULTS_DIR,
)
from src.explainability import rerun_stability as rs  # noqa: E402
from src.utils.isolation import run_isolated  # noqa: E402

logger = logging.getLogger("run_rerun")

SUMMARY_JSON = RESULTS_DIR / "cies_rerun_summary.json"
SUMMARY_CSV = RESULTS_DIR / "cies_rerun_summary.csv"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR, help="Thư mục có train_encoded/test_encoded.parquet")
    p.add_argument("--cache-root", type=Path, default=RESAMPLE_CACHE_RERUN_DIR, help="Gốc cache resample theo seed")
    p.add_argument("--seeds", default=f"0-{RERUN_N_SEEDS - 1}", help="vd. 0-9 hoặc 0,2,5")
    p.add_argument("--techniques", nargs="+", default=list(IMBALANCE_TECHNIQUES), choices=IMBALANCE_TECHNIQUES)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    p.add_argument("--machine", required=True, help="Nhãn máy, vd. pc4070 / ryzen / m2 (đặt tên file kết quả)")
    p.add_argument("--force-cpu", action="store_true", help="Ép CPU cho XGBoost/CatBoost/ANN dù máy có GPU NVIDIA")
    p.add_argument("--shap-batch", type=int, default=10_000, help="Số dòng mỗi đợt SHAP (chỉ ảnh hưởng RAM/tốc độ)")
    p.add_argument("--shap-backend", default="auto", choices=rs_backends())


def rs_backends():
    from src.explainability.shap_utils import SHAP_BACKENDS
    return list(SHAP_BACKENDS)


def _check_best_params(models, allow_default: bool) -> None:
    """Optuna là bắt buộc giữ nguyên: thiếu best_params.json thì build_model lặng lẽ dùng tham số mặc định → dừng."""
    from src.models.tune import load_best_params

    missing = [m for m in models if not load_best_params(m)]
    if missing and not allow_default:
        sys.exit(f"Thiếu tham số Optuna của {missing} trong results/best_params.json. "
                 f"Copy file đó sang máy này, hoặc thêm --allow-default-params nếu CHỦ Ý dùng tham số mặc định.")


def cmd_pilot(a) -> None:
    _check_best_params(a.models, a.allow_default_params)
    out, rows = {}, []
    for m in a.models:
        logger.info(f"Pilot {m} ...")
        try:
            r = run_isolated(rs.pilot_job, a.data_dir, m, pilot_rows=a.pilot_rows, shap_batch=a.shap_batch,
                             shap_backend=a.shap_backend, force_cpu=a.force_cpu, timeout=None)
            out[m] = r
            rows.append(r)
            logger.info(f"  {m}: train {r['seconds_train']}s, SHAP toàn test ước {r['est_seconds_shap_full_test']}s, "
                        f"~{r['est_hours_for_model']} giờ cho 50 lần train (chưa gồm resample) [{r['device']}]")
        except Exception as e:  # noqa: BLE001
            out[m] = {"error": str(e)[:2000]}
            logger.error(f"  {m} lỗi: {str(e)[:500]}")
    rs.atomic_write_json(RERUN_RESULTS_DIR / f"pilot_{a.machine}.json", out)
    print("\nmodel | thiết bị | train(s) | SHAP toàn test ước(s) | giờ/model(50 lần)")
    for r in rows:
        print(f"{r['model']} | {r['device']} | {r['seconds_train']} | {r['est_seconds_shap_full_test']:.0f} | {r['est_hours_for_model']}")


def cmd_cache(a) -> None:
    seeds = rs.parse_seeds(a.seeds)
    techs = [t for t in a.techniques if t != "class_weighting"]
    for seed in seeds:
        for t in techs:
            t0 = time.time()
            logger.info(f"Cache {t} seed {seed} ...")
            try:
                r = run_isolated(rs.cache_job, a.data_dir, t, seed, a.cache_root, timeout=None)
                logger.info(f"  xong sau {time.time() - t0:.0f}s: {r.get('n_after')} dòng ({r.get('n_fraud_after')} fraud)")
            except Exception as e:  # noqa: BLE001
                logger.error(f"  {t} seed {seed} lỗi: {str(e)[:500]}")


def _load_machine_file(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"design": rs.RERUN_DESIGN, "results": [], "errors": []}


def cmd_train(a) -> None:
    _check_best_params(a.models, a.allow_default_params)
    seeds = rs.parse_seeds(a.seeds)
    path = RERUN_RESULTS_DIR / f"{a.machine}.json"
    store = _load_machine_file(path)
    done = {r["combination"] for r in store["results"] if r.get("complete")}
    for m in a.models:
        for t in a.techniques:
            combo = f"{m}__{t}"
            if combo in done:
                logger.info(f"Bỏ qua {combo} (đã xong trên máy này)")
                continue
            logger.info(f"=== {combo} ===")
            try:
                r = run_isolated(
                    rs.combo_job, a.data_dir, m, t, seeds, cache_root=a.cache_root, shap_rows=a.shap_rows,
                    shap_batch=a.shap_batch, shap_backend=a.shap_backend, force_cpu=a.force_cpu,
                    machine=a.machine, timeout=None,
                )
                store["results"] = [x for x in store["results"] if x["combination"] != combo] + [r]
                store["errors"] = [e for e in store["errors"] if e.get("combination") != combo]
                logger.info(f"  CIES {r['cies_metrics']['cies_score']:.4f}, PR-AUC {r['pr_auc_summary']['mean']:.4f} "
                            f"± {r['pr_auc_summary']['std']:.4f}")
            except Exception as e:  # noqa: BLE001
                store["errors"] = [x for x in store["errors"] if x.get("combination") != combo] + [
                    {"combination": combo, "error": str(e)[:2000]}]
                logger.error(f"  {combo} lỗi: {str(e)[:500]}")
            rs.atomic_write_json(path, store)  # ghi sau MỖI tổ hợp → mất điện/ngắt giữa chừng vẫn giữ phần đã xong


def cmd_status(a) -> None:
    seeds = rs.parse_seeds(a.seeds)
    print("Cache resample (có file / tổng seed):")
    for t in [t for t in a.techniques if t != "class_weighting"]:
        have = [s for s in seeds if (rs.seed_cache_dir(a.cache_root, s) / f"train_encoded_{t}.parquet").exists()]
        print(f"  {t}: {len(have)}/{len(seeds)}  thiếu seed: {sorted(set(seeds) - set(have)) or '-'}")
    print("Kết quả theo máy:")
    for f in sorted(glob.glob(str(RERUN_RESULTS_DIR / "*.json"))):
        if Path(f).name.startswith("pilot_"):
            continue
        s = _load_machine_file(Path(f))
        ok = [r["combination"] for r in s["results"] if r.get("complete")]
        print(f"  {Path(f).stem}: {len(ok)} tổ hợp đủ seed, {len(s['errors'])} lỗi")


def cmd_merge(a) -> None:
    per_file = {}
    for f in sorted(glob.glob(str(RERUN_RESULTS_DIR / "*.json"))):
        if Path(f).name.startswith("pilot_"):
            continue
        per_file[Path(f).name] = _load_machine_file(Path(f))["results"]
    if not per_file:
        sys.exit(f"Không có file kết quả nào trong {RERUN_RESULTS_DIR}")
    merged = rs.merge_rerun_results(per_file)
    for p in merged["problems"]:
        logger.warning(p)
    rs.atomic_write_json(SUMMARY_JSON, {"design": rs.RERUN_DESIGN, **merged})
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "technique", "cies", "mean_spearman", "pr_auc_mean", "pr_auc_std", "pr_auc_min", "pr_auc_max",
                    "n_seeds_ok", "identical_shap", "device", "machine"])
        for r in merged["results"]:
            c, p = r["cies_metrics"], r["pr_auc_summary"]
            w.writerow([r["model_name"], r["imbalance_technique"], round(c["cies_score"], 6), round(c["mean_spearman"], 6),
                        round(p["mean"], 6), round(p["std"], 6), round(p["min"], 6), round(p["max"], 6), p["n"],
                        r["diagnostics"]["identical_mean_abs_shap"], r["config"]["device"], r["config"]["machine"]])
    print(f"Đã gộp {len(merged['results'])} tổ hợp, {len(merged['problems'])} cảnh báo → {SUMMARY_JSON.name}, {SUMMARY_CSV.name}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pilot", help="Đo thời gian để ước lượng")
    _add_common(p); _add_model_args(p)
    p.add_argument("--pilot-rows", type=int, default=2_000, help="Số dòng test dùng đo tốc độ SHAP")
    p.add_argument("--allow-default-params", action="store_true")
    p.set_defaults(fn=cmd_pilot)

    p = sub.add_parser("cache", help="Tạo cache dữ liệu resample theo seed")
    _add_common(p)
    p.set_defaults(fn=cmd_cache)

    p = sub.add_parser("train", help="Chạy tổ hợp model × kỹ thuật trọn các seed")
    _add_common(p); _add_model_args(p)
    p.add_argument("--shap-rows", type=int, default=None, help="Mặc định: SHAP TOÀN BỘ test. Số nguyên = mẫu phân tầng (chỉ để chạy thử)")
    p.add_argument("--allow-default-params", action="store_true")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("status", help="Xem tiến độ")
    _add_common(p)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("merge", help="Gộp kết quả các máy")
    p.set_defaults(fn=cmd_merge)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":  # bắt buộc cho spawn trên Windows/macOS
    main()
