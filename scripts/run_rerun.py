"""
run_rerun.py — Chạy lại theo seed trên toàn bộ train (Sparkov hoặc ULB), không bootstrap.

Các giai đoạn (subcommand):
    pilot   Đo thời gian 1 lần train + SHAP mẫu nhỏ cho từng model → ước lượng tổng giờ. Chạy trước tiên.
    cache   Tạo dữ liệu đã resample theo (kỹ thuật, seed). Việc nặng nhất là SMOTE-ENN (~145 phút/lần trên CPU;
            scripts/gpu_smote_enn.py làm bước này trên GPU). Cache dùng chung cho mọi model.
    train   Chạy các tổ hợp (model × kỹ thuật) trọn các seed, ghi results/rerun/<name>.json.
            Chạy lại cùng lệnh sẽ bỏ qua tổ hợp đã xong (resume).
    status  Xem tiến độ: cache theo seed và kết quả đã có.
    merge   Gộp các file results/rerun/<name>.json → results/cies_rerun_summary.json + .csv.

Ví dụ:
    python scripts/run_rerun.py pilot --force-cpu
    python scripts/run_rerun.py cache --techniques smote_enn --seeds 0-3
    python scripts/run_rerun.py train --models xgboost catboost ann random_forest --shap-backend native
    python scripts/run_rerun.py train --models logistic_regression --force-cpu
    python scripts/run_rerun.py merge
    python scripts/run_rerun.py train --dataset ulb                  # ULB cùng thiết kế với Sparkov

Quy tắc: mỗi (model, kỹ thuật) phải chạy trọn mọi seed trong MỘT lần chạy, cùng thiết bị (xem rerun_stability.py).
Chạy song song nhiều tiến trình thì đặt `--name` khác nhau (mỗi tiến trình ghi file riêng), rồi `merge`.

Mẫu SHAP: mặc định Sparkov = mẫu phân tầng cố định 20% test (111.144 dòng), ULB = toàn bộ test;
PR-AUC luôn trên toàn bộ test. `--shap-rows all` ép toàn bộ test, `--shap-rows N` ép đúng N dòng. Mọi tổ hợp của một
dataset PHẢI dùng cùng mẫu SHAP (`merge` cảnh báo nếu lệch).
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
    IMBALANCE_TECHNIQUES, MODEL_NAMES, PROCESSED_DATA_DIR, RERUN_N_SEEDS,
    RESAMPLE_CACHE_RERUN_DIR, RESULTS_DIR,
)
from src.explainability import rerun_stability as rs  # noqa: E402
from src.utils.isolation import run_isolated  # noqa: E402

logger = logging.getLogger("run_rerun")


def _shap_rows_arg(text: str):
    """--shap-rows: 'all' (toàn bộ test) hoặc số nguyên >= 2 (mẫu phân tầng cố định cỡ đó)."""
    if text.strip().lower() in ("all", "full"):
        return "all"
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"cần 'all' hoặc số nguyên, nhận '{text}'") from None
    if n < 2:
        raise argparse.ArgumentTypeError("số dòng SHAP phải >= 2 (cần cả hai lớp)")
    return n


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", default="sparkov", choices=rs.DATASETS,
                   help="sparkov hoặc ulb — hai dataset được đối xử ngang nhau, cùng thiết kế")
    p.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR,
                   help="Thư mục có train_encoded/test_encoded.parquet (Sparkov) hoặc ulb_train/ulb_test.parquet (ULB)")
    p.add_argument("--cache-root", type=Path, default=None,
                   help=f"Gốc cache resample theo seed (mặc định {RESAMPLE_CACHE_RERUN_DIR}); ULB luôn dùng thư mục con ulb/ của gốc")
    p.add_argument("--seeds", default=f"0-{RERUN_N_SEEDS - 1}", help="vd. 0-9 hoặc 0,2,5")
    p.add_argument("--techniques", nargs="+", default=list(IMBALANCE_TECHNIQUES), choices=IMBALANCE_TECHNIQUES)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    p.add_argument("--name", default="main", help="Tên lần chạy; kết quả ghi vào results/rerun/<name>.json (mặc định main)")
    p.add_argument("--force-cpu", action="store_true", help="Ép CPU cho XGBoost/CatBoost/ANN dù có GPU NVIDIA")
    p.add_argument("--shap-batch", type=int, default=10_000, help="Số dòng mỗi đợt SHAP (chỉ ảnh hưởng RAM/tốc độ)")
    p.add_argument("--shap-backend", default="auto", choices=rs_backends())
    p.add_argument("--shap-jobs", type=int, default=1,
                   help="Số tiến trình tính SHAP song song cho Random Forest (-1 = mọi lõi). Không đổi kết quả; tốn RAM theo số tiến trình")


def rs_backends():
    from src.explainability.shap_utils import SHAP_BACKENDS
    return list(SHAP_BACKENDS)


def _check_best_params(models, allow_default: bool, dataset: str = "sparkov") -> None:
    """Optuna là bắt buộc giữ nguyên: thiếu file tham số thì build_model lặng lẽ dùng tham số mặc định → dừng."""
    from src.config import BEST_PARAMS_ULB_FILE
    from src.models.tune import load_best_params

    path = BEST_PARAMS_ULB_FILE if dataset == "ulb" else None
    missing = [m for m in models if not load_best_params(m, filepath=path)]
    if missing and not allow_default:
        sys.exit(f"Thiếu tham số Optuna của {missing} ({path or 'results/best_params.json'}). "
                 f"Chạy bước tune trước, hoặc thêm --allow-default-params nếu CHỦ Ý dùng tham số mặc định.")


def _paths(a):
    """(thư mục kết quả, gốc cache, file summary json, file summary csv) theo dataset."""
    suffix = "_ulb" if a.dataset == "ulb" else ""
    cache_root = rs.dataset_cache_root(a.dataset, getattr(a, "cache_root", None))
    return (rs.dataset_results_dir(a.dataset), cache_root,
            RESULTS_DIR / f"cies_rerun_summary{suffix}.json", RESULTS_DIR / f"cies_rerun_summary{suffix}.csv")


def cmd_pilot(a) -> None:
    _check_best_params(a.models, a.allow_default_params, a.dataset)
    res_dir, _, _, _ = _paths(a)
    out, rows = {}, []
    for m in a.models:
        logger.info(f"Pilot {m} ...")
        try:
            r = run_isolated(rs.pilot_job, a.data_dir, m, dataset=a.dataset, pilot_rows=a.pilot_rows, shap_batch=a.shap_batch,
                             shap_backend=a.shap_backend, force_cpu=a.force_cpu, shap_jobs=a.shap_jobs,
                             shap_rows=a.shap_rows, timeout=None)
            out[m] = r
            rows.append(r)
            logger.info(f"  {m}: train {r['seconds_train']}s, SHAP trên {r['n_eval_shap']} dòng ước {r['est_seconds_shap_eval']}s "
                        f"(toàn test {r['est_seconds_shap_full_test']}s), ~{r['est_hours_for_model']} giờ cho 50 lần train "
                        f"(chưa gồm resample) [{r['device']}]")
        except Exception as e:  # noqa: BLE001
            out[m] = {"error": str(e)[:2000]}
            logger.error(f"  {m} lỗi: {str(e)[:500]}")
    rs.atomic_write_json(res_dir / f"pilot_{a.name}.json", out)
    print("\nmodel | thiết bị | train(s) | dòng SHAP | SHAP ước(s) | giờ/model(50 lần)")
    for r in rows:
        print(f"{r['model']} | {r['device']} | {r['seconds_train']} | {r['n_eval_shap']} | {r['est_seconds_shap_eval']:.0f} | "
              f"{r['est_hours_for_model']}")


def cmd_cache(a) -> None:
    seeds = rs.parse_seeds(a.seeds)
    _, cache_root, _, _ = _paths(a)
    techs = [t for t in a.techniques if t != "class_weighting"]
    for seed in seeds:
        for t in techs:
            t0 = time.time()
            logger.info(f"Cache {t} seed {seed} ...")
            try:
                r = run_isolated(rs.cache_job, a.data_dir, t, seed, cache_root, dataset=a.dataset, timeout=None)
                logger.info(f"  xong sau {time.time() - t0:.0f}s: {r.get('n_after')} dòng ({r.get('n_fraud_after')} fraud)")
            except Exception as e:  # noqa: BLE001
                logger.error(f"  {t} seed {seed} lỗi: {str(e)[:500]}")


_NON_RESULT_PREFIXES = ("pilot_", "gpu_enn_")  # file đo thời gian / kiểm tra GPU nằm cùng thư mục kết quả


def _load_result_file(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"design": rs.RERUN_DESIGN, "results": [], "errors": []}


def cmd_train(a) -> None:
    _check_best_params(a.models, a.allow_default_params, a.dataset)
    seeds = rs.parse_seeds(a.seeds)
    res_dir, cache_root, _, _ = _paths(a)
    path = res_dir / f"{a.name}.json"
    store = _load_result_file(path)
    done = {r["combination"] for r in store["results"] if r.get("complete")}
    logger.info("Mẫu SHAP: " + ("mặc định theo dataset (Sparkov 20% test, ULB toàn bộ test)" if a.shap_rows is None
                                else f"{a.shap_rows}"))
    for m in a.models:
        for t in a.techniques:
            combo = f"{m}__{t}"
            if combo in done:
                logger.info(f"Bỏ qua {combo} (đã xong)")
                continue
            logger.info(f"=== {combo} ===")
            try:
                r = run_isolated(
                    rs.combo_job, a.data_dir, m, t, seeds, dataset=a.dataset, cache_root=cache_root, shap_rows=a.shap_rows,
                    shap_batch=a.shap_batch, shap_backend=a.shap_backend, force_cpu=a.force_cpu,
                    shap_jobs=a.shap_jobs, timeout=None,
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
    res_dir, cache_root, _, _ = _paths(a)
    print(f"[{a.dataset}] Cache resample (có file / tổng seed):")
    for t in [t for t in a.techniques if t != "class_weighting"]:
        have = [s for s in seeds if (rs.seed_cache_dir(cache_root, s) / f"train_encoded_{t}.parquet").exists()]
        print(f"  {t}: {len(have)}/{len(seeds)}  thiếu seed: {sorted(set(seeds) - set(have)) or '-'}")
    print("Kết quả đã có:")
    for f in sorted(glob.glob(str(res_dir / "*.json"))):
        if Path(f).name.startswith(_NON_RESULT_PREFIXES):
            continue
        s = _load_result_file(Path(f))
        ok = [r["combination"] for r in s["results"] if r.get("complete")]
        evals = sorted({r.get("config", {}).get("n_eval_shap") for r in s["results"] if r.get("complete")} - {None})
        print(f"  {Path(f).stem}: {len(ok)} tổ hợp đủ seed, {len(s['errors'])} lỗi, SHAP trên {evals or '?'} dòng")


def cmd_merge(a) -> None:
    res_dir, _, summary_json, summary_csv = _paths(a)
    per_file = {}
    for f in sorted(glob.glob(str(res_dir / "*.json"))):
        if Path(f).name.startswith(_NON_RESULT_PREFIXES):
            continue
        per_file[Path(f).name] = _load_result_file(Path(f))["results"]
    if not per_file:
        sys.exit(f"Không có file kết quả nào trong {res_dir}")
    merged = rs.merge_rerun_results(per_file)
    for p in merged["problems"]:
        logger.warning(p)
    rs.atomic_write_json(summary_json, {"design": rs.RERUN_DESIGN, "dataset": a.dataset, **merged})
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "technique", "cies", "mean_spearman", "pr_auc_mean", "pr_auc_std", "pr_auc_min", "pr_auc_max",
                    "n_seeds_ok", "identical_shap", "device"])
        for r in merged["results"]:
            c, p = r["cies_metrics"], r["pr_auc_summary"]
            w.writerow([r["model_name"], r["imbalance_technique"], round(c["cies_score"], 6), round(c["mean_spearman"], 6),
                        round(p["mean"], 6), round(p["std"], 6), round(p["min"], 6), round(p["max"], 6), p["n"],
                        r["diagnostics"]["identical_mean_abs_shap"], r["config"]["device"]])
    print(f"Đã gộp {len(merged['results'])} tổ hợp, {len(merged['problems'])} cảnh báo → {summary_json.name}, {summary_csv.name}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pilot", help="Đo thời gian để ước lượng")
    _add_common(p); _add_model_args(p)
    p.add_argument("--pilot-rows", type=int, default=2_000, help="Số dòng test dùng đo tốc độ SHAP")
    p.add_argument("--shap-rows", type=_shap_rows_arg, default=None,
                   help="Tập SHAP để ước giờ, như ở train (mặc định theo dataset: Sparkov 20%% test, ULB toàn bộ test)")
    p.add_argument("--allow-default-params", action="store_true")
    p.set_defaults(fn=cmd_pilot)

    p = sub.add_parser("cache", help="Tạo cache dữ liệu resample theo seed")
    _add_common(p)
    p.set_defaults(fn=cmd_cache)

    p = sub.add_parser("train", help="Chạy tổ hợp model × kỹ thuật trọn các seed")
    _add_common(p); _add_model_args(p)
    p.add_argument("--shap-rows", type=_shap_rows_arg, default=None,
                   help="Mặc định theo dataset (config.RERUN_SHAP_TEST_FRACTION): Sparkov = mẫu phân tầng cố định 20%% test "
                        "(111.144 dòng), ULB = toàn bộ test. 'all' = ép toàn bộ test; số nguyên = ép đúng số dòng đó "
                        "(mọi tổ hợp của một dataset phải cùng mẫu)")
    p.add_argument("--allow-default-params", action="store_true")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("status", help="Xem tiến độ")
    _add_common(p)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("merge", help="Gộp các file kết quả thành bảng tổng hợp")
    p.add_argument("--dataset", default="sparkov", choices=rs.DATASETS)
    p.set_defaults(fn=cmd_merge)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":  # bắt buộc cho spawn trên Windows/macOS
    main()
