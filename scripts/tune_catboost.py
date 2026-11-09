"""
tune_catboost.py - Tune lại RIÊNG CatBoost trên CPU với depth bị chặn (mặc định 4..7), cho Sparkov hoặc ULB.

Vì sao: SHAP chính xác (Regular) của CatBoost tốn ~ số_cây * 2^depth * depth^2, nên depth 11 làm SHAP 100% test rất chậm. Chặn depth <= 7
ngay trong không gian tìm kiếm Optuna vẫn tối ưu PR-AUC (CV 3 fold theo thời gian, như mọi model khác), và không phải "chọn lại
trial cũ". Tune và train phải CÙNG thiết bị (AGENTS.md), mà CatBoost kết quả chính chạy CPU, nên script này ÉP CPU
(CUDA_VISIBLE_DEVICES=-1) và từ chối chạy nếu build_model vẫn thấy GPU (kể cả trên máy có A100: dùng biến CIES_FORCE_CPU).

    python scripts/tune_catboost.py --dataset sparkov            # 50 trial, resume được (checkpoint sqlite)
    python scripts/tune_catboost.py --dataset ulb
    python scripts/tune_catboost.py --dataset sparkov --session-budget 20000   # dừng mềm sau ~20000 giây, chạy lại để tiếp tục
    python scripts/tune_catboost.py --dataset sparkov --dry-run  # chỉ in cấu hình, không tune

Ra: <checkpoint-dir>/catboost.db (study) và <checkpoint-dir>/best_params_catboost_dN.json. KHÔNG đụng best_params*.json chính. Bước sau:
    python scripts/select_catboost_params.py --dataset sparkov --db <checkpoint-dir>/catboost.db            # xem bảng
    python scripts/select_catboost_params.py --dataset sparkov --db <checkpoint-dir>/catboost.db --write    # ghi
"""

import argparse
import os
import sys
from pathlib import Path

# Ép CPU TRƯỚC mọi import có thể nạp CUDA/CatBoost. Tune chạy trong tiến trình spawn (run_isolated) nên chỉ biến MÔI TRƯỜNG
# mới tới được nó: CIES_FORCE_CPU=1 tắt nhánh task_type=GPU của build_model (nvidia-smi vẫn thấy GPU nên cần cờ này).
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CIES_FORCE_CPU"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    from src.config import (BEST_PARAMS_FILE, BEST_PARAMS_ULB_FILE, N_TUNE_TRIALS, PROCESSED_DATA_DIR, RESULTS_DIR,
                            TARGET_COL, ULB_FEATURE_COLS, ULB_TARGET_COL)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["sparkov", "ulb"], required=True)
    ap.add_argument("--depth-min", type=int, default=4)
    ap.add_argument("--depth-max", type=int, default=7)
    ap.add_argument("--n-trials", type=int, default=N_TUNE_TRIALS, help=f"mặc định {N_TUNE_TRIALS} như các model khác")
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="mặc định results/tuning_checkpoints_catboost_cpu_d<max>[_ulb]")
    ap.add_argument("--session-budget", type=float, default=None, help="giây tối đa cho lượt chạy này (mềm)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not (1 <= a.depth_min <= a.depth_max <= 16):
        ap.error("cần 1 <= depth-min <= depth-max <= 16")

    suffix = "" if a.dataset == "sparkov" else "_ulb"
    ckpt = a.checkpoint_dir or RESULTS_DIR / f"tuning_checkpoints_catboost_cpu_d{a.depth_max}{suffix}"
    out_name = f"best_params_catboost_d{a.depth_max}{suffix}.json"
    main_file = BEST_PARAMS_FILE if a.dataset == "sparkov" else BEST_PARAMS_ULB_FILE
    if ckpt.resolve() == (RESULTS_DIR / ("tuning_checkpoints" + suffix)).resolve():
        ap.error("không dùng thư mục checkpoint của study cũ (không gian tìm kiếm khác); chọn thư mục mới")

    import pandas as pd

    from src.models.train import _has_gpu
    from src.models.tune import tune_all_models

    if _has_gpu():
        print("[LỖI] build_model vẫn thấy GPU dù đã đặt CIES_FORCE_CPU=1; từ chối tune vì CatBoost sẽ chạy GPU và lệch thiết bị với train. "
              "Kiểm tra `_has_gpu()` trong src/models/train.py.")
        return 2

    if a.dataset == "sparkov":
        path, target, cols = PROCESSED_DATA_DIR / "train_raw.parquet", TARGET_COL, None
    else:
        path, target, cols = PROCESSED_DATA_DIR / "ulb_train.parquet", ULB_TARGET_COL, ULB_FEATURE_COLS + [ULB_TARGET_COL]
    if not path.exists():
        print(f"[LỖI] Thiếu {path}. " + ("Chạy scripts/prepare_ulb.py trước." if a.dataset == "ulb" else ""))
        return 2
    df = pd.read_parquet(path)
    if cols:
        df = df[cols]
    print(f"Dataset {a.dataset}: {len(df):,} dòng train, {int(df[target].sum())} fraud | depth {a.depth_min}..{a.depth_max} | "
          f"{a.n_trials} trial x 3 fold | CPU | checkpoint {ckpt}")
    if a.dataset == "ulb" and len(df) != 226_981:
        print(f"[CẢNH BÁO] ULB train có {len(df):,} dòng, số chuẩn sau bỏ trùng là 226.981 — đã chạy prepare_ulb.py chưa?")
    if a.dry_run:
        print(f"(dry-run) sẽ ghi {ckpt / out_name}; không đụng {main_file.name}")
        return 0

    tune_all_models(
        df, target_col=target, models=["catboost"], n_trials=a.n_trials, n_splits=3,
        output_dir=ckpt, filename=out_name, checkpoint_dir=ckpt, session_budget=a.session_budget,
        catboost_depth=(a.depth_min, a.depth_max),
    )
    db = ckpt / "catboost.db"
    print(f"\nXong lượt này. Nếu đủ {a.n_trials} trial, chọn tham số bằng:\n"
          f"  python scripts/select_catboost_params.py --dataset {a.dataset} --db {db}\n"
          f"  python scripts/select_catboost_params.py --dataset {a.dataset} --db {db} --write")
    return 0


if __name__ == "__main__":  # bắt buộc: run_isolated dùng spawn
    raise SystemExit(main())
