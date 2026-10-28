"""
gpu_smote_enn.py — SMOTE-ENN trên Sparkov với bước ENN chạy trên GPU; chỉ dùng sau khi đã kiểm tra TRÙNG bản CPU.

Các bước (cần GPU NVIDIA và torch bản CUDA):
    python scripts/gpu_smote_enn.py bench             # đo tốc độ trên 20.000 truy vấn → ước thời gian mỗi seed (1–3 phút)
    python scripts/gpu_smote_enn.py check --seed 0    # chạy đủ seed 0 trên GPU rồi so với file CPU seed 0 (đối chứng)
    python scripts/gpu_smote_enn.py cache --seeds 1-9 # ghi cache các seed còn lại; TỪ CHỐI chạy nếu check chưa đạt

`check` cần file CPU đối chứng: data/processed/resampled_rerun/seed00/train_encoded_smote_enn.parquet (tạo bởi
`run_rerun.py cache` trên CPU), đúng định dạng và dấu vân tay mà train đọc. Kết quả check lưu ở
results/rerun/gpu_enn_check.json — `cache` đọc file này làm điều kiện.

Mặc định float64 (khớp bản CPU của scikit-learn). float32 nhanh hơn nhưng có thể lệch ở các điểm hoà khoảng cách
(đã thử trên dữ liệu giống Sparkov: float64 trùng 100%, float32 lệch vài dòng) — chỉ dùng nếu `check` vẫn trùng.

Mã thoát: 0 = trùng khít / xong, 1 = KHÁC bản CPU, 2 = không chạy được (không CUDA, thiếu file đối chứng, chưa check...).
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.config import PROCESSED_DATA_DIR, RERUN_RESULTS_DIR, TARGET_COL  # noqa: E402
from src.explainability.rerun_stability import (  # noqa: E402
    dataset_cache_root, load_encoded_data, parse_seeds, seed_cache_dir,
)
from src.imbalance import gpu_enn  # noqa: E402
from src.imbalance.resamplers import FINGERPRINT_KEY, _resample_fingerprint  # noqa: E402

logger = logging.getLogger("gpu_smote_enn")
TECHNIQUE = "smote_enn"
CACHE_NAME = f"train_encoded_{TECHNIQUE}.parquet"
REPORT_FILE = RERUN_RESULTS_DIR / "gpu_enn_check.json"
BENCH_FILE = RERUN_RESULTS_DIR / "gpu_enn_bench.json"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    p.add_argument("--cache-root", type=Path, default=None,
                   help="Gốc cache resample theo seed (mặc định data/processed/resampled_rerun)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="auto = cuda, báo lỗi nếu không có. cpu chỉ để thử code")
    p.add_argument("--dtype", default="float64", choices=list(gpu_enn.DTYPES))
    p.add_argument("--q-chunk", type=int, default=1024, help="Số dòng truy vấn mỗi khối (tự giảm khi hết VRAM)")
    p.add_argument("--db-chunk", type=int, default=262_144, help="Số dòng cơ sở dữ liệu mỗi khối")


def _cache_path(a, seed: int) -> Path:
    root = a.cache_root if a.cache_root is not None else dataset_cache_root("sparkov")
    return seed_cache_dir(root, seed) / CACHE_NAME


def _knn_kwargs(a, progress=None) -> dict:
    return dict(device=a.device, dtype=a.dtype, q_chunk=a.q_chunk, db_chunk=a.db_chunk, progress=progress)


def _progress_printer(label: str, every: float = 60.0):
    state = {"last": time.time()}

    def cb(done: int, total: int, elapsed: float) -> None:
        now = time.time()
        if done >= total or now - state["last"] >= every:
            state["last"] = now
            eta = elapsed / max(done, 1) * (total - done)
            logger.info(f"  {label}: {100 * done / total:5.1f}%  đã {elapsed / 60:5.1f} phút, còn ~{eta / 60:5.1f} phút")

    return cb


def _fingerprint_of(path: Path):
    import pyarrow.parquet as pq

    return ((pq.read_schema(path).metadata or {}).get(FINGERPRINT_KEY) or b"").decode() or None


def _row_hashes(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    import pandas as pd

    df = pd.DataFrame(X)
    df["__y"] = y
    return pd.util.hash_pandas_object(df, index=False).to_numpy()


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _env_info() -> dict:
    import imblearn
    import sklearn
    import torch

    return {"torch": torch.__version__, "sklearn": sklearn.__version__,
            "imblearn": imblearn.__version__, "numpy": np.__version__}


def _need_device(a) -> int:
    """0 nếu thiết bị dùng được; 2 nếu không (in lý do)."""
    try:
        dev = gpu_enn.resolve_device(a.device)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Không dùng được GPU: {e}")
        return 2
    logger.info(f"Thiết bị: {gpu_enn.device_description(dev)} | dtype {a.dtype}")
    return 0


# ===== bench =====

def cmd_bench(a) -> int:
    from imblearn.over_sampling import SMOTE

    if _need_device(a):
        return 2
    X, y, _, _, _, _ = load_encoded_data(a.data_dir, "sparkov")
    t0 = time.time()
    X_s, y_s = SMOTE(sampling_strategy="auto", random_state=a.seed).fit_resample(X, y)
    t_smote = time.time() - t0
    n = len(y_s)
    logger.info(f"SMOTE (CPU) seed {a.seed}: {t_smote:.0f}s, {len(y)} → {n} dòng")
    gpu_enn.knn_indices(X_s, 4, start=0, stop=256, **{**_knn_kwargs(a), "progress": None})  # khởi động CUDA
    nq = min(a.queries, n)
    t0 = time.time()
    gpu_enn.knn_indices(X_s, 4, start=0, stop=nq, **_knn_kwargs(a))
    t = time.time() - t0
    est = t / nq * n
    logger.info(f"{nq} truy vấn mất {t:.1f}s → ước ENN cho {n} dòng ≈ {est / 60:.1f} phút/seed "
                f"(+ SMOTE {t_smote:.0f}s). Bản CPU đã đo: ~145 phút/seed.")
    _write_json(BENCH_FILE, {
        "seed": a.seed, "device": a.device, "dtype": a.dtype, "n_rows_after_smote": int(n), "queries": int(nq),
        "seconds_queries": round(t, 1), "seconds_smote_cpu": round(t_smote, 1), "est_minutes_enn_per_seed": round(est / 60, 1),
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "env": _env_info(),
    })
    return 0


# ===== check =====

def _previous_check_passed(a) -> bool:
    """Lần check gần nhất đã TRÙNG bản CPU cho đúng seed/dtype/thiết bị này (dùng khi chạy lại sau khi bị ngắt)."""
    if not REPORT_FILE.exists():
        return False
    try:
        rep = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
    except ValueError:
        return False
    ok = (bool(rep.get("identical")) and rep.get("seed") == a.seed and rep.get("dtype") == a.dtype
          and rep.get("device") == a.device and bool(rep.get("fingerprint_input_match")))
    if ok:
        logger.info(f"KẾT LUẬN: đã có lần check TRÙNG KHÍT bản CPU (seed {a.seed}, {a.dtype}) lúc {rep.get('checked_at')} → "
                    f"không chạy lại. Xoá {REPORT_FILE.name} nếu muốn check lại.")
    return ok


def cmd_check(a) -> int:
    if a.skip_if_passed and _previous_check_passed(a):
        return 0
    if _need_device(a):
        return 2
    ref_path = _cache_path(a, a.seed)
    if not ref_path.exists():
        logger.error(f"Thiếu file CPU đối chứng: {ref_path}. Hãy tạo nó bằng `run_rerun.py cache --techniques smote_enn --seeds 0` trên CPU.")
        return 2
    import pandas as pd

    X, y, _, _, names, onehot_groups = load_encoded_data(a.data_dir, "sparkov")
    fp_now = _resample_fingerprint(TECHNIQUE, X, y, a.seed, onehot_groups)
    fp_ref = _fingerprint_of(ref_path)
    ref = pd.read_parquet(ref_path)
    X_ref, y_ref = ref[names].to_numpy(dtype=np.float64), ref[TARGET_COL].to_numpy()

    t0 = time.time()
    X_g, y_g, _ = gpu_enn.smote_enn_gpu(X, y, a.seed, onehot_groups, **_knn_kwargs(a, _progress_printer("ENN GPU")))
    seconds = time.time() - t0

    identical = bool(X_g.shape == X_ref.shape and np.array_equal(y_g, y_ref) and np.array_equal(X_g, X_ref))
    report = {
        "seed": a.seed, "identical": identical, "dtype": a.dtype, "device": a.device, "seconds_total": round(seconds, 1),
        "n_rows_cpu": int(len(y_ref)), "n_rows_gpu": int(len(y_g)),
        "n_fraud_cpu": int((y_ref == 1).sum()), "n_fraud_gpu": int((y_g == 1).sum()),
        "fingerprint_input_match": fp_now == fp_ref, "fingerprint_cpu_file": fp_ref, "fingerprint_now": fp_now,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "env": _env_info(),
    }
    if not identical:
        h_ref, h_gpu = _row_hashes(X_ref, y_ref), _row_hashes(X_g, y_g)
        report["n_rows_only_in_cpu"] = int(np.setdiff1d(h_ref, h_gpu).size)
        report["n_rows_only_in_gpu"] = int(np.setdiff1d(h_gpu, h_ref).size)
    if not report["fingerprint_input_match"]:
        logger.warning("Dấu vân tay đầu vào KHÁC file CPU (dữ liệu hoặc version imblearn/scikit-learn không giống lúc tạo file CPU) "
                       "→ so sánh dưới đây ít ý nghĩa; cài đúng requirements-lock.txt rồi chạy lại.")
    _write_json(REPORT_FILE, report)

    logger.info(f"GPU: {report['n_rows_gpu']} dòng ({report['n_fraud_gpu']} fraud) | CPU: {report['n_rows_cpu']} dòng "
                f"({report['n_fraud_cpu']} fraud) | {seconds / 60:.1f} phút")
    if identical:
        logger.info(f"KẾT LUẬN: TRÙNG KHÍT bản CPU (seed {a.seed}, {a.dtype}). Có thể chạy: "
                    f"python scripts/gpu_smote_enn.py cache --seeds 1-9")
        return 0
    logger.error(f"KẾT LUẬN: KHÁC bản CPU — {report.get('n_rows_only_in_cpu')} dòng chỉ có ở CPU, "
                 f"{report.get('n_rows_only_in_gpu')} dòng chỉ có ở GPU. Chưa dùng GPU; dán dòng này cho Claude. "
                 f"Báo cáo: {REPORT_FILE}")
    return 1


# ===== cache =====

def _guard_ok(a) -> bool:
    if a.skip_guard:
        logger.warning("--skip-guard: bỏ qua điều kiện check đạt.")
        return True
    if not REPORT_FILE.exists():
        logger.error(f"Chưa có {REPORT_FILE}. Chạy `check --seed 0` trước.")
        return False
    rep = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
    if not rep.get("identical"):
        logger.error("Lần check gần nhất KHÔNG trùng bản CPU → không ghi cache GPU.")
        return False
    if rep.get("dtype") != a.dtype:
        logger.error(f"Lần check đạt với dtype {rep.get('dtype')} nhưng đang chạy {a.dtype}. Dùng cùng dtype hoặc check lại.")
        return False
    if not rep.get("fingerprint_input_match"):
        logger.error("Lần check đạt nhưng dấu vân tay đầu vào không khớp file CPU → check lại cho chắc.")
        return False
    return True


def cmd_cache(a) -> int:
    if _need_device(a):
        return 2
    if not _guard_ok(a):
        return 2
    seeds = parse_seeds(a.seeds)
    X, y, _, _, names, onehot_groups = load_encoded_data(a.data_dir, "sparkov")
    done = 0
    for seed in seeds:
        path = _cache_path(a, seed)
        fp = _resample_fingerprint(TECHNIQUE, X, y, seed, onehot_groups)
        if path.exists():
            if _fingerprint_of(path) == fp:
                logger.info(f"Seed {seed}: đã có, dấu vân tay khớp → bỏ qua")
                continue
            if not a.force:
                logger.error(f"Seed {seed}: file đã có nhưng dấu vân tay KHÁC ({path}). Thêm --force để thay (bản cũ giữ thành .backup).")
                return 2
            backup = path.with_name(path.stem + ".backup.parquet")
            os.replace(path, backup)
            logger.warning(f"Seed {seed}: đã chuyển bản cũ sang {backup.name}")
        t0 = time.time()
        logger.info(f"Seed {seed}: SMOTE (CPU) + ENN (GPU) ...")
        X_r, y_r, _ = gpu_enn.smote_enn_gpu(X, y, seed, onehot_groups, **_knn_kwargs(a, _progress_printer(f"seed {seed}")))
        gpu_enn.write_resample_cache(path, X_r, y_r, names, fp, TARGET_COL)
        logger.info(f"Seed {seed}: xong sau {(time.time() - t0) / 60:.1f} phút → {len(y_r)} dòng ({int((y_r == 1).sum())} fraud)")
        done += 1
    logger.info(f"Xong {done} seed mới; cache nằm ở data\\processed\\resampled_rerun\\seedNN\\{CACHE_NAME}.")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("bench", help="Đo tốc độ GPU, ước thời gian mỗi seed")
    _add_common(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--queries", type=int, default=20_000)
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("check", help="Chạy đủ 1 seed trên GPU và so với file CPU")
    _add_common(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-if-passed", action="store_true",
                   help="Nếu lần check gần nhất của seed/dtype/thiết bị này đã TRÙNG thì không chạy lại (chạy lại sau khi bị ngắt)")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("cache", help="Ghi cache SMOTE-ENN các seed (chỉ khi check đạt)")
    _add_common(p)
    p.add_argument("--seeds", default="1-9")
    p.add_argument("--force", action="store_true", help="Thay file đã có nhưng dấu vân tay khác (giữ bản cũ thành .backup)")
    p.add_argument("--skip-guard", action="store_true", help="Bỏ điều kiện check đạt (chỉ khi đã bàn với Claude/thầy)")
    p.set_defaults(fn=cmd_cache)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
