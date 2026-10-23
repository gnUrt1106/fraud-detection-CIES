"""
rerun_stability.py — Độ ổn định của giải thích qua các LẦN CHẠY LẠI (chỉ đổi seed), không bootstrap.

Khác `cies.run_cies_experiment` (bootstrap lấy mẫu lại train + fit lại encoding mỗi run), chế độ này theo
yêu cầu của giảng viên:
    - Train trên TOÀN BỘ train (dữ liệu và phân phối giữ nguyên); encoding fit MỘT lần (file
      train_encoded.parquet); mỗi run chỉ đổi seed (resampler + model) — nguồn dao động duy nhất là
      tính ngẫu nhiên của thuật toán ("các lần hội tụ khác nhau").
    - PR-AUC (toàn bộ test) và SHAP lấy từ CÙNG model của mỗi run — không tách nhánh hiệu năng / nhánh ổn định.
    - SHAP tính trên TOÀN BỘ test (phân phối thật), chia đợt, chỉ giữ mean|SHAP| từng feature.
    - Tham số model giữ nguyên từ Optuna (results/best_params.json), không tune lại.
    - CIES = công thức cũ (compute_stability_metric), trên n_seeds bảng hạng (10 seed = 45 cặp).

Lưu ý khi đọc kết quả: model tất định (vd. Logistic Regression + class_weighting) cho 10 model GIỐNG HỆT
nhau → CIES = 1. Đó không phải lỗi; `diagnostics.identical_mean_abs_shap` đánh dấu trường hợp này.

Mỗi tổ hợp (model × kỹ thuật) PHẢI chạy trọn mọi seed trên 1 máy/thiết bị: khi đo dao động theo seed,
sai khác số học giữa CPU/GPU hay giữa kiến trúc cùng cỡ với thứ cần đo. Dữ liệu đã resample thì dùng
chung được giữa các máy (cache theo (kỹ thuật, seed), không phụ thuộc model).

Không import torch/xgboost ở top-level (xem src/utils/isolation.py): hàm chạy tổ hợp phải được gọi qua
run_isolated.
"""

import hashlib
import json
import logging
import os
import platform
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.config import SEED, SHAP_BACKGROUND_N

logger = logging.getLogger(__name__)

RERUN_DESIGN = "seed_rerun_full_train"

_VERSION_PACKAGES = (
    "numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost", "catboost", "shap", "torch", "pyarrow",
)


# ===== Tiện ích =====

def parse_seeds(spec: Any) -> List[int]:
    """'0-9' → 0..9; '0,2,5' → [0,2,5]; '3' → [3]; list/int giữ nguyên. Không cho trùng seed."""
    if isinstance(spec, int):
        seeds = [spec]
    elif isinstance(spec, (list, tuple)):
        seeds = [int(s) for s in spec]
    else:
        seeds = []
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                lo, hi = int(lo), int(hi)
                if hi < lo:
                    raise ValueError(f"Khoảng seed ngược: '{part}'")
                seeds.extend(range(lo, hi + 1))
            else:
                seeds.append(int(part))
    if not seeds:
        raise ValueError(f"Không đọc được seed từ '{spec}'")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seed bị trùng: {seeds}")
    return seeds


def stratified_eval_indices(y: np.ndarray, n: Optional[int], seed: int = SEED) -> np.ndarray:
    """
    Chỉ số tập eval cố định. n=None (hoặc >= số dòng) → TOÀN BỘ tập. Ngược lại lấy mẫu PHÂN TẦNG giữ
    nguyên tỷ lệ fraud (ít nhất 1 dòng mỗi lớp), xác định theo `seed` — cùng 1 tập cho mọi model/run.
    """
    y = np.asarray(y)
    if n is None or n >= len(y):
        return np.arange(len(y))
    if n < 2:
        raise ValueError("shap_rows phải >= 2 (cần cả hai lớp)")
    rng = np.random.RandomState(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_pos = int(min(len(pos), max(1, round(n * len(pos) / len(y)))))
    n_neg = int(min(len(neg), n - n_pos))
    idx = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])
    return np.sort(idx)


def data_fingerprint(*arrays: np.ndarray) -> str:
    """Dấu vân tay nội dung dữ liệu — để chắc chắn mọi máy chạy trên CÙNG file đã tiền xử lý."""
    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(repr((a.shape, str(a.dtype))).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()[:16]


def environment_info() -> Dict[str, Any]:
    """Thông tin máy + phiên bản thư viện (importlib.metadata — KHÔNG import torch/xgboost vào process này)."""
    versions = {}
    for pkg in _VERSION_PACKAGES:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine_arch": platform.machine(),
        "cpu_count": os.cpu_count(),
        "versions": versions,
    }


def seed_cache_dir(cache_root: Path, seed: int) -> Path:
    """Thư mục cache resample của 1 seed. (apply_imbalance_cached đặt tên file theo kỹ thuật, KHÔNG theo seed.)"""
    return Path(cache_root) / f"seed{int(seed):02d}"


def pick_background(X_train: np.ndarray, model_name: str, n: int = SHAP_BACKGROUND_N, seed: int = SEED) -> np.ndarray:
    """
    Nền SHAP CỐ ĐỊNH cho mọi seed và mọi kỹ thuật (cùng quy ước cies.run_cies_experiment): LR dùng toàn bộ
    train (LinearExplainer chỉ cần trung bình), model khác dùng mẫu `n` dòng. TreeExplainer không dùng nền.
    """
    if model_name == "logistic_regression":
        return X_train
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(X_train), size=min(n, len(X_train)), replace=False))
    return X_train[idx]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Không serialize được {type(obj).__name__}")


def atomic_write_json(path: Path, data: Any) -> None:
    """Ghi JSON atomic (file tạm + os.replace). Dùng cho file RIÊNG của 1 máy/1 tiến trình — không khoá chéo tiến trình."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, default=_json_default, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def summarize_values(values: Sequence[float]) -> Dict[str, float]:
    """Trung bình, độ lệch chuẩn mẫu (ddof=1), min, max, số giá trị."""
    arr = np.asarray(list(values), dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(len(arr)),
    }


# ===== Cache resample theo seed (làm được trên máy khác rồi copy sang) =====

def build_resample_cache(
    technique: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    feature_names: Sequence[str],
    onehot_groups: Optional[Sequence[Sequence[int]]],
    cache_root: Path,
) -> Dict[str, Any]:
    """
    Tính (hoặc đọc lại) dữ liệu train đã resample của (kỹ thuật, seed) và lưu parquet. Cùng seed thì mọi model dùng
    chung. SMOTE-ENN trên toàn bộ train ~145 phút/lần (đã đo) — đây là việc nên chia cho nhiều máy.
    class_weighting không resample nên không có gì để cache.
    """
    from src.imbalance.resamplers import apply_imbalance_cached

    if technique == "class_weighting":
        return {"technique": technique, "seed": seed, "skipped": "class_weighting không resample"}
    t0 = time.time()
    X_res, y_res, _ = apply_imbalance_cached(
        technique, X_train, y_train, seed=seed, onehot_groups=onehot_groups,
        cache_dir=seed_cache_dir(cache_root, seed), feature_names=list(feature_names),
    )
    return {
        "technique": technique,
        "seed": seed,
        "seconds": round(time.time() - t0, 1),
        "n_after": int(len(y_res)),
        "n_fraud_after": int(np.sum(y_res == 1)),
    }


# ===== Chạy 1 tổ hợp (model × kỹ thuật) trọn các seed =====

def run_rerun_combo(
    model_name: str,
    technique: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    onehot_groups: Optional[Sequence[Sequence[int]]],
    seeds: Sequence[int],
    cache_root: Optional[Path] = None,
    params: Optional[Dict[str, Any]] = None,
    shap_rows: Optional[int] = None,
    shap_batch: int = 10_000,
    shap_backend: str = "auto",
    background_n: int = SHAP_BACKGROUND_N,
    force_cpu: bool = False,
    machine: str = "",
) -> Dict[str, Any]:
    """
    Chạy 1 tổ hợp: với mỗi seed → (resample, đọc cache nếu có) → train → PR-AUC trên TOÀN BỘ test → mean|SHAP| trên
    tập eval (mặc định TOÀN BỘ test). Rồi CIES từ các bảng hạng. Hàm top-level, picklable — gọi qua run_isolated.

    Args:
        X_train, y_train: train đã encode, CHƯA resample (cả 1,3 triệu dòng).
        X_test, y_test: test đã encode (không resample).
        seeds: danh sách seed (mặc định của dự án: 0..9; seed i dùng cho resampler và model).
        cache_root: gốc cache resample theo seed; None = không cache.
        params: tham số model; None = build_model tự nạp results/best_params.json (Optuna, không tune lại).
        shap_rows: None = SHAP trên toàn bộ test; số nguyên = mẫu phân tầng cố định cỡ đó (chỉ để chạy thử/đo giờ).
        shap_batch: số dòng mỗi đợt SHAP (chỉ ảnh hưởng bộ nhớ/tốc độ, không ảnh hưởng kết quả).
        shap_backend: "auto" | "shap" | "native" (xem shap_utils._make_shap_batch_fn).
        force_cpu: ép chạy CPU (XGBoost/CatBoost/ANN) kể cả khi máy có GPU NVIDIA.
        machine: nhãn máy, chỉ để ghi vào kết quả.
    """
    if force_cpu:
        # Phải đặt TRƯỚC khi torch/xgboost/catboost được nạp (hàm này chạy trong subprocess spawn riêng).
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import src.models.train as train_mod

        train_mod._has_gpu_cache = False

    from src.evaluation.metrics import evaluate_model
    from src.explainability.cies import compute_stability_metric
    from src.explainability.shap_utils import compute_mean_abs_shap
    from src.imbalance.resamplers import apply_imbalance_cached
    from src.models.train import _has_gpu, build_model, predict_proba, train_model

    seeds = [int(s) for s in seeds]
    combination = f"{model_name}__{technique}"
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    eval_idx = stratified_eval_indices(y_test, shap_rows)
    X_eval = X_test[eval_idx]
    y_eval = y_test[eval_idx]
    X_background = pick_background(X_train, model_name, background_n)

    logger.info(f"=== Rerun: {combination} ({len(seeds)} seed, eval SHAP {len(X_eval)} dòng) ===")

    run_logs: List[Dict[str, Any]] = []
    mean_abs_list: List[np.ndarray] = []
    explainers = set()

    for seed in seeds:
        t_run = time.time()
        try:
            t = time.time()
            cache_dir = seed_cache_dir(cache_root, seed) if cache_root is not None else None
            X_res, y_res, class_weights = apply_imbalance_cached(
                technique, X_train, y_train, seed=seed, onehot_groups=onehot_groups,
                cache_dir=cache_dir, feature_names=list(feature_names),
            )
            sec_resample = time.time() - t

            t = time.time()
            model = build_model(
                model_name, input_dim=X_res.shape[1], class_weights=class_weights, params=params, seed=seed,
            )
            trained = train_model(model, X_res, y_res, model_name=model_name)
            sec_train = time.time() - t

            t = time.time()
            metrics = evaluate_model(y_test, predict_proba(trained, X_test))
            sec_predict = time.time() - t

            t = time.time()
            mean_abs, explainer_used = compute_mean_abs_shap(
                trained, model_name, X_eval, X_background=X_background,
                batch_size=shap_batch, backend=shap_backend,
            )
            sec_shap = time.time() - t
            # Model hằng số → SHAP toàn 0 → mọi hạng hoà → CIES = 1 GIẢ. Loại run này (cùng quy ước cies.py).
            if np.allclose(mean_abs, 0.0):
                raise ValueError("SHAP toàn 0 — model không học được (dự đoán hằng số), bỏ run này")

            explainers.add(explainer_used)
            mean_abs_list.append(mean_abs)
            run_logs.append({
                "run_idx": len(run_logs),
                "seed": seed,
                "status": "success",
                "n_train_after_imbalance": int(len(y_res)),
                "metrics": metrics,
                "mean_abs_shap": mean_abs.tolist(),
                "explainer": explainer_used,
                "seconds": {
                    "resample": round(sec_resample, 1), "train": round(sec_train, 1),
                    "predict": round(sec_predict, 1), "shap": round(sec_shap, 1),
                    "total": round(time.time() - t_run, 1),
                },
            })
            logger.info(f"  seed {seed}: PR-AUC {metrics['pr_auc']:.4f} ({time.time() - t_run:.0f}s)")
        except Exception as e:  # noqa: BLE001 — một seed hỏng không làm mất các seed còn lại
            logger.error(f"  seed {seed} failed: {e}")
            run_logs.append({"seed": seed, "status": "failed", "error": f"{type(e).__name__}: {e}"})

    n_ok = len(mean_abs_list)
    if n_ok < 2:
        errors = [r["error"] for r in run_logs if r["status"] == "failed"]
        raise RuntimeError(
            f"{combination}: chỉ {n_ok}/{len(seeds)} seed thành công, cần ít nhất 2. "
            f"Lỗi đầu tiên: {errors[0] if errors else '(không có)'}"
        )

    # Mỗi mean|SHAP| đóng vai "ma trận SHAP 1 dòng": shap_to_ranks/Spearman trong compute_stability_metric
    # chỉ dùng trung bình |SHAP| theo cột nên kết quả y hệt dùng cả ma trận — công thức CIES giữ nguyên.
    cies_metrics = compute_stability_metric([m[None, :] for m in mean_abs_list])

    first = mean_abs_list[0]
    identical_values = all(np.allclose(m, first, rtol=0.0, atol=1e-12) for m in mean_abs_list[1:])
    pr_aucs = [r["metrics"]["pr_auc"] for r in run_logs if r["status"] == "success"]

    return {
        "design": RERUN_DESIGN,
        "combination": combination,
        "model_name": model_name,
        "imbalance_technique": technique,
        "complete": n_ok == len(seeds),
        "cies_metrics": cies_metrics,
        "pr_auc_summary": summarize_values(pr_aucs),
        "diagnostics": {
            "identical_mean_abs_shap": bool(identical_values),
            "identical_ranks": bool(cies_metrics["cies_score"] >= 1.0 - 1e-12),
            "note": (
                "Các run cho SHAP giống hệt nhau (model tất định theo seed) → CIES = 1 là hệ quả của thiết kế, không phải lỗi."
                if identical_values else ""
            ),
        },
        "feature_names": list(feature_names),
        "config": {
            "seeds": seeds,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_eval_shap": int(len(X_eval)),
            "n_eval_fraud": int(np.sum(y_eval == 1)),
            "shap_rows_setting": shap_rows,
            "shap_batch": shap_batch,
            "shap_backend": shap_backend,
            "explainers": sorted(explainers),
            "background_n": int(len(X_background)),
            "data_fingerprint": data_fingerprint(X_train, y_train, X_test, y_test),
            "device": "cpu (ép)" if force_cpu else ("gpu" if _has_gpu() else "cpu"),
            "machine": machine,
            "params_source": "tham số truyền vào" if params else "results/best_params.json",
            "environment": environment_info(),
        },
        "run_logs": run_logs,
    }


# ===== Chạy thử: đo thời gian từng model để ước lượng tổng =====

def pilot_timing(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    pilot_rows: int = 2_000,
    shap_batch: int = 10_000,
    shap_backend: str = "auto",
    seed: int = 0,
    params: Optional[Dict[str, Any]] = None,
    force_cpu: bool = False,
    n_runs_total: int = 50,
) -> Dict[str, Any]:
    """
    Train 1 lần (class_weighting — không cần cache resample) trên toàn bộ train, đo thời gian train, dự đoán
    toàn bộ test, và SHAP trên `pilot_rows` dòng; suy ra thời gian SHAP toàn bộ test (tuyến tính theo số dòng) và
    tổng giờ cho `n_runs_total` lần train (5 kỹ thuật × 10 seed), CHƯA gồm thời gian resample.
    Hàm top-level — gọi qua run_isolated.
    """
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import src.models.train as train_mod

        train_mod._has_gpu_cache = False

    from src.evaluation.metrics import evaluate_model
    from src.explainability.shap_utils import compute_mean_abs_shap
    from src.imbalance.resamplers import apply_imbalance
    from src.models.train import _has_gpu, build_model, predict_proba, train_model

    X_res, y_res, class_weights = apply_imbalance("class_weighting", X_train, y_train, seed=seed)
    t = time.time()
    model = build_model(model_name, input_dim=X_res.shape[1], class_weights=class_weights, params=params, seed=seed)
    trained = train_model(model, X_res, y_res, model_name=model_name)
    sec_train = time.time() - t

    t = time.time()
    metrics = evaluate_model(y_test, predict_proba(trained, X_test))
    sec_predict = time.time() - t

    # Mẫu thử giữ ~tỷ lệ fraud thật; SHAP tốn gần như tuyến tính theo số dòng nên suy ra được toàn bộ test.
    idx = stratified_eval_indices(y_test, pilot_rows)
    background = pick_background(X_train, model_name)
    t = time.time()
    compute_mean_abs_shap(trained, model_name, X_test[idx], X_background=background,
                          batch_size=shap_batch, backend=shap_backend)
    sec_shap = time.time() - t
    sec_per_row = sec_shap / len(idx)
    est_shap_full = sec_per_row * len(y_test)
    per_run = sec_train + sec_predict + est_shap_full
    return {
        "model": model_name,
        "device": "cpu (ép)" if force_cpu else ("gpu" if _has_gpu() else "cpu"),
        "pr_auc": metrics["pr_auc"],
        "seconds_train": round(sec_train, 1),
        "seconds_predict_full_test": round(sec_predict, 1),
        "pilot_rows": int(len(idx)),
        "seconds_shap_pilot": round(sec_shap, 2),
        "est_seconds_shap_full_test": round(est_shap_full, 0),
        "est_seconds_per_run": round(per_run, 0),
        "est_hours_for_model": round(per_run * n_runs_total / 3600, 1),
        "note": f"Ước lượng cho {n_runs_total} lần train, CHƯA gồm thời gian resample (SMOTE-ENN riêng ~145 phút/lần ở toàn bộ train).",
        "environment": environment_info(),
    }


# ===== Gộp kết quả từ nhiều máy =====

def merge_rerun_results(per_file: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Gộp danh sách kết quả của nhiều file (mỗi máy 1 file). Trả:
        {"results": [...], "problems": [...]}.
    Báo `problems` (không tự chọn bản nào) khi: cùng 1 tổ hợp xuất hiện ở nhiều file; dấu vân tay dữ liệu
    khác nhau giữa các tổ hợp; tổ hợp chưa đủ seed (`complete` = False).
    """
    problems: List[str] = []
    seen: Dict[str, str] = {}
    results: List[Dict[str, Any]] = []
    fingerprints: Dict[str, List[str]] = {}
    for fname, items in per_file.items():
        for r in items:
            if "cies_metrics" not in r:
                continue  # bản ghi lỗi
            combo = r["combination"]
            if combo in seen:
                problems.append(f"Tổ hợp '{combo}' có ở cả '{seen[combo]}' và '{fname}' — giữ MỘT bản rồi gộp lại.")
                continue
            seen[combo] = fname
            if not r.get("complete", False):
                problems.append(f"Tổ hợp '{combo}' ({fname}) chưa đủ seed thành công.")
            fp = r.get("config", {}).get("data_fingerprint")
            fingerprints.setdefault(fp, []).append(combo)
            results.append(r)
    if len(fingerprints) > 1:
        detail = "; ".join(f"{fp}: {len(c)} tổ hợp" for fp, c in fingerprints.items())
        problems.append(f"Dấu vân tay dữ liệu KHÁC nhau giữa các tổ hợp ({detail}) — các máy không dùng cùng file đã tiền xử lý.")
    from src.explainability.cies import combo_order_key

    results.sort(key=lambda r: combo_order_key(r["model_name"], r["imbalance_technique"]))
    return {"results": results, "problems": problems}


# ===== Wrapper đọc dữ liệu TRONG process con (tránh pickle dữ liệu lớn qua spawn) =====
# Hai dataset được đối xử NGANG NHAU (cùng thiết kế, cùng công thức, cùng số seed); chỉ khác nguồn dữ liệu,
# file tham số Optuna riêng và thư mục cache/kết quả riêng.

DATASETS = ("sparkov", "ulb")


def dataset_cache_root(dataset: str, cache_root: Optional[Path] = None) -> Path:
    """Gốc cache resample của dataset (Sparkov giữ nguyên đường dẫn cũ; ULB nằm trong thư mục con `ulb`)."""
    from src.config import RESAMPLE_CACHE_RERUN_DIR

    base = Path(cache_root) if cache_root is not None else Path(RESAMPLE_CACHE_RERUN_DIR)
    return base / "ulb" if dataset == "ulb" and cache_root is None else base


def dataset_results_dir(dataset: str) -> Path:
    from src.config import RERUN_RESULTS_DIR

    return Path(RERUN_RESULTS_DIR) / "ulb" if dataset == "ulb" else Path(RERUN_RESULTS_DIR)


def dataset_params(dataset: str, model_name: str) -> Optional[Dict[str, Any]]:
    """
    Tham số Optuna của model cho dataset. Sparkov: None (build_model tự nạp results/best_params.json).
    ULB: nạp results/best_params_ulb.json và BÁO LỖI nếu thiếu (không âm thầm dùng tham số của Sparkov).
    """
    if dataset != "ulb":
        return None
    from src.config import BEST_PARAMS_ULB_FILE
    from src.models.tune import load_best_params

    params = load_best_params(model_name, filepath=BEST_PARAMS_ULB_FILE)
    if params is None:
        raise RuntimeError(f"{model_name} chưa có tham số ULB trong {BEST_PARAMS_ULB_FILE}")
    return params


def load_encoded_data(data_dir: Optional[Path] = None, dataset: str = "sparkov"):
    """
    Sparkov: train_encoded.parquet / test_encoded.parquet (encoding fit một lần).
    ULB: ulb_train.parquet / ulb_test.parquet (đã là số, chia theo thời gian), chỉ lấy ULB_FEATURE_COLS (bỏ Time).
    Trả (X_train, y_train, X_test, y_test, feature_names, onehot_groups).
    """
    import pandas as pd

    from src.config import PROCESSED_DATA_DIR, TARGET_COL, ULB_FEATURE_COLS, ULB_TARGET_COL
    from src.imbalance.resamplers import onehot_groups_from_columns

    if dataset not in DATASETS:
        raise ValueError(f"dataset phải là một trong {DATASETS}")
    data_dir = Path(data_dir) if data_dir is not None else PROCESSED_DATA_DIR
    if dataset == "ulb":
        train_df = pd.read_parquet(data_dir / "ulb_train.parquet")
        test_df = pd.read_parquet(data_dir / "ulb_test.parquet")
        feature_names, target = list(ULB_FEATURE_COLS), ULB_TARGET_COL
        onehot_groups: List[List[int]] = []
    else:
        train_df = pd.read_parquet(data_dir / "train_encoded.parquet")
        test_df = pd.read_parquet(data_dir / "test_encoded.parquet")
        target = TARGET_COL
        feature_names = [c for c in train_df.columns if c != target]
        if [c for c in test_df.columns if c != target] != feature_names:
            raise ValueError("Cột train_encoded và test_encoded khác nhau (tên hoặc thứ tự)")
        onehot_groups = onehot_groups_from_columns(feature_names)
    return (
        train_df[feature_names].to_numpy(dtype=np.float64),
        train_df[target].to_numpy(),
        test_df[feature_names].to_numpy(dtype=np.float64),
        test_df[target].to_numpy(),
        feature_names,
        onehot_groups,
    )


def cache_job(data_dir, technique: str, seed: int, cache_root, dataset: str = "sparkov") -> Dict[str, Any]:
    """Việc 'cache': đọc dữ liệu rồi build_resample_cache. Top-level — gọi qua run_isolated."""
    X_train, y_train, _, _, feature_names, onehot_groups = load_encoded_data(data_dir, dataset)
    return build_resample_cache(technique, X_train, y_train, seed, feature_names, onehot_groups, cache_root)


def combo_job(data_dir, model_name: str, technique: str, seeds, dataset: str = "sparkov", **kwargs) -> Dict[str, Any]:
    """Việc 'train': đọc dữ liệu rồi run_rerun_combo. Top-level — gọi qua run_isolated."""
    X_train, y_train, X_test, y_test, feature_names, onehot_groups = load_encoded_data(data_dir, dataset)
    kwargs.setdefault("params", dataset_params(dataset, model_name))
    result = run_rerun_combo(
        model_name, technique, X_train, y_train, X_test, y_test, feature_names, onehot_groups, seeds, **kwargs,
    )
    result["dataset"] = dataset
    result["config"]["dataset"] = dataset
    if kwargs.get("params"):
        result["config"]["params_source"] = f"results/best_params_{dataset}.json"
    return result


def pilot_job(data_dir, model_name: str, dataset: str = "sparkov", **kwargs) -> Dict[str, Any]:
    """Việc 'pilot': đọc dữ liệu rồi pilot_timing. Top-level — gọi qua run_isolated."""
    X_train, y_train, X_test, y_test, _, _ = load_encoded_data(data_dir, dataset)
    kwargs.setdefault("params", dataset_params(dataset, model_name))
    return pilot_timing(model_name, X_train, y_train, X_test, y_test, **kwargs)
