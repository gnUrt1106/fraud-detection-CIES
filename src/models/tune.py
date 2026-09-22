"""
tune.py — Module tối ưu hóa siêu tham số (Hyperparameter Optimization - HPO) với Optuna.

Phương pháp: Decoupled Baseline Pre-tuning (Trường phái Controlled Experiment).
    - Tối ưu hóa siêu tham số trên tập train (Stratified K-Fold CV) với objective là PR-AUC.
    - Tìm ra bộ tham số tốt nhất cho mỗi mô hình và lưu ra `results/best_params.json`.
    - Đóng băng (freeze) bộ tham số này khi so sánh 5 kỹ thuật imbalance và chạy CIES,
      đảm bảo không phát sinh biến gây nhiễu (confounding bias) về độ phức tạp mô hình.

RÀNG BUỘC (spec mục 5.2):
    - CatBoost: KHÔNG dùng cat_features trong search space.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from src.config import SEED, RESULTS_DIR, MODEL_NAMES
from src.models.train import build_model, train_model, predict_proba

logger = logging.getLogger(__name__)

# Tắt log chi tiết của Optuna để tránh tràn màn hình
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ===== Search Space Builders =====

def sample_hyperparameters(trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
    """
    Định nghĩa không gian tìm kiếm siêu tham số (Search Space) cho từng model.
    """
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-4, 1e4, log=True),
            "max_iter": 2000,
            "solver": "lbfgs",
        }

    elif model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }

    elif model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric": "aucpr",
        }

    elif model_name == "catboost":
        # RÀNG BUỘC: KHÔNG dùng cat_features
        return {
            "iterations": trial.suggest_int("iterations", 100, 600, step=50),
            "depth": trial.suggest_int("depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True),
            "eval_metric": "PRAUC",
            "verbose": 0,
        }

    elif model_name == "ann":
        return {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "epochs": trial.suggest_int("epochs", 10, 100, step=10),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        }

    else:
        raise ValueError(f"Không hỗ trợ search space cho model: {model_name}")


# ===== Model Tuning Runner =====

def tune_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 100,
    n_splits: int = 5,
    timeout: Optional[int] = None,
    seed: int = SEED,
    use_pruning: bool = True,
    n_startup_trials: int = 5,
    storage_path: Optional[Path] = None,
    optimize_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Tối ưu hóa siêu tham số cho 1 model sử dụng Optuna với Stratified K-Fold CV.
    Metric tối ưu: PR-AUC (Average Precision).

    Args:
        model_name: Tên model (logistic_regression, random_forest, xgboost, catboost, ann)
        X: Feature matrix
        y: Target array
        n_trials: Số lần thử (trials)
        n_splits: Số fold cross-validation
        timeout: Thời gian tối đa (giây)
        seed: Random seed
        use_pruning: Bật MedianPruner — sau mỗi fold, so PR-AUC trung bình tạm
            thời (intermediate) với median các trial ĐÃ HOÀN THÀNH trước đó;
            nếu kém hơn rõ rệt thì dừng trial ngay (TrialPruned), không chạy
            hết n_splits fold. KHÔNG ảnh hưởng đến cách tính PR-AUC của các
            trial hoàn thành — chỉ cắt bớt trial rõ ràng tệ để tiết kiệm thời
            gian, đúng ngữ nghĩa Optuna pruning chuẩn cho CV-based HPO.
        n_startup_trials: Số trial đầu tiên LUÔN chạy đủ n_splits fold (không
            bị prune) để có đủ dữ liệu tham chiếu median trước khi bắt đầu
            prune các trial sau.
        storage_path: File SQLite lưu study (checkpoint). Nếu đặt, mọi trial được ghi xuống
            đĩa ngay khi xong; chạy lại với cùng file sẽ TIẾP TỤC study cũ và chỉ chạy nốt số
            trial còn thiếu để đủ n_trials (trial đang dở khi phiên bị ngắt bị đánh dấu FAIL và
            chạy lại). Trial COMPLETE và PRUNED đều tính vào n_trials.
        optimize_timeout: Giây tối đa cho lượt chạy này (mềm: dừng SAU khi trial hiện tại xong).
            Dùng với storage_path để chia nhiều phiên Kaggle; không làm đổi ngân sách n_trials.

    Returns:
        Dict chứa best_params, best_score (PR-AUC), số trials hoàn thành và
        số trials bị pruned. Nếu chưa đủ n_trials (hết optimize_timeout) trả thêm
        "complete": False — khi đó chạy lại để tiếp tục.
    """
    logger.info(f"Bắt đầu Optuna HPO cho '{model_name}' ({n_trials} trials, {n_splits} folds)...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial, model_name)
        val_pr_aucs = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Build model với tham số thử nghiệm của trial
            model = build_model(
                model_name=model_name,
                input_dim=X.shape[1],
                params=params,
                seed=seed,
            )

            # Huấn luyện fold
            trained = train_model(model, X_train_fold, y_train_fold, model_name=model_name)

            # Dự đoán xác suất trên validation fold
            y_val_proba = predict_proba(trained, X_val_fold)

            # Tính PR-AUC trên validation
            fold_prauc = average_precision_score(y_val_fold, y_val_proba)
            val_pr_aucs.append(fold_prauc)

            # Pruning: báo cáo PR-AUC trung bình tạm thời sau mỗi fold
            if use_pruning:
                trial.report(float(np.mean(val_pr_aucs)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        # Trung bình PR-AUC qua các fold
        mean_prauc = float(np.mean(val_pr_aucs))
        return mean_prauc

    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=0)
        if use_pruning
        else optuna.pruners.NopPruner()
    )
    FINISHED = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)

    if storage_path is not None:
        storage_path = Path(storage_path)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            study_name=model_name,
            storage=f"sqlite:///{storage_path}",
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=pruner,
        )
        # Trial đang chạy dở lúc phiên bị ngắt: đánh dấu FAIL (không tính vào n_trials).
        for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)
        n_prev = sum(1 for t in study.trials if t.state in FINISHED)
        if n_prev:
            # Đổi seed theo số trial đã có để lượt tiếp theo không lặp lại đúng dãy ngẫu nhiên cũ.
            study.sampler = optuna.samplers.TPESampler(seed=seed + n_prev)
            print(f"  [{model_name}] tiếp tục từ checkpoint: đã có {n_prev}/{n_trials} trial.", flush=True)
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=pruner,
        )

    def _log_trial_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        status = "PRUNED" if trial.state == optuna.trial.TrialState.PRUNED else "done"
        score = f"{trial.value:.4f}" if trial.value is not None else "N/A"
        try:
            best_so_far = f"{study.best_value:.4f}"
        except ValueError:
            best_so_far = "N/A"
        n_finished_now = sum(1 for t in study.trials if t.state in FINISHED)
        print(
            f"  [{model_name}] trial {n_finished_now}/{n_trials} {status} "
            f"(PR-AUC={score}, best={best_so_far})",
            flush=True,
        )

    n_finished = sum(1 for t in study.trials if t.state in FINISHED)
    remaining = n_trials - n_finished
    if remaining > 0:
        study.optimize(
            objective, n_trials=remaining, timeout=optimize_timeout if optimize_timeout is not None else timeout,
            callbacks=[_log_trial_progress],
        )
    n_finished = sum(1 for t in study.trials if t.state in FINISHED)
    is_complete = n_finished >= n_trials

    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        return {"model_name": model_name, "complete": False, "n_finished": n_finished}

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    n_completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)

    # Thêm các fixed config cần thiết cho từng model
    if model_name == "logistic_regression":
        best_params.setdefault("max_iter", 2000)
        best_params.setdefault("solver", "lbfgs")
    elif model_name == "xgboost":
        best_params.setdefault("eval_metric", "aucpr")
    elif model_name == "catboost":
        best_params.setdefault("eval_metric", "PRAUC")
        best_params.setdefault("verbose", 0)
        best_params.pop("cat_features", None)

    logger.info(
        f"Hoàn thành HPO cho '{model_name}': Best PR-AUC = {best_trial.value:.4f} "
        f"({n_completed} completed, {n_pruned} pruned / {n_finished} tổng)"
    )

    return {
        "model_name": model_name,
        "best_pr_auc": float(best_trial.value),
        "best_params": best_params,
        "n_trials": n_finished,
        "n_completed_trials": n_completed,
        "n_pruned_trials": n_pruned,
        "complete": is_complete,
    }


def _merge_write(out_file: Path, model_name: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Đọc LẠI file kết quả ngay trước khi ghi rồi chỉ cập nhật mục của `model_name`.

    Tránh ghi đè kết quả do tiến trình khác thêm vào file trong lúc lượt chạy này đang tune (một
    lượt RF ~8 giờ đã từng ghi đè các model được merge vào file giữa chừng, vì bản đọc ở đầu lượt
    đã cũ). Trả về nội dung file sau khi ghi.
    """
    current: Dict[str, Any] = {}
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                current = json.load(f)
        except Exception:
            current = {}
    current[model_name] = entry
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2, ensure_ascii=False)
    return current


def tune_all_models(
    X: np.ndarray,
    y: np.ndarray,
    models: Optional[list] = None,
    n_trials: int = 100,
    n_splits: int = 5,
    output_dir: Optional[Path] = None,
    filename: str = "best_params.json",
    seed: int = SEED,
    timeout: Optional[float] = None,
    checkpoint_dir: Optional[Path] = None,
    session_budget: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Tối ưu hóa siêu tham số cho toàn bộ danh sách models và lưu kết quả ra file JSON.

    Args:
        X: Feature matrix
        y: Target array
        models: Danh sách model (mặc định lấy MODEL_NAMES từ config)
        n_trials: Số trials cho mỗi model
        n_splits: Số fold cross-validation
        output_dir: Thư mục lưu (mặc định RESULTS_DIR)
        filename: Tên file kết quả
        seed: Random seed
        timeout: Giây tối đa cho MỖI model (qua run_isolated). Mặc định None
            = KHÔNG giới hạn (chấp nhận đánh đổi có chủ đích — random_forest/
            ann trên dataset lớn có thể cần >2 tiếng chỉ để chạy xong
            n_startup_trials=5 trial đầu của Optuna MedianPruner, và người
            dùng còn dư quota Kaggle để chờ). LƯU Ý: None tắt luôn lưới an
            toàn SIGTERM→SIGKILL của run_isolated() cho model đó — nếu
            subprocess treo THẬT SỰ (không phải chậm, vd. GPU driver
            deadlock), nó sẽ chạy tới khi Kaggle tự ngắt session (tối đa 9h
            GPU) thay vì bị phát hiện sớm. Truyền số giây cụ thể để bật lại
            giới hạn.

        checkpoint_dir: Nếu đặt, mỗi model lưu study Optuna vào <checkpoint_dir>/<model>.db
            (SQLite) sau từng trial; chạy lại sẽ tiếp tục thay vì bắt đầu từ đầu. Model chỉ được
            ghi vào file kết quả khi ĐỦ n_trials.
        session_budget: Tổng giây cho lượt chạy này (mềm, dừng sau trial đang chạy). Dùng để chia
            nhiều phiên Kaggle mà không đổi n_trials. Cần checkpoint_dir, nếu không phần đã chạy
            sẽ mất khi hết giờ.

    Returns:
        Dict tổng hợp best_params của tất cả models
    """
    import time

    session_start = time.time()
    if models is None:
        models = MODEL_NAMES
    if output_dir is None:
        output_dir = RESULTS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / filename
    # Đọc/ghi file kết quả qua _merge_write() ở mỗi model — không đọc trước ở đây, để tránh
    # ghi đè kết quả do tiến trình khác thêm vào file trong lúc lượt chạy này đang tune (xem
    # docstring _merge_write).
    all_results: Dict[str, Any] = {}

    # Mỗi model được tune trong 1 subprocess riêng (run_isolated) — vòng lặp
    # này chạy cả 'ann' và 'xgboost' trong cùng 1 process nếu gọi trực tiếp,
    # có thể segfault/treo do xung đột OpenMP runtime (xem src/utils/isolation.py).
    from src.utils.isolation import run_isolated

    for model_name in models:
        print(f"=== [Optuna HPO] Tối ưu hóa siêu tham số cho: {model_name} ===")
        extra = {}
        if checkpoint_dir is not None:
            extra["storage_path"] = Path(checkpoint_dir) / f"{model_name}.db"
        if session_budget is not None:
            left = session_budget - (time.time() - session_start)
            if left <= 0:
                print(f"  ⏹ Hết ngân sách thời gian của phiên — bỏ qua '{model_name}' (chạy lại để tiếp tục).\n")
                continue
            extra["optimize_timeout"] = left
        try:
            res = run_isolated(
                tune_model,
                model_name=model_name,
                X=X,
                y=y,
                n_trials=n_trials,
                n_splits=n_splits,
                seed=seed,
                timeout=timeout,
                **extra,
            )
        except (TimeoutError, RuntimeError) as e:
            # KHÔNG để 1 model bị treo/crash làm mất kết quả của các model đã
            # xong trước đó — ghi nhận lỗi, in cảnh báo, rồi tune tiếp model kế.
            logger.error(f"'{model_name}' thất bại ({type(e).__name__}: {e}) — bỏ qua, tune tiếp model kế.")
            print(f"  ❌ '{model_name}' thất bại: {e}\n")
            all_results = _merge_write(out_file, model_name, {"error": f"{type(e).__name__}: {e}"})
            continue

        if not res.get("complete", True):
            # Chưa đủ n_trials (hết ngân sách thời gian): KHÔNG ghi vào file kết quả để tránh dùng
            # tham số của study dở dang. Checkpoint đã lưu — chạy lại notebook để tiếp tục.
            print(
                f"  ⏸ '{model_name}' mới xong {res.get('n_finished', res.get('n_trials'))}/{n_trials} trial "
                f"(hết ngân sách thời gian). Checkpoint đã lưu — chạy lại để tiếp tục.\n"
            )
            continue

        print(f"  -> Best PR-AUC: {res['best_pr_auc']:.4f}")
        print(f"  -> Best Params: {res['best_params']}\n")

        # Lưu NGAY sau mỗi model — nếu model kế tiếp bị treo/crash, kết quả
        # của các model đã xong không bị mất (trước đây chỉ ghi 1 lần ở cuối
        # vòng lặp, nên 1 model timeout là mất sạch kết quả của toàn bộ lần
        # chạy, kể cả các model đã tune xong).
        all_results = _merge_write(out_file, model_name, res)

    print(f"✅ Đã lưu best hyperparameters ra: {out_file}")
    return all_results


def load_best_params(
    model_name: Optional[str] = None,
    filepath: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Nạp bộ tham số tối ưu từ file JSON đã lưu (nếu có).

    Args:
        model_name: Tên model cần lấy tham số (nếu None, trả về dict của tất cả models)
        filepath: Đường dẫn tới file JSON (mặc định RESULTS_DIR / 'best_params.json')

    Returns:
        Dict tham số tối ưu hoặc None nếu file chưa tồn tại
    """
    if filepath is None:
        filepath = RESULTS_DIR / "best_params.json"

    if not filepath.exists():
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if model_name is not None:
            if model_name in data:
                return data[model_name].get("best_params", None)
            return None
        return data
    except Exception as e:
        logger.warning(f"Không thể đọc best_params từ {filepath}: {e}")
        return None
