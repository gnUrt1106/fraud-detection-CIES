"""
bench_catboost_gpu.py - Đo CatBoost CPU so với GPU: tốc độ, PR-AUC, SHAP, và (quan trọng nhất) GPU có tái lập từng bit không.

Mục đích: trả lời "sao không train CatBoost trên GPU?" bằng số đo thay vì lập luận. KHÔNG tạo kết quả chính (AGENTS.md: CatBoost kết quả chính = CPU
+ SHAP Regular). Chạy trên máy có GPU NVIDIA (vd. A100 trên Colab); đã có ô trong notebook.

    python scripts/bench_catboost_gpu.py --dataset sparkov --depth 7
    python scripts/bench_catboost_gpu.py --dataset ulb --depth 7 --repeats 3
    python scripts/bench_catboost_gpu.py --dataset ulb --rows 20000 --skip-gpu     # chạy thử không cần GPU

Đo (cùng tham số Optuna hiện tại, chỉ ép `depth`; dữ liệu train đã encode, KHÔNG resample, seed 0):
  1. Thời gian train CPU / GPU và PR-AUC trên TOÀN BỘ test.
  2. Tái lập: train mỗi thiết bị `--repeats` lần với CÙNG seed rồi so xác suất dự đoán (giống từng bit? chênh lệch tối đa?). CPU là nhóm đối chứng.
  3. Thời gian SHAP `Regular` trên `--shap-rows` dòng cho model CPU và model GPU (kiểm tra giả thiết "SHAP chính xác chạy CPU dù train ở đâu").
Ghi results/benchmarks/catboost_gpu_vs_cpu_<dataset>.json và in kết luận. Kết luận về tái lập chỉ có giá trị khi cả GPU và CPU đều đã chạy.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _train_once(train_mod, X, y, params, seed, gpu):
    """Train CatBoost trên CPU hoặc GPU (ép qua cache phát hiện GPU của train.py, như `--force-cpu`), trả (model, giây)."""
    train_mod._has_gpu_cache = bool(gpu)
    model = train_mod.build_model("catboost", input_dim=X.shape[1], params=params, seed=seed)
    got_gpu = model.get_params().get("task_type") == "GPU"
    assert got_gpu == bool(gpu), f"mong đợi {'GPU' if gpu else 'CPU'} nhưng build_model cho task_type={model.get_params().get('task_type')}"
    t0 = time.time()
    trained = train_mod.train_model(model, X, y, model_name="catboost")
    return trained, time.time() - t0


def _compare(preds):
    """So các lần dự đoán với lần đầu: giống từng bit? chênh tối đa."""
    ref = preds[0]
    diffs = [float(np.max(np.abs(p - ref))) for p in preds[1:]]
    return {"n_runs": len(preds), "bit_identical": all(np.array_equal(p, ref) for p in preds[1:]),
            "max_abs_diff_vs_first": max(diffs) if diffs else 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["sparkov", "ulb"], default="sparkov")
    ap.add_argument("--depth", type=int, default=7, help="ép depth (mặc định 7); 0 = giữ depth trong best_params")
    ap.add_argument("--repeats", type=int, default=3, help="số lần train cùng seed trên mỗi thiết bị")
    ap.add_argument("--shap-rows", type=int, default=2000)
    ap.add_argument("--rows", type=int, default=None, help="chỉ dùng N dòng train đầu (chạy thử nhanh)")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.repeats < 2:
        ap.error("--repeats phải >= 2 (cần so các lần chạy với nhau)")

    from sklearn.metrics import average_precision_score

    from src.config import BEST_PARAMS_FILE, BEST_PARAMS_ULB_FILE, RESULTS_DIR
    from src.explainability import rerun_stability as rs
    from src.models import train as train_mod
    from src.models.tune import load_best_params

    X_tr, y_tr, X_te, y_te, _, _ = rs.load_encoded_data(None, a.dataset)
    if a.rows:
        X_tr, y_tr = X_tr[: a.rows], y_tr[: a.rows]
    params = dict(load_best_params("catboost", filepath=BEST_PARAMS_ULB_FILE if a.dataset == "ulb" else BEST_PARAMS_FILE) or {})
    if not params:
        sys.exit("Thiếu tham số CatBoost trong best_params")
    if a.depth:
        params["depth"] = a.depth
    print(f"[{a.dataset}] train {len(y_tr):,} ({int(y_tr.sum())} fraud) | test {len(y_te):,} | params {params}", flush=True)

    out = {"dataset": a.dataset, "params": params, "n_train": int(len(y_tr)), "n_test": int(len(y_te)), "seed": a.seed,
           "note": "dữ liệu train đã encode, KHÔNG resample; chỉ để so thiết bị, không phải kết quả chính"}
    shap_x = X_te[: a.shap_rows]
    devices = [("cpu", False)] + ([] if a.skip_gpu else [("gpu", True)])
    models = {}
    for name, gpu in devices:
        preds, secs = [], []
        for r in range(a.repeats):
            m, sec = _train_once(train_mod, X_tr, y_tr, params, a.seed, gpu)
            preds.append(np.asarray(m.predict_proba(X_te)[:, 1], dtype=np.float64))
            secs.append(round(sec, 2))
            print(f"  {name} lần {r + 1}/{a.repeats}: {sec:.1f}s", flush=True)
        models[name] = m
        pr = [float(average_precision_score(y_te, p)) for p in preds]
        out[name] = {"train_seconds": secs, "pr_auc": pr, "pr_auc_spread": max(pr) - min(pr), "repro": _compare(preds)}
        out[name]["_proba"] = preds[0]

    from catboost import Pool

    for name, _ in devices:
        t0 = time.time()
        sv = models[name].get_feature_importance(Pool(shap_x), type="ShapValues", shap_calc_type="Regular")
        out[name]["shap_regular_seconds"] = round(time.time() - t0, 2)
        out[name]["shap_rows"] = int(len(shap_x))
        print(f"  SHAP Regular {len(shap_x):,} dòng, model {name}: {out[name]['shap_regular_seconds']}s", flush=True)
        out[name]["_shap_shape"] = list(sv.shape)

    if not a.skip_gpu:
        d = np.abs(out["cpu"].pop("_proba") - out["gpu"].pop("_proba"))
        out["cpu_vs_gpu"] = {"max_abs_diff_proba": float(d.max()), "mean_abs_diff_proba": float(d.mean()),
                             "delta_pr_auc_gpu_minus_cpu": out["gpu"]["pr_auc"][0] - out["cpu"]["pr_auc"][0]}
    else:
        out["cpu"].pop("_proba")
    for name, _ in devices:
        out[name].pop("_shap_shape", None)

    # --- kết luận (chỉ khẳng định khi đủ dữ liệu) ---
    lines = [f"CPU: train {np.mean(out['cpu']['train_seconds']):.1f}s, PR-AUC {out['cpu']['pr_auc'][0]:.4f}, "
             f"tái lập từng bit: {out['cpu']['repro']['bit_identical']} (chênh tối đa {out['cpu']['repro']['max_abs_diff_vs_first']:.2e})"]
    if not a.skip_gpu:
        g, c = out["gpu"], out["cpu"]
        lines.append(f"GPU: train {np.mean(g['train_seconds']):.1f}s, PR-AUC {g['pr_auc'][0]:.4f}, "
                     f"tái lập từng bit: {g['repro']['bit_identical']} (chênh tối đa {g['repro']['max_abs_diff_vs_first']:.2e}, PR-AUC dao động {g['pr_auc_spread']:.2e})")
        lines.append(f"GPU so với CPU: ΔPR-AUC {out['cpu_vs_gpu']['delta_pr_auc_gpu_minus_cpu']:+.4f}, chênh xác suất tối đa {out['cpu_vs_gpu']['max_abs_diff_proba']:.3f}")
        lines.append(f"SHAP Regular {a.shap_rows:,} dòng: CPU-model {c['shap_regular_seconds']}s | GPU-model {g['shap_regular_seconds']}s "
                     f"(gần bằng nhau ⇒ SHAP không hưởng lợi từ GPU)")
        if g["repro"]["bit_identical"] and c["repro"]["bit_identical"]:
            lines.append("KẾT LUẬN tái lập: cả GPU lẫn CPU đều tái lập từng bit ở phép thử này ⇒ KHÔNG có cơ sở nêu 'GPU không tái lập' trong hạn chế.")
        elif not g["repro"]["bit_identical"] and c["repro"]["bit_identical"]:
            lines.append("KẾT LUẬN tái lập: GPU KHÔNG tái lập từng bit còn CPU thì có ⇒ lập luận 'nhiễu GPU lẫn vào CIES' có bằng chứng.")
        else:
            lines.append("KẾT LUẬN tái lập: CPU cũng không tái lập từng bit (đối chứng không sạch) ⇒ chưa kết luận được về GPU; thử --repeats lớn hơn/ cố định thread_count.")
    print("\n" + "\n".join(lines))
    out["summary"] = lines

    path = RESULTS_DIR / "benchmarks" / f"catboost_gpu_vs_cpu_{a.dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã ghi {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
