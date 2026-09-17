---
title: "Sơ đồ Pipeline Thực Tế — Fraud Detection CIES"
---

# Sơ đồ Pipeline Thực Tế — Fraud Detection CIES

> Sơ đồ này phản ánh **đúng code hiện tại** trong `notebooks/` và `src/` (sau audit), không phải bản thiết kế lý thuyết trong `AGENT_SPEC.md`. Các điểm mà code hiện tại đang chạy khác/hẹp hơn so với spec gốc được đánh dấu ⚠️ trực tiếp trên sơ đồ.

Sơ đồ trực quan (Excalidraw, có thể chỉnh sửa): [`pipeline_diagram.excalidraw`](pipeline_diagram.excalidraw) — mở tại [excalidraw.com](https://excalidraw.com) (kéo-thả file vào canvas).

---

## Bảng giải thích từng bước

| Bước | Kỹ thuật dùng | File / Function liên quan |
|---|---|---|
| **Dataset chính** | Sparkov (`kartik2112/fraud-detection`), tải qua `kagglehub`, ~1.85M dòng, fraud rate ~0.52% | `src/data/download.py::download_dataset`; load ở `notebooks/02_preprocessing.ipynb` |
| **Dataset phụ** | ULB (`mlg-ulb/creditcardfraud`), ~284K giao dịch, 28 feature PCA sẵn (V1..V28) + Amount | `notebooks/02_preprocessing.ipynb` (§6) — **chưa được notebook 03/04 tiêu thụ**, chỉ mới tải + split + lưu parquet |
| **Feature engineering & cleaning** | Bỏ cột ID/định danh cá nhân (`trans_num`, `cc_num`, `first`, `last`, `street`); derive `hour`, `day_of_week` từ `trans_date_trans_time`; derive `age` từ `dob` | `notebooks/02_preprocessing.ipynb::preprocess_features` |
| **Train/Test Split** | Stratified random split 80/20 theo `is_fraud` (`sklearn.train_test_split`, seed=42) | `notebooks/02_preprocessing.ipynb` (§3) |
| **Encoding — One-hot** | One-hot cho low-cardinality: `gender`, `category`, `state` | `src/data/encoding.py::encode_train / encode_test` |
| **Encoding — Target Encoding** | Stratified K-Fold Target Encoding cho high-cardinality: `merchant`, `city`, `job` (`StratifiedKFold n_splits=5`, smoothing=10) | `src/data/encoding.py::stratified_kfold_target_encode` |
| **Lưu dữ liệu đã encode** | `train_encoded.parquet`, `test_encoded.parquet`, `encoding_maps.joblib` — dùng cho benchmark (notebook 03) | `notebooks/02_preprocessing.ipynb` (§5) |
| **Lưu dữ liệu raw (chưa encode)** | `train_raw.parquet`, `test_raw.parquet` — bắt buộc cho CIES vì mỗi run phải re-encode từ đầu | `notebooks/02_preprocessing.ipynb` (§5) |
| **Xử lý mất cân bằng** | Đủ 5 kỹ thuật: SMOTE, SMOTE-ENN, ADASYN, Borderline-SMOTE (resample), Class Weighting (`compute_class_weight('balanced')`, không resample data) | `src/imbalance/resamplers.py::apply_imbalance / get_class_weights` |
| **Models** | 5 model: Logistic Regression, Random Forest (CPU), XGBoost, CatBoost (KHÔNG dùng `cat_features` — dùng chung feature đã encode), ANN (PyTorch MLP 128→64→32→1) — XGBoost/CatBoost/ANN **tự động dùng GPU nếu có** (phát hiện qua `nvidia-smi`, không import torch để tránh xung đột OpenMP) | `src/models/train.py::build_model / train_model / _train_ann / _has_gpu` |
| **Hyperparameter tuning (tùy chọn)** | Optuna HPO, tham số tốt nhất đóng băng ở `results/best_params.json`, nạp lại qua `build_model` nếu có | `src/models/tune.py::load_best_params` |
| **Cách ly subprocess** | Mỗi tổ hợp (model × technique) chạy trong 1 subprocess `spawn` riêng — tránh torch (ANN) + xgboost cùng process gây segfault/hang do xung đột OpenMP. Timeout có escalation SIGTERM → SIGKILL (trước đó `process.join()` có thể treo vô hạn nếu process con bỏ qua SIGTERM — đã tái hiện trên 1 GPU driver hang thực tế trên Kaggle) | `src/utils/isolation.py::run_isolated / _terminate_hard` |
| **Benchmark 25 tổ hợp** | Vòng lặp 5 model × 5 kỹ thuật, đánh giá PR-AUC (chính), F1, F2, ROC-AUC, Precision@Recall | `notebooks/03_train_models.ipynb`; `src/models/train.py::train_and_evaluate_combo`; `src/evaluation/metrics.py::evaluate_model` |
| **SHAP — Linear** | `LinearExplainer` (exact) cho Logistic Regression | `src/explainability/shap_utils.py::_compute_shap_linear` |
| **SHAP — Tree** | `TreeExplainer` (exact) cho Random Forest, XGBoost, CatBoost | `src/explainability/shap_utils.py::_compute_shap_tree` |
| **SHAP — Kernel** | `KernelExplainer` (xấp xỉ) cho ANN — log cảnh báo riêng, không so sánh thô với TreeSHAP | `src/explainability/shap_utils.py::_compute_shap_kernel` |
| **CIES — ĐÃ IMPLEMENT** | Bootstrap resample train → re-encode từ đầu → xử lý mất cân bằng → train → SHAP trên eval set cố định, lặp N_RUNS lần | `src/explainability/cies.py::run_cies_experiment` |
| **CIES — Stability metric** | Rank-weighted distance (`w_r = 1/r`) + Spearman correlation → `cies_score` | `src/explainability/cies.py::compute_stability_metric / compute_rank_weighted_distance / shap_to_ranks` |
| **CIES — Feature-level** | Stability cấp từng feature (mean\|SHAP\|, hệ số biến thiên, rank_std) — chỉ cho Sparkov | `src/explainability/cies.py::compute_feature_level_cies` |
| **CIES — chạy cách ly** | Wrap `run_cies_experiment` trong subprocess riêng | `src/explainability/cies.py::run_cies_experiment_isolated` |
| **Lưu kết quả CIES** | `results/cies_summary_results.json` | `src/explainability/cies.py::save_cies_results` |

---

## ⚠️ Ghi chú quan trọng — sai lệch giữa code hiện tại và spec đầy đủ

1. **`notebooks/04_cies_experiment.ipynb` (cell 4) hiện chỉ chạy SUBSET**, không phải full 5×5:
   - Model: chỉ `logistic_regression`, `random_forest`, `xgboost` — **CatBoost và ANN chưa nằm trong vòng lặp thực chạy**.
   - Kỹ thuật imbalance: chỉ `class_weighting`, `smote`, `smote_enn` — **ADASYN và Borderline-SMOTE chưa được chạy**.
   - Train set bị subsample xuống 10,000 dòng trước khi bootstrap (`train_raw.sample(n=min(10000, len(train_raw)))`).
   - Bản thân module `src/explainability/cies.py` hỗ trợ đầy đủ 5 model × 5 kỹ thuật — giới hạn này nằm ở notebook (có thể là chủ đích để chạy debug/local nhanh trước khi lên Kaggle/Colab chạy full scale, đúng như comment "hoặc toàn bộ 25 tổ hợp khi chạy trên GPU server/Kaggle").
2. **Dataset phụ ULB** đã được tải, split, lưu (`ulb_train.parquet`/`ulb_test.parquet`) nhưng **chưa có notebook/code nào tiêu thụ nó** ở bước model hay CIES.
