# File Reference — Chức năng từng file trong project

Danh mục theo trạng thái code hiện tại (chỉ liệt kê file nằm trong git). Sơ đồ tổng thể: `reports/system_architecture.html`.

## Root

| File | Chức năng |
|------|-----------|
| `README.md` | Mô tả project, cài đặt, thứ tự chạy thí nghiệm, chạy trên Kaggle, trạng thái |
| `AGENT_SPEC.md` | Đặc tả nghiên cứu: quyết định đã chốt, ràng buộc bắt buộc, danh sách cấm; cuối file có mục 11 ghi các chỗ code hiện tại lệch spec |
| `AGENTS.md` | Entry point cho AI agent — trỏ tới `.agents/rules/` và tóm tắt các bẫy đã gặp |
| `FILE_REFERENCE.md` | File này |
| `requirements.txt` | Thư viện Python (pandas, scikit-learn, imbalanced-learn, xgboost, catboost, torch, shap, optuna, pyarrow, kagglehub, ydata-profiling...) |
| `.env.example` | Mẫu biến môi trường `KAGGLE_USERNAME`, `KAGGLE_KEY` |
| `.gitignore` | Ignore `/data/` (chỉ thư mục ở gốc — không phải `src/data/`), `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `*.log`, `scratch/`, `catboost_info/`... |

---

## `src/` — Mã nguồn

| File | Chức năng |
|------|-----------|
| `config.py` | **Trung tâm cấu hình**: đường dẫn, `SEED=42`, `N_RUNS=20`, tên dataset Kaggle, cột target (`is_fraud` / `Class`), `ONEHOT_COLS`, `TARGET_ENCODE_COLS`, `ULB_FEATURE_COLS` (bỏ `Time`), `MODEL_NAMES`, `IMBALANCE_TECHNIQUES`, công tắc `SNAP_SYNTHETIC_ONEHOT` |
| `data/download.py` | Tải Sparkov về `data/raw/` qua `kagglehub` (bỏ qua nếu đã có). Chạy: `python -m src.data.download` |
| `data/encoding.py` | One-Hot (`gender, category, state`) và Stratified K-fold Target Encoding out-of-fold (`merchant, city, job`). Prior làm mịn chỉ từ fold train; tham số `groups` (dùng `StratifiedGroupKFold`) giữ bản sao của bootstrap cùng fold; `fixed_categories` giữ số cột one-hot ổn định qua các run; `encode_test` dùng thống kê toàn train |
| `imbalance/resamplers.py` | Đúng 5 kỹ thuật: SMOTE, SMOTE-ENN, ADASYN, Borderline-SMOTE (resample) và Class Weighting (trả `class_weights`, không resample). `snap_onehot_groups` / `onehot_groups_from_columns` ép dòng tổng hợp về one-hot hợp lệ khi `SNAP_SYNTHETIC_ONEHOT=True` |
| `models/train.py` | `build_model` / `train_model` / `predict_proba` cho 5 model. CatBoost dùng chung feature đã encode (không `cat_features`). Logistic Regression và ANN được bọc kèm `StandardScaler` (trong dict model). XGBoost/CatBoost/ANN tự dùng GPU nếu có (`_has_gpu` qua `nvidia-smi`, cố ý không import torch). `train_and_evaluate_combo` chạy 1 tổ hợp cho notebook 03. Import thư viện model là lazy để tránh nạp torch cùng xgboost |
| `models/tune.py` | Optuna HPO: TPE + MedianPruner, CV 5 fold, tối ưu PR-AUC, tune trên dữ liệu **chưa** xử lý imbalance. `tune_all_models` chạy mỗi model trong subprocess, lưu/merge `results/best_params.json` sau mỗi model; `load_best_params` để `build_model` tự nạp |
| `evaluation/metrics.py` | PR-AUC (metric chính), ROC-AUC, F1, F2, Precision@Recall (0.5/0.7/0.8), confusion matrix |
| `explainability/shap_utils.py` | `compute_shap` chọn explainer theo model: `LinearExplainer` (LR), `TreeExplainer` (RF, XGBoost, CatBoost), `DeepExplainer` (ANN, tự lùi về `KernelExplainer` nếu lỗi). Luôn trả `(n_mẫu, n_feature)` cho lớp fraud; LR/ANN giải thích trong không gian đã scale |
| `explainability/cies.py` | Thuật toán CIES: `shap_to_ranks` (hạng trung bình khi hoà), `compute_rank_weighted_distance` (trọng số theo hạng), `compute_stability_metric`, `compute_feature_level_cies`, `run_cies_experiment` (bootstrap có `groups` → encode lại → imbalance → train → SHAP trên eval cố định; loại run có SHAP toàn 0), `run_cies_experiment_isolated`, `save_cies_results` |
| `utils/isolation.py` | `run_isolated`: chạy hàm trong subprocess `spawn` (cách ly torch/xgboost), có timeout với leo thang SIGTERM → SIGKILL; `daemon=False` để Random Forest song song được |
| `visualization/dataset.py` | Biểu đồ insight dataset: mất cân bằng, giờ, category×giờ, số tiền, khoảng cách, nhịp giao dịch, mùa vụ, ULB top-feature |
| `visualization/explain.py` | Biểu đồ SHAP (beeswarm, so sánh model, đồng thuận thứ hạng, dependence) và CIES (heatmap, ổn định thứ hạng, top-k, Spearman giữa run, Sparkov vs ULB, trade-off với PR-AUC) |
| `__init__.py` (các thư mục) | Đánh dấu Python package |

---

## `notebooks/`

| File | Chức năng |
|------|-----------|
| `01_eda.ipynb` | Khám phá dữ liệu Sparkov + data profiling (`reports/profiling/`); xuất biểu đồ ra `reports/figures/` |
| `02_preprocessing.ipynb` | Làm sạch, chia 80/20 stratified, encoding, lưu `data/processed/*.parquet` và `encoding_maps.joblib`; tải và chia ULB |
| `kaggle_optuna_tuning.ipynb` | Tune trên Kaggle: clone repo, link `train_encoded.parquet` từ dataset input, chạy `tune_all_models`. Chọn model qua `MODELS_TO_TUNE` |
| `03_train_models.ipynb` | Benchmark 5 model × 5 kỹ thuật → `results/model_benchmark_results.csv` (lưu/resume từng tổ hợp, `timeout=None`) |
| `04_cies_experiment.ipynb` | CIES trên Sparkov (tập con 3 model × 3 kỹ thuật, train subsample 10.000 dòng) → `results/cies_summary_results.json` |
| `05_cies_experiment_ulb.ipynb` | CIES trên ULB (cùng tập con, toàn bộ train, `feature_level=False`) → `results/cies_summary_results_ulb.json` |
| `06_visualizations.ipynb` | Biểu đồ insight dataset, SHAP và CIES → `reports/figures/` |

---

## `results/`, `reports/`, `tests/`

| File/Thư mục | Chức năng |
|--------------|-----------|
| `results/best_params.json` | Hyperparameter đóng băng từ Optuna, dùng lại ở notebook 03/04/05 |
| `reports/system_architecture.html` | Sơ đồ kiến trúc hệ thống (3 góc nhìn, bảng tra cứu module, giới hạn đã biết) |
| `reports/pipeline_report.md` | Bảng giải thích từng bước pipeline, trạng thái chạy, lệch so với spec, nhật ký sửa lỗi |
| `reports/benchmark_literature.md` | Kết quả PR-AUC/F1 từ các paper dùng Sparkov (2024–2026), cột CIES = N/A minh hoạ khoảng trống nghiên cứu |
| `reports/figures/` | Biểu đồ PNG: 9 ảnh EDA (`01_..09_`) và các ảnh insight/SHAP/CIES từ notebook 06 |
| `reports/profiling/sparkov_profile_report.html` | Báo cáo data profiling tự động |
| `tests/test_pipeline.py` | 19 test: encoding chống rò rỉ (fold prior, index không mặc định, nhóm bootstrap), CatBoost không `cat_features`, 5 kỹ thuật imbalance và one-hot hợp lệ sau resample, công thức CIES (hạng thay vì vị trí cột, hoà hạng), hình dạng SHAP của Tree/ANN, `scale_pos_weight`, ANN (scale, batch 1 mẫu), end-to-end CIES |

---

## `.agents/rules/` — Hướng dẫn cho agent

| File | Chức năng |
|------|-----------|
| `project-overview.md` | Tổng quan, cấu trúc thư mục, tiến độ |
| `coding-standards.md` | Quy chuẩn code Python, data, ML và các bẫy đã gặp |
| `workflow.md` | Quy trình: task mới, sửa bug, thêm feature |
