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
| `config.py` | **Trung tâm cấu hình**: đường dẫn, `SEED=42`, `N_RUNS=20`, tên dataset Kaggle, cột target (`is_fraud` / `Class`), `CUSTOMER_IDENTITY_COLS` (không đưa vào model), `ONEHOT_COLS`, `TARGET_ENCODE_COLS`, `ULB_FEATURE_COLS` (bỏ `Time`), `MODEL_NAMES`, `IMBALANCE_TECHNIQUES`, công tắc `SNAP_SYNTHETIC_ONEHOT` |
| `data/download.py` | Tải Sparkov về `data/raw/` qua `kagglehub` (bỏ qua nếu đã có). Chạy: `python -m src.data.download` |
| `data/preprocess.py` | `preprocess_sparkov`: sắp theo thời gian, thêm `hour`/`day_of_week`/`age` (theo năm của từng giao dịch), bỏ ID và `CUSTOMER_IDENTITY_COLS`. `split_ulb_by_time`: ULB sắp theo `Time`, 20% cuối làm test |
| `data/encoding.py` | One-Hot (`gender, category`) và Stratified K-fold Target Encoding out-of-fold (`merchant`). Prior làm mịn chỉ từ fold train; tham số `groups` (dùng `StratifiedGroupKFold`) giữ bản sao của bootstrap cùng fold; `fixed_categories` giữ số cột one-hot ổn định qua các run; `encode_test` dùng thống kê toàn train |
| `imbalance/resamplers.py` | Đúng 5 kỹ thuật: SMOTE, SMOTE-ENN, ADASYN, Borderline-SMOTE (resample) và Class Weighting (trả `class_weights`, không resample). `snap_onehot_groups` / `onehot_groups_from_columns` ép dòng tổng hợp về one-hot hợp lệ khi `SNAP_SYNTHETIC_ONEHOT=True` (mặc định `False`) |
| `models/train.py` | `build_model` / `train_model` / `predict_proba` cho 5 model. CatBoost dùng chung feature đã encode (không `cat_features`). Logistic Regression và ANN được bọc kèm `StandardScaler` (trong dict model). XGBoost/CatBoost/ANN tự dùng GPU nếu có (`_has_gpu` qua `nvidia-smi`, cố ý không import torch). `train_and_evaluate_combo` chạy 1 tổ hợp cho notebook 03. Import thư viện model là lazy để tránh nạp torch cùng xgboost |
| `models/tune.py` | Optuna HPO: TPE + MedianPruner, nhận tập train **chưa encode** (`train_raw`, sắp theo thời gian); validation **theo thời gian** (`time_series_folds`: cửa sổ mở rộng, 3 khối liên tiếp ở 1/3 cuối tập train), mỗi fold encode riêng chỉ bằng dòng trước khối validation (`encode_time_folds`, tính 1 lần, dùng lại mọi trial), tối ưu PR-AUC, tune trên dữ liệu **chưa** xử lý imbalance. `tune_all_models` chạy mỗi model trong subprocess, lưu/merge `results/best_params.json` khi model đủ n_trials; hỗ trợ checkpoint SQLite (`checkpoint_dir`) và ngân sách thời gian mềm (`session_budget`) để chia nhiều phiên Kaggle; `load_best_params` để `build_model` tự nạp |
| `evaluation/metrics.py` | PR-AUC (metric chính), ROC-AUC, F1, F2, Precision@Recall (0.5/0.7/0.8), confusion matrix |
| `explainability/shap_utils.py` | `compute_shap` chọn explainer theo model: `LinearExplainer` (LR), `TreeExplainer` (RF, XGBoost, CatBoost), `DeepExplainer` (ANN, tự lùi về `KernelExplainer` nếu lỗi). Luôn trả `(n_mẫu, n_feature)` cho lớp fraud; LR/ANN giải thích trong không gian đã scale |
| `explainability/cies.py` | Thuật toán CIES: `shap_to_ranks` (hạng trung bình khi hoà), `compute_rank_weighted_distance` (trọng số theo hạng), `compute_stability_metric`, `compute_feature_level_cies`, `run_cies_experiment` (bootstrap có `groups` → encode lại → imbalance → train → SHAP trên eval cố định; loại run có SHAP toàn 0), `run_cies_experiment_isolated`, `compute_condition_agreement_matrix` (đồng thuận thứ hạng giữa các điều kiện, vd. 5 kỹ thuật), `merge_cies_result` (ghi từng tổ hợp, an toàn nhiều tiến trình — notebook 04/05 dùng hàm này), `save_cies_results` (ghi đè cả file, chỉ an toàn với 1 tiến trình) |
| `utils/isolation.py` | `run_isolated`: chạy hàm trong subprocess `spawn` (cách ly torch/xgboost), có timeout với leo thang SIGTERM → SIGKILL; `daemon=False` để Random Forest song song được |
| `utils/jsonio.py` | `update_json`: đọc–sửa–ghi file JSON kết quả dưới khoá `fcntl`, ghi atomic (file tạm + `os.replace`), báo lỗi nếu JSON hỏng thay vì coi là rỗng. Dùng bởi `tune._merge_write` và `cies.merge_cies_result` |
| `visualization/dataset.py` | Biểu đồ insight dataset: mất cân bằng, giờ, category×giờ, số tiền, khoảng cách, nhịp giao dịch, mùa vụ, ULB top-feature |
| `visualization/explain.py` | Biểu đồ SHAP (beeswarm, so sánh model, đồng thuận thứ hạng, dependence) và CIES (heatmap, ổn định thứ hạng, top-k, Spearman giữa run, Sparkov vs ULB, trade-off với PR-AUC) |
| `__init__.py` (các thư mục) | Đánh dấu Python package |

---

## `notebooks/`

| File | Chức năng |
|------|-----------|
| `01_eda.ipynb` | Khám phá dữ liệu Sparkov + data profiling (`reports/profiling/`); xuất biểu đồ ra `reports/figures/` |
| `02_preprocessing.ipynb` | Chia train/test theo thời gian (Sparkov: file gốc; ULB: theo `Time`), bỏ cột định danh khách hàng (`src/data/preprocess.py`), encoding, lưu `data/processed/*.parquet` (giữ thứ tự thời gian) và `encoding_maps.joblib` |
| `kaggle_pipeline.ipynb` | Kaggle: tune (50 trial, `MODELS_SCOPE`) → benchmark → CIES Sparkov nối tiếp; bỏ kết quả clone từ repo 1 lần mỗi phiên, tự resume qua nhiều phiên, ngân sách thời gian chung. Hướng dẫn upload dữ liệu: `KAGGLE_UPLOAD_README.md` |
| `03_train_models.ipynb` | Benchmark 5 model × 5 kỹ thuật → `results/model_benchmark_results.csv` (lưu/resume từng tổ hợp, `timeout=None`) |
| `04_cies_experiment.ipynb` | CIES trên Sparkov (đủ 5 model × 5 kỹ thuật, train subsample phân tầng `SUBSAMPLE_N=100.000` dòng; mục 8 kiểm tra độ nhạy theo cỡ mẫu) → `results/cies_summary_results.json` |
| `05_cies_experiment_ulb.ipynb` | CIES trên ULB (đủ 5×5, toàn bộ train, `feature_level=False`) → `results/cies_summary_results_ulb.json` |
| `06_visualizations.ipynb` | Biểu đồ insight dataset, SHAP và CIES → `reports/figures/` |

---

## `results/`, `reports/`, `tests/`

| File/Thư mục | Chức năng |
|--------------|-----------|
| `results/best_params.json` | Hyperparameter đóng băng từ Optuna, dùng lại ở notebook 03/04/05 |
| `reports/system_architecture.html` | Sơ đồ kiến trúc hệ thống (3 góc nhìn, bảng tra cứu module, giới hạn đã biết) |
| `reports/pipeline_report.md` | Bảng giải thích từng bước pipeline, trạng thái chạy, lệch so với spec, nhật ký sửa lỗi |
| `reports/literature_support.md` | Nguồn cho từng quyết định thiết kế, so sánh CIES repo với bài gốc, danh sách chỗ chưa có nguồn |
| `reports/benchmark_literature.md` | Kết quả PR-AUC/F1 từ các paper dùng Sparkov (2024–2026), cột CIES = N/A minh hoạ khoảng trống nghiên cứu |
| `reports/figures/` | Biểu đồ PNG: 9 ảnh EDA (`01_..09_`) và các ảnh insight/SHAP/CIES từ notebook 06 |
| `reports/profiling/sparkov_profile_report.html` | Báo cáo data profiling tự động |
| `tests/test_pipeline.py` | 30 test (gồm tiền xử lý Sparkov + chia theo thời gian, fold tune theo thời gian và encode riêng từng fold, checkpoint tune, merge kết quả khi nhiều tiến trình ghi song song, file kết quả hỏng không bị ghi đè, độ đồng thuận giữa kỹ thuật): encoding chống rò rỉ (fold prior, index không mặc định, nhóm bootstrap), CatBoost không `cat_features`, 5 kỹ thuật imbalance và one-hot hợp lệ sau resample, công thức CIES (hạng thay vì vị trí cột, hoà hạng), hình dạng SHAP của Tree/ANN, `scale_pos_weight`, ANN (scale, batch 1 mẫu), end-to-end CIES |

---

## `.agents/rules/` — Hướng dẫn cho agent

| File | Chức năng |
|------|-----------|
| `project-overview.md` | Tổng quan, cấu trúc thư mục, tiến độ |
| `coding-standards.md` | Quy chuẩn code Python, data, ML và các bẫy đã gặp |
| `workflow.md` | Quy trình: task mới, sửa bug, thêm feature |
