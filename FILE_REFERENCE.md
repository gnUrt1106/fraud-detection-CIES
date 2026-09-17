# 📋 File Reference — Chức năng từng file trong project

## Root Files

| File | Chức năng |
|------|-----------|
| `README.md` | Tài liệu dự án: mô tả, hướng dẫn cài đặt, cách chạy, tiến độ |
| `AGENTS.md` | Entry point cho AI agent — trỏ đến `.agents/rules/` |
| `requirements.txt` | Danh sách thư viện Python (pandas, numpy, matplotlib, seaborn, kagglehub, ydata-profiling, ...) |
| `.env.example` | Mẫu biến môi trường — chứa `KAGGLE_USERNAME` và `KAGGLE_KEY` |
| `.gitignore` | Quy tắc git ignore: data, .venv, __pycache__, .ipynb_checkpoints, ... |

---

## `src/` — Source Code

| File | Chức năng |
|------|-----------|
| `src/__init__.py` | Đánh dấu `src` là Python package |
| `src/config.py` | **Trung tâm cấu hình** — khai báo tất cả đường dẫn (`PROJECT_ROOT`, `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `FIGURES_DIR`, `MODELS_DIR`, `RESULTS_DIR`), tên dataset Kaggle (Sparkov & ULB), hằng số tái lập (`SEED=42`, `N_RUNS=20`), target column (`is_fraud`), danh sách cột one-hot (`ONEHOT_COLS`) và target encode (`TARGET_ENCODE_COLS`), danh sách models & imbalance techniques. |
| `src/data/__init__.py` | Đánh dấu `src/data` là Python subpackage |
| `src/data/download.py` | Script tải dataset Sparkov từ Kaggle về `data/raw/` qua `kagglehub`. Kiểm tra nếu data đã có thì bỏ qua. Chạy bằng: `python -m src.data.download` |
| `src/data/encoding.py` | **Encoding pipeline**: Stratified K-fold Target Encoding (bắt buộc dùng `StratifiedKFold`) cho `merchant, city, job`; One-hot cho `gender, category, state`. Hỗ trợ `fixed_categories` chống lệch feature khi bootstrap. Fit chỉ trên train, transform test. |
| `src/imbalance/__init__.py` | Đánh dấu `src/imbalance` là Python subpackage |
| `src/imbalance/resamplers.py` | Wrapper cho **chính xác 5 kỹ thuật imbalance**: SMOTE, SMOTE-ENN, ADASYN, Borderline-SMOTE, và Class Weighting (không resample data, truyền class weight vào loss/model). |
| `src/models/__init__.py` | Đánh dấu `src/models` là Python subpackage |
| `src/models/train.py` | Build và huấn luyện **5 models**: Logistic Regression, Random Forest, XGBoost, CatBoost (dùng chung features đã encode, KHÔNG dùng `cat_features`), ANN (PyTorch MLP). Hỗ trợ hàm `predict_proba` thống nhất. |
| `src/evaluation/__init__.py` | Đánh dấu `src/evaluation` là Python subpackage |
| `src/evaluation/metrics.py` | Bộ metrics đánh giá: **PR-AUC (metric chính)**, ROC-AUC (chỉ tham khảo), F1, F2, Precision@Recall cố định (0.5, 0.7, 0.8). |
| `src/explainability/__init__.py` | Đánh dấu `src/explainability` là Python subpackage |
| `src/explainability/shap_utils.py` | Tính SHAP values chuẩn hóa theo loại mô hình: `LinearExplainer` (LR), `TreeExplainer` (RF, XGBoost, CatBoost), `KernelExplainer` (ANN). |
| `src/explainability/cies.py` | **Thuật toán cốt lõi CIES**: `compute_rank_weighted_distance` ($w_r = 1/r$), `compute_stability_metric`, `compute_feature_level_cies`, `run_cies_experiment` (bootstrap resample -> re-encode -> resample -> train -> SHAP trên eval cố định -> N runs -> CIES score), và `save_cies_results`. |

---

## `notebooks/` — Jupyter Notebooks

| File | Chức năng |
|------|-----------|
| `01_eda.ipynb` | **Exploratory Data Analysis** — phân tích khám phá dữ liệu: phân bố class imbalance, phân bố amount, phân tích theo giờ/ngày, fraud rate theo category/gender, correlation heatmap, phân bố tuổi. Xuất biểu đồ ra `reports/figures/`. |
| `02_preprocessing.ipynb` | **Preprocessing & Encoding Pipeline** — load raw data, stratified train/test split 80/20, cleaning/feature engineering, fit Stratified K-fold Target Encoding & One-hot, transform test, lưu dữ liệu vào `data/processed/`, xử lý dataset phụ ULB. |
| `03_train_models.ipynb` | **Model Training & Evaluation Benchmark** — chạy 25 tổ hợp (5 models × 5 kỹ thuật imbalance), đánh giá qua PR-AUC, F1, F2, ROC-AUC, xuất bảng kết quả và biểu đồ so sánh ra `results/`. |
| `04_cies_experiment.ipynb` | **CIES Explanation Stability Experiment** — chạy pipeline CIES với bootstrap resample và eval set cố định, tính CIES score, vẽ CIES Heatmap, biểu đồ Trade-off (PR-AUC vs CIES), phân tích độ ổn định cấp đặc trưng (Feature-level CIES). |

---

## `data/` — Dữ liệu

| Thư mục | Chức năng |
|---------|-----------|
| `data/raw/` | Dữ liệu gốc từ Kaggle: `fraudTrain.csv` (~351MB), `fraudTest.csv` (~150MB) |
| `data/processed/` | Dữ liệu đã tiền xử lý: `train_encoded.parquet`, `test_encoded.parquet`, `encoding_maps.joblib`, `ulb_train.parquet`, `ulb_test.parquet` |

---

## `reports/` — Báo cáo & Biểu đồ

| Thư mục/File | Chức năng |
|--------------|-----------|
| `reports/benchmark_literature.md` | Bảng tổng hợp kết quả PR-AUC/F1 từ các paper nghiên cứu trên Sparkov (2024–2026) với cột CIES = N/A minh họa khoảng trống nghiên cứu |
| `reports/figures/` | Biểu đồ xuất từ EDA notebook (9 biểu đồ PNG) |
| `reports/profiling/` | Báo cáo data profiling tự động (`sparkov_profile_report.html`) |

---

## `tests/` — Kiểm thử tự động

| File | Chức năng |
|------|-----------|
| `tests/test_pipeline.py` | Test suite kiểm tra toàn diện: encoding chống leakage, CatBoost không dùng `cat_features`, 5 kỹ thuật imbalance, tính đúng đắn của công thức CIES, end-to-end CIES run trên sample data. |

---

## `.agents/rules/` — Hướng dẫn cho Agent

| File | Chức năng |
|------|-----------|
| `project-overview.md` | Tổng quan project, cấu trúc thư mục, dataset info, tiến độ |
| `coding-standards.md` | Quy chuẩn code: PEP 8, type hints, pathlib, config pattern, data/ML conventions |
| `workflow.md` | Quy trình làm việc: nhận task → sửa bug → thêm feature |

---

## `.vscode/` — IDE Config

| File | Chức năng |
|------|-----------|
| `settings.json` | Cấu hình VSCode: Python interpreter trỏ về `.venv`, extra paths cho import |
