# Project Overview

## Thông tin chung
- **Project**: Phát hiện gian lận giao dịch tài chính (Fraud Detection) cho CIES
- **Dataset**: Sparkov — mô phỏng giao dịch thẻ tín dụng ([kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection))
- **Quy mô**: ~1.3 triệu giao dịch train + ~550K test
- **Ngôn ngữ**: Python
- **Virtual env**: `.venv`

## Cấu trúc thư mục

```
Fraud Detection - CIES/
├── AGENTS.md                 # Entry point cho agent
├── AGENT_SPEC.md             # Đặc tả chi tiết nghiên cứu luận văn CIES
├── FILE_REFERENCE.md         # Danh mục tra cứu chức năng từng file
├── README.md                 # Tài liệu dự án
├── requirements.txt          # Dependencies (sklearn, imblearn, xgboost, catboost, shap, torch)
├── .env.example              # Mẫu biến môi trường (Kaggle API)
├── .gitignore
│
├── src/                      # Source code chính
│   ├── __init__.py
│   ├── config.py             # Đường dẫn, hằng số, lists model/techniques, encoding cols
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py       # Tải dataset Sparkov qua kagglehub
│   │   └── encoding.py       # Stratified K-fold Target Encoding, One-hot, fit/transform
│   ├── imbalance/
│   │   ├── __init__.py
│   │   └── resamplers.py     # 5 kỹ thuật: SMOTE, SMOTE-ENN, ADASYN, Borderline-SMOTE, Class Weighting
│   ├── models/
│   │   ├── __init__.py
│   │   └── train.py          # 5 models (LR, RF, XGB, CatBoost [không cat_features], ANN), predict_proba
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py        # PR-AUC chính, ROC-AUC tham khảo, F1, F2, Precision@Recall
│   └── explainability/
│       ├── __init__.py
│       ├── shap_utils.py     # SHAP explainers (Linear, Tree, Kernel/Deep)
│       └── cies.py           # Thuật toán CIES, rank-weighted distance, runner
│
├── notebooks/
│   ├── 01_eda.ipynb          # EDA — phân tích khám phá dữ liệu
│   ├── 02_preprocessing.ipynb# Tiền xử lý & Encoding pipeline
│   ├── 03_train_models.ipynb # Huấn luyện 25 tổ hợp & benchmark PR-AUC/F1
│   └── 04_cies_experiment.ipynb # Thí nghiệm CIES đo độ ổn định giải thích
│
├── data/
│   ├── raw/                  # Dữ liệu gốc (fraudTrain.csv, fraudTest.csv)
│   └── processed/            # Dữ liệu đã tiền xử lý
│
├── reports/
│   ├── figures/              # Biểu đồ EDA (9 file PNG)
│   ├── profiling/            # Data profiling HTML report
│   └── benchmark_literature.md # Bảng literature review Sparkov (2024–2026), CIES = N/A
│
├── tests/
│   └── test_pipeline.py      # Bộ test suite kiểm thử tự động toàn diện
│
└── .agents/
    └── rules/                # Hướng dẫn cho agent
        ├── project-overview.md
        ├── coding-standards.md
        └── workflow.md
```

## Tiến độ hiện tại
- [x] Khởi tạo cấu trúc project
- [x] Script tải dữ liệu (`src/data/download.py`)
- [x] EDA — Exploratory Data Analysis (`notebooks/01_eda.ipynb`)
- [x] Data Profiling tự động
- [x] Module Tiền xử lý dữ liệu (`src/data/encoding.py` - Stratified K-Fold Target Encoding)
- [x] Module Xử lý mất cân bằng (`src/imbalance/resamplers.py` - 5 kỹ thuật)
- [x] Module Huấn luyện mô hình (`src/models/train.py` - 5 models, CatBoost dùng chung feature)
- [x] Module Đánh giá (`src/evaluation/metrics.py` - PR-AUC là metric chính)
- [x] Module SHAP & CIES (`src/explainability/` - Rank-weighted distance, CIES score)
- [x] Test suite kiểm thử toàn diện (`tests/test_pipeline.py` - PASSED)
- [x] Notebooks thực nghiệm (`02_preprocessing`, `03_train_models`, `04_cies_experiment`)
- [x] Báo cáo Literature Benchmark (`reports/benchmark_literature.md`)
