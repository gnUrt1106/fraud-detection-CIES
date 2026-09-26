# Project Overview

## Thông tin chung
- **Project**: Đo độ ổn định giải thích (CIES) của mô hình phát hiện gian lận dưới các kỹ thuật xử lý mất cân bằng
- **Dataset chính**: Sparkov — giao dịch thẻ tín dụng mô phỏng ([kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)), ~1.85M dòng, fraud ~0.52%
- **Dataset phụ**: ULB ([mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)), ~284.8K dòng, fraud ~0.17%
- **Model**: Logistic Regression, Random Forest, XGBoost, CatBoost, ANN
- **Kỹ thuật imbalance**: SMOTE, SMOTE-ENN, ADASYN, Borderline-SMOTE, Class Weighting
- **Ngôn ngữ**: Python 3.12, virtual env `.venv`

## Cấu trúc thư mục

```
Fraud Detection - CIES/
├── AGENTS.md  AGENT_SPEC.md  FILE_REFERENCE.md  README.md
├── requirements.txt  .env.example  .gitignore
│
├── src/
│   ├── config.py                 # Đường dẫn, hằng số, danh sách model/kỹ thuật, công tắc
│   ├── data/                     # download.py, preprocess.py (chia theo thời gian), encoding.py
│   ├── imbalance/resamplers.py   # 5 kỹ thuật + ép one-hot hợp lệ
│   ├── models/                   # train.py (5 model), tune.py (Optuna, fold theo thời gian)
│   ├── evaluation/metrics.py     # PR-AUC chính, F1, F2, ROC-AUC, Precision@Recall
│   ├── explainability/           # shap_utils.py, cies.py
│   ├── visualization/            # dataset.py, explain.py
│   └── utils/                    # isolation.py (run_isolated), jsonio.py (ghi file kết quả an toàn)
│
├── notebooks/                    # 01 EDA, 02 tiền xử lý, kaggle_pipeline,
│                                 # 03 benchmark, 04 CIES Sparkov, 05 CIES ULB, 06 trực quan
├── data/                         # raw/, processed/ — KHÔNG nằm trong git (ignore /data/)
├── results/                      # best_params.json và kết quả benchmark/CIES
├── reports/                      # figures/, profiling/, system_architecture.html,
│                                 # pipeline_report.md, benchmark_literature.md
├── tests/test_pipeline.py        # 30 test
└── .agents/rules/               # Hướng dẫn cho agent
```

## Tiến độ hiện tại (2026-09-26)
- [x] Toàn bộ module `src/` và 7 notebook (01–06 + `kaggle_pipeline.ipynb`)
- [x] Test suite (30 test, gồm hồi quy các lỗi đã sửa)
- [x] Thiết kế hiện tại: chia train/test theo thời gian, không dùng cột định danh khách hàng, tune validate theo thời gian (xem `AGENT_SPEC.md` §8, §11)
- [ ] Chạy lại toàn bộ theo thiết kế hiện tại: tune + benchmark + CIES Sparkov (`kaggle_pipeline.ipynb` hoặc local), rồi CIES ULB (05) và trực quan (06) — `results/` đang trống
- [ ] Thí nghiệm đối chứng KernelSHAP (`AGENT_SPEC.md` §6.2)
