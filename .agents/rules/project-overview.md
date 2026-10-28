# Project Overview

## Thông tin chung
- **Project**: Đo độ ổn định giải thích (CIES) của mô hình phát hiện gian lận dưới các kỹ thuật xử lý mất cân bằng
- **Dataset 1**: Sparkov — giao dịch thẻ tín dụng mô phỏng ([kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)), ~1.85M dòng, fraud ~0.52%
- **Dataset 2** (đối xử ngang Sparkov): ULB ([mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)), ~284.8K dòng, fraud ~0.17%
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
│   ├── imbalance/                # resamplers.py (5 kỹ thuật + ép one-hot hợp lệ), gpu_enn.py (ENN trên GPU)
│   ├── models/                   # train.py (5 model), tune.py (Optuna, fold theo thời gian)
│   ├── evaluation/metrics.py     # PR-AUC chính, F1, F2, ROC-AUC, Precision@Recall
│   ├── explainability/           # shap_utils.py, cies.py, rerun_stability.py (chạy lại theo seed),
│   │                             # deeplift_native.py, rf_gpu_shap.py (SHAP nhanh, GPU)
│   ├── visualization/            # dataset.py, explain.py
│   └── utils/                    # isolation.py (run_isolated), jsonio.py (ghi file kết quả an toàn)
│
├── notebooks/                    # 01 EDA, 02 tiền xử lý, kaggle_pipeline,
│                                 # 03 benchmark, 04 CIES Sparkov, 05 CIES ULB, 06 trực quan
├── data/                         # raw/, processed/ — KHÔNG nằm trong git (ignore /data/)
├── results/                      # best_params.json và kết quả benchmark/CIES
├── reports/                      # figures/, profiling/, system_architecture.html,
│                                 # pipeline_report.md, benchmark_literature.md
├── scripts/                      # run_rerun.py (CLI chính), gpu_smote_enn.py, setup_gpu_env.py
├── tests/                        # test_pipeline.py; test của phần chạy lại theo seed, GPU và backend native
└── .agents/rules/               # Hướng dẫn cho agent
```

## Tiến độ hiện tại (2026-10-26)
- [x] Toàn bộ module `src/` và 7 notebook (01–06 + `kaggle_pipeline.ipynb`)
- [x] Thiết kế đầu tiên (bootstrap 20 run) chạy xong 5 model × 5 kỹ thuật × 2 dataset — kết quả tham khảo (`results/cies_summary_results*.json`)
- [x] Thiết kế hiện tại: chia train/test theo thời gian, không dùng cột định danh khách hàng, tune validate theo thời gian (xem `AGENT_SPEC.md` §8, §11)
- [x] Chế độ chạy lại theo seed (yêu cầu giảng viên, `AGENT_SPEC.md` §12): ULB 25/25 tổ hợp; Sparkov 22/25 (còn Random Forest × SMOTE-ENN, ADASYN, Borderline-SMOTE)
- [x] Mẫu SHAP Sparkov = 20% tập test cố định, ULB = toàn bộ test
- [ ] Giảng viên xác nhận mẫu SHAP 20% (đã có bằng chứng 15 tổ hợp so với test đầy đủ)
- [ ] Chạy 3 tổ hợp Random Forest còn lại, `merge` Sparkov + ULB, cập nhật bảng kết quả
- [ ] Đo lại độ nhạy theo cỡ mẫu (slide 21) theo thiết kế chạy lại theo seed
- [ ] Thí nghiệm đối chứng KernelSHAP (`AGENT_SPEC.md` §6.2)
