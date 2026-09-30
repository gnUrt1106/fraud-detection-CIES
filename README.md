# Fraud Detection — CIES

Đồ án nghiên cứu **độ ổn định của giải thích mô hình phát hiện gian lận** dưới các kỹ thuật xử lý mất cân bằng dữ liệu.

Chỉ số đo là **CIES (Credibility Index via Explanation Stability)**: huấn luyện lại mô hình nhiều lần trên các mẫu bootstrap của tập
train, tính SHAP trên một tập đánh giá cố định, rồi đo thứ hạng các feature dao động bao nhiêu giữa các lần chạy. Thứ hạng càng ít đổi thì
giải thích càng đáng tin (CIES = 1: giống hệt nhau ở mọi lần).

- **5 mô hình**: Logistic Regression, Random Forest, XGBoost, CatBoost, ANN (PyTorch).
- **5 kỹ thuật imbalance**: SMOTE, SMOTE-ENN, ADASYN, Borderline-SMOTE, Class Weighting.
- **2 dataset**: Sparkov (chính) và ULB Credit Card Fraud (phụ, để kiểm chứng xu hướng).

## Dữ liệu

| | Sparkov (chính) | ULB (phụ) |
|---|---|---|
| Nguồn | [kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) | [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Chia train/test | **Theo thời gian**, đúng 2 file gốc: `fraudTrain.csv` (01/2019 → 06/2020, 1.296.675 dòng) / `fraudTest.csv` (06/2020 → 12/2020, 555.719 dòng) | **Theo thời gian**: sắp theo `Time`, 20% cuối làm test (227.846 / 56.961 dòng) |
| Tỷ lệ fraud | 0,58% train / 0,39% test | 0,18% train / 0,13% test |
| Feature | 7 feature: `amt`, `category`, `merchant`, `hour`, `day_of_week`, `age`, `gender` → 21 cột sau encoding | V1..V28 (PCA) + `Amount` (bỏ `Time`) |
| Ghi chú | Dữ liệu **mô phỏng** ([Sparkov Data Generation](https://github.com/namebrandon/Sparkov_Data_Generation)) | Dữ liệu thật, ẩn danh |

Sparkov: các cột định danh khách hàng (`lat`, `long`, `city`, `state`, `job`, `city_pop`, `merch_lat`, `merch_long`, `cc_num`, `zip`)
**không** đưa vào model — gộp lại chúng chỉ ra đúng 1 khách hàng và model dùng chúng để học thuộc "khách nào từng bị hack" (xem
`CUSTOMER_IDENTITY_COLS` trong `src/config.py`). Dữ liệu không kèm trong bản nộp — tải theo mục "Chạy thí nghiệm".

## Cấu trúc

```
├── notebooks/
│   ├── 01_eda.ipynb                 # Khám phá dữ liệu Sparkov
│   ├── 02_preprocessing.ipynb       # Chia theo thời gian, bỏ cột định danh, encoding, tải + chia ULB
│   ├── 03_train_models.ipynb        # Benchmark Sparkov: 5 model × 5 kỹ thuật
│   ├── 04_cies_experiment.ipynb     # CIES Sparkov (+ kiểm tra độ nhạy theo cỡ mẫu)
│   ├── 05_cies_experiment_ulb.ipynb # ULB: tune + benchmark + CIES
│   ├── 06_visualizations.ipynb      # Biểu đồ insight, SHAP, CIES, hiệu năng — kèm nhận xét
│   ├── kaggle_pipeline.ipynb        # Chạy trên Kaggle GPU (dùng cho ANN): tune → benchmark → CIES
│   └── KAGGLE_UPLOAD_README.md      # Hướng dẫn chạy notebook Kaggle
├── src/
│   ├── config.py             # Đường dẫn, hằng số, danh sách model/kỹ thuật, cột feature
│   ├── data/                 # download.py, preprocess.py (chia theo thời gian), encoding.py
│   ├── imbalance/            # resamplers.py — 5 kỹ thuật
│   ├── models/               # train.py (5 model), tune.py (Optuna, validation theo thời gian)
│   ├── evaluation/           # metrics.py (PR-AUC là chỉ số chính)
│   ├── explainability/       # shap_utils.py, cies.py (thuật toán CIES)
│   ├── visualization/        # dataset.py, explain.py
│   └── utils/                # isolation.py (mỗi tổ hợp chạy trong tiến trình riêng), jsonio.py
├── tests/test_pipeline.py    # 40 test
├── results/                  # Kết quả thí nghiệm (xem bên dưới)
└── requirements.txt  .env.example
```

## Kết quả (`results/`)

| File | Nội dung |
|---|---|
| `best_params.json`, `best_params_ulb.json` | Tham số tune (Optuna, 50 trial, 3 fold theo thời gian) cho từng model, riêng mỗi dataset |
| `model_benchmark_results.csv`, `model_benchmark_results_ulb.csv` | PR-AUC, ROC-AUC, F1, F2, Precision@Recall, TP/FP/TN/FN trên tập test — 25 tổ hợp mỗi dataset |
| `cies_summary_results.json`, `cies_summary_results_ulb.json` | CIES, Spearman và mean\|SHAP\| của từng lần bootstrap (20 lần) — 25 tổ hợp mỗi dataset |
| `cies_sensitivity_subsample.json` | CIES theo cỡ mẫu bootstrap (kiểm tra độ nhạy) |

**Kết quả chính** (1 seed; chi tiết và biểu đồ trong notebook 06):
- **Borderline-SMOTE cho giải thích kém ổn định nhất** ở 9/10 cặp model × dataset.
- **Mô hình quyết định mức ổn định nhiều hơn kỹ thuật imbalance** (Sparkov: model cây CIES 0,945–0,973, LR 0,906–0,928, ANN
  0,843–0,906).
- **PR-AUC cao không đảm bảo giải thích ổn định** trên Sparkov (Spearman giữa PR-AUC và CIES −0,15, không tính LR).
- Hiệu năng tốt nhất: Sparkov CatBoost × SMOTE PR-AUC **0,887**; ULB Random Forest × SMOTE **0,819**. Trong cùng 1 model, đổi kỹ thuật
  imbalance chỉ làm PR-AUC chênh ≤ 3,4 điểm %.
- CIES tăng theo cỡ mẫu bootstrap nên chỉ so CIES **trong cùng dataset**; giữa 2 dataset chỉ so thứ hạng các kỹ thuật.

## Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
pip install -r requirements.txt
```

Cần Kaggle credentials để tải dữ liệu qua `kagglehub`: đặt `kaggle.json` (tạo tại [Kaggle Settings](https://www.kaggle.com/settings),
mục API) vào `~/.kaggle/`, hoặc `cp .env.example .env` rồi điền `KAGGLE_USERNAME`, `KAGGLE_KEY`.

## Chạy thí nghiệm

Chạy theo thứ tự (notebook mở bằng `jupyter lab` từ thư mục gốc):

1. **Tải Sparkov**: `python -m src.data.download` → `data/raw/`.
2. **`02_preprocessing.ipynb`**: sinh `data/processed/*.parquet` (và tải + chia ULB).
3. **Tune Sparkov**: `src.models.tune.tune_all_models(train_raw)` (local) hoặc `kaggle_pipeline.ipynb` (Kaggle GPU, nên dùng cho ANN) →
   `results/best_params.json`.
4. **`03_train_models.ipynb`**, **`04_cies_experiment.ipynb`**: benchmark và CIES Sparkov.
5. **`05_cies_experiment_ulb.ipynb`**: tune, benchmark và CIES cho ULB (tham số riêng của ULB).
6. **`06_visualizations.ipynb`**: biểu đồ và nhận xét.

Mỗi tổ hợp được lưu ngay khi xong, chạy lại sẽ bỏ qua phần đã có (kết quả đã kèm trong `results/`, nên notebook 03–06 chạy lại chỉ vẽ lại
biểu đồ). Chạy test: `python -m pytest tests -q`.

**Lưu ý kỹ thuật:** torch (ANN) và xgboost không được nạp chung 1 tiến trình (xung đột OpenMP) — mọi model/tổ hợp chạy qua
`src/utils/isolation.py::run_isolated`.

## Tài liệu phương pháp

Lý do cho từng quyết định thiết kế, báo cáo đầy đủ và sơ đồ kiến trúc nằm trong repo GitHub (không kèm trong bản nộp):
[design_decisions.md](https://github.com/gnUrt1106/fraud-detection-CIES/blob/main/reports/design_decisions.md) ·
[pipeline_report.md](https://github.com/gnUrt1106/fraud-detection-CIES/blob/main/reports/pipeline_report.md) ·
[system_architecture.html](https://github.com/gnUrt1106/fraud-detection-CIES/blob/main/reports/system_architecture.html) ·
[literature_support.md](https://github.com/gnUrt1106/fraud-detection-CIES/blob/main/reports/literature_support.md).

CIES trong đồ án là **biến thể** của CIES gốc (Văduva et al., 2026, arXiv:2603.05024): bài gốc nhiễu hoá đầu vào lúc suy luận; đồ án
nhiễu hoá dữ liệu huấn luyện (bootstrap + train lại) và đo trên thứ hạng feature.

## Trích dẫn dữ liệu

```
@misc{sparkov2020,
  author = {Brandon Harris}, title = {Sparkov Data Generation}, year = {2020},
  publisher = {GitHub}, url = {https://github.com/namebrandon/Sparkov_Data_Generation}
}
@misc{kartik2020fraud,
  author = {Kartik Shenoy}, title = {Fraud Detection Dataset}, year = {2020},
  publisher = {Kaggle}, url = {https://www.kaggle.com/datasets/kartik2112/fraud-detection}
}
@misc{ulb2018,
  author = {{Machine Learning Group, ULB}}, title = {Credit Card Fraud Detection}, year = {2018},
  publisher = {Kaggle}, url = {https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud}
}
```
