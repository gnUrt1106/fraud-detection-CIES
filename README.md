# Fraud Detection — CIES

Đồ án nghiên cứu **độ ổn định của giải thích mô hình phát hiện gian lận** dưới các kỹ thuật xử lý mất cân bằng dữ liệu.

Chỉ số đo là **CIES (Credibility Index via Explanation Stability)**: huấn luyện lại mô hình nhiều lần trên các mẫu bootstrap của tập train, tính SHAP trên một tập eval cố định, rồi đo thứ hạng các feature dao động bao nhiêu giữa các lần chạy. Thứ hạng càng ít đổi thì giải thích càng đáng tin.

- **5 mô hình**: Logistic Regression, Random Forest, XGBoost, CatBoost, ANN (PyTorch).
- **5 kỹ thuật imbalance**: SMOTE, SMOTE-ENN, ADASYN, Borderline-SMOTE, Class Weighting.
- **2 dataset**: Sparkov (chính) và ULB Credit Card Fraud (phụ, để kiểm chứng xu hướng có khái quát hoá không).

Đặc tả nghiên cứu và các ràng buộc thiết kế nằm ở [`AGENT_SPEC.md`](AGENT_SPEC.md). Sơ đồ kiến trúc hệ thống: [`reports/system_architecture.html`](reports/system_architecture.html) (mở bằng trình duyệt).

## Dữ liệu

| | Sparkov (chính) | ULB (phụ) |
|---|---|---|
| Nguồn | [kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) | [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Quy mô | ~1.85M giao dịch (train ~1.3M + test ~555K gốc, sau đó gộp và chia lại 80/20 stratified) | ~284.8K giao dịch |
| Tỷ lệ fraud | ~0.52% | ~0.17% |
| Feature | Cột thô (categorical + số), cần encoding | V1..V28 (PCA) + Amount, đã là số |
| Ghi chú | Dữ liệu **mô phỏng** bởi [Sparkov Data Generation](https://github.com/namebrandon/Sparkov_Data_Generation) | Dữ liệu thật, ẩn danh |

Cột `Time` của ULB bị loại (chỉ là thứ tự giao dịch); xem `ULB_FEATURE_COLS` trong `src/config.py`.

## Cấu trúc project

```
├── data/                     # KHÔNG nằm trong git (chỉ ignore /data/ ở gốc, xem .gitignore)
│   ├── raw/                  # fraudTrain.csv, fraudTest.csv
│   └── processed/            # train_/test_encoded, train_/test_raw, ulb_*, encoding_maps
├── notebooks/
│   ├── 01_eda.ipynb                 # Khám phá dữ liệu + data profiling
│   ├── 02_preprocessing.ipynb       # Làm sạch, chia 80/20, encoding, xử lý ULB
│   ├── kaggle_optuna_tuning.ipynb   # Tune hyperparameter bằng Optuna (chạy trên Kaggle)
│   ├── 03_train_models.ipynb        # Benchmark 5 model x 5 kỹ thuật
│   ├── 04_cies_experiment.ipynb     # CIES trên Sparkov
│   ├── 05_cies_experiment_ulb.ipynb # CIES trên ULB
│   └── 06_visualizations.ipynb      # Biểu đồ insight dataset, SHAP, CIES
├── src/
│   ├── config.py             # Đường dẫn, hằng số, danh sách model/kỹ thuật, công tắc
│   ├── data/                 # download.py, encoding.py
│   ├── imbalance/            # resamplers.py
│   ├── models/               # train.py, tune.py
│   ├── evaluation/           # metrics.py
│   ├── explainability/       # shap_utils.py, cies.py
│   ├── visualization/        # dataset.py, explain.py
│   └── utils/                # isolation.py (chạy mỗi tổ hợp trong subprocess riêng)
├── tests/test_pipeline.py    # 19 test, gồm hồi quy cho các lỗi đã sửa
├── results/                  # best_params.json và kết quả benchmark/CIES
├── reports/                  # figures/, profiling/, tài liệu và sơ đồ kiến trúc
├── AGENT_SPEC.md  AGENTS.md  FILE_REFERENCE.md
└── requirements.txt  .env.example
```

Chi tiết từng file: [`FILE_REFERENCE.md`](FILE_REFERENCE.md).

## Cài đặt

```bash
git clone https://github.com/gnUrt1106/fraud-detection-CIES.git
cd fraud-detection-CIES
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
pip install -r requirements.txt
```

### Kaggle credentials

Cần để tải dữ liệu qua `kagglehub`. Chọn một trong hai:

- **`kaggle.json`**: tạo token tại [Kaggle Settings](https://www.kaggle.com/settings) (mục API), rồi `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`.
- **Biến môi trường**: `cp .env.example .env` rồi điền `KAGGLE_USERNAME`, `KAGGLE_KEY` (hoặc `export` trực tiếp).

## Chạy thí nghiệm

Chạy theo thứ tự. Các notebook mở bằng `jupyter lab`.

1. **Tải dữ liệu Sparkov**: `python -m src.data.download` (bỏ qua nếu `data/raw/` đã có).
2. **`02_preprocessing.ipynb`**: sinh `data/processed/*.parquet` (và tải/chia ULB).
3. **Tune trên Kaggle** (`kaggle_optuna_tuning.ipynb`, bật GPU): ghi `results/best_params.json`. Xem mục [Chạy trên Kaggle](#chạy-trên-kaggle).
4. **`03_train_models.ipynb`**, **`04_cies_experiment.ipynb`**, **`05_cies_experiment_ulb.ipynb`**: benchmark và CIES. Cả ba đều nạp tham số từ `best_params.json`; mỗi tổ hợp được lưu ngay khi xong nên chạy lại sẽ bỏ qua phần đã có.
5. **`06_visualizations.ipynb`**: vẽ biểu đồ vào `reports/figures/` (phần CIES tự bỏ qua nếu chưa có kết quả).

Chạy test: `python -m pytest tests -q`.

### Chạy trên Kaggle

Repo phải ở chế độ Public để Kaggle `git clone` ẩn danh được.

1. Upload `data/processed/train_encoded.parquet` thành một Kaggle Dataset (tên `cies-processed`) và Add Input vào notebook.
2. Bật **Internet: On** và **Accelerator: GPU**.
3. Sửa `MODELS_TO_TUNE` trong cell config (kết quả tự merge vào `results/best_params.json` có sẵn trong repo).
4. **Save Version → Save & Run All (Commit)**: chạy nền trên server Kaggle, tắt máy vẫn được. Tải `best_params.json` từ tab Output rồi đưa vào `results/`.

## Trạng thái (cập nhật 2026-09-21)

- [x] Pipeline đầy đủ: encoding, 5 kỹ thuật imbalance, 5 model, metrics, SHAP, CIES, tune, trực quan hoá.
- [x] Đợt rà soát toàn bộ mã nguồn: đã sửa các lỗi nghiêm trọng (chi tiết ở [`reports/pipeline_report.md`](reports/pipeline_report.md)).
- [x] Tune: Logistic Regression, XGBoost, CatBoost, ANN đã có tham số trong `results/best_params.json`.
- [ ] Tune: **Random Forest** còn thiếu (notebook Kaggle đang đặt sẵn `MODELS_TO_TUNE = ["random_forest"]`).
- [ ] Chạy benchmark (03), CIES Sparkov (04), CIES ULB (05) trên tham số và code mới — **chưa có kết quả nào**.
- [ ] Thí nghiệm đối chứng KernelSHAP (`AGENT_SPEC.md` §6.2).

## Lưu ý quan trọng

- **Không nạp torch và xgboost trong cùng một process** (xung đột OpenMP có thể segfault hoặc treo). Mỗi model/tổ hợp chạy qua `src/utils/isolation.py::run_isolated` (subprocess `spawn`).
- **`SNAP_SYNTHETIC_ONEHOT`** (`src/config.py`): ép dòng tổng hợp của SMOTE-family về one-hot hợp lệ. Có đánh đổi lớn về PR-AUC — đọc `reports/pipeline_report.md` trước khi đổi.
- **CIES trong repo là biến thể của CIES gốc** (Văduva et al., 2026, arXiv:2603.05024): bài gốc nhiễu hoá đầu vào lúc suy luận, repo nhiễu hoá dữ liệu huấn luyện (bootstrap + train lại) và đo trên thứ hạng. Chi tiết và danh mục nguồn: [`reports/literature_support.md`](reports/literature_support.md).

## License

Project phục vụ mục đích nghiên cứu học thuật.

## Trích dẫn

```
@misc{sparkov2020,
  author = {Brandon Harris},
  title = {Sparkov Data Generation},
  year = {2020},
  publisher = {GitHub},
  url = {https://github.com/namebrandon/Sparkov_Data_Generation}
}

@misc{kartik2020fraud,
  author = {Kartik Shenoy},
  title = {Fraud Detection Dataset},
  year = {2020},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/kartik2112/fraud-detection}
}
```
