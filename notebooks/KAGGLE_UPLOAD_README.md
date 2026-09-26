# Upload dữ liệu cho `kaggle_pipeline.ipynb`

`kaggle_pipeline.ipynb` cần 4 file đã tiền xử lý trong `data/processed/` (tạo bằng
`notebooks/02_preprocessing.ipynb` ở máy local). Upload đúng 4 file này thành 1 Kaggle Dataset.
Mỗi khi chạy lại `02_preprocessing.ipynb`, upload bản mới (tạo version mới của dataset) để Kaggle
dùng đúng dữ liệu hiện tại.

## File cần upload (từ `data/processed/`)

| File | Kích thước |
|---|---|
| `train_encoded.parquet` | ~52 MB |
| `test_encoded.parquet` | ~13 MB |
| `train_raw.parquet` | ~46 MB |
| `test_raw.parquet` | ~12 MB |

Tổng ~123 MB — trong giới hạn Kaggle Dataset dễ dàng.

**Nên upload thêm (tuỳ chọn):** 4 tập train đã resample trong `data/processed/resampled/`
(`train_encoded_{smote,borderline_smote,adasyn,smote_enn}.parquet`, ~300 MB). Notebook tự dùng lại nếu
dấu vân tay khớp (cùng dữ liệu, seed, version `imbalanced-learn`/`scikit-learn`) — tiết kiệm ~2,5 giờ
SMOTE-ENN trên CPU mỗi lần benchmark. Không khớp thì notebook tự tính lại, kết quả vẫn đúng.

## Các bước

1. Vào [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset** (hoặc **New Version** nếu đã có dataset cũ).
2. Kéo thả (hoặc chọn file) đúng 4 file ở bảng trên từ `data/processed/`.
3. Đặt tên dataset, ví dụ `cies-processed` (đặt **Private** nếu không muốn công khai dữ liệu).
4. Create → đợi upload xong.
5. Mở `notebooks/kaggle_pipeline.ipynb` trên Kaggle (Import Notebook từ GitHub) →
   **Add Input** → chọn dataset vừa tạo.
6. Bật **Internet: On**, khuyến nghị bật **Accelerator: GPU**.
7. Chạy từ đầu — cell "Chuẩn bị dữ liệu" tự tìm và symlink cả 4 file.

## Nếu chạy nhiều phiên (Kaggle giới hạn 9h/phiên)

Sau mỗi phiên: **Save Version** → mở phiên mới → **Add Input** → chọn tab **Your Work** →
Notebook Output của phiên vừa Save (không phải dataset gốc) → chạy lại từ đầu. Cell "Khôi phục"
tự tìm checkpoint Optuna + `best_params.json`/`model_benchmark_results.csv`/
`cies_summary_results.json` dở dang từ phiên trước và tiếp tục đúng chỗ dừng.

## Chạy một phần ở máy local, một phần trên Kaggle

Có thể tune + chạy một số model ở máy local (vd. LR, XGBoost), còn lại trên Kaggle (vd. ANN):

1. Chạy xong các model local, **commit và push** `results/` lên GitHub.
2. Trên Kaggle đặt `MODELS_SCOPE` chỉ gồm model còn lại (vd. `["ann"]`). Notebook giữ nguyên kết quả của
   repo cho các model ngoài `MODELS_SCOPE` và chỉ chạy các model trong đó.
3. Trong lúc Kaggle chạy, không sửa kết quả local của các model ngoài `MODELS_SCOPE` (nếu có, push lại
   trước khi mở phiên Kaggle mới — notebook luôn lấy bản mới nhất trong repo cho các model đó).

## Sau khi xong

Tải `results/best_params.json`, `results/model_benchmark_results.csv`,
`results/cies_summary_results.json` từ tab Output — chúng đã gồm cả kết quả của repo lẫn của Kaggle —
ghi đè vào `results/` ở repo local, rồi:

```bash
python -m pytest tests -q
```

để chắc chắn không có gì vỡ trước khi commit.
