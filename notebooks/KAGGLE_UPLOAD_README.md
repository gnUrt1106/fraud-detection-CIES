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

## Sau khi xong

Tải `results/best_params.json`, `results/model_benchmark_results.csv`,
`results/cies_summary_results.json` từ tab Output, ghi đè vào `results/` ở repo local, rồi:

```bash
python -m pytest tests -q
```

để chắc chắn không có gì vỡ trước khi commit.
