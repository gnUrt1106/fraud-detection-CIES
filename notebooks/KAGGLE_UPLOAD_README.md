# Upload dữ liệu cho `kaggle_retune_benchmark_cies.ipynb`

Sau khi sửa bug `age` (xem commit `a93e1d7`), 4 file dưới đây trong `data/processed/` đã được
tái tạo lại đúng ở máy local. Upload đúng 4 file này thành 1 Kaggle Dataset để notebook Kaggle
dùng — **không dùng lại dataset cũ** (dữ liệu cũ vẫn còn bug `age`).

## File cần upload (từ `data/processed/`)

| File | Kích thước |
|---|---|
| `train_encoded.parquet` | ~52 MB |
| `test_encoded.parquet` | ~13 MB |
| `train_raw.parquet` | ~46 MB |
| `test_raw.parquet` | ~12 MB |

Tổng ~123 MB — trong giới hạn Kaggle Dataset dễ dàng.

## Các bước

1. Vào [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**.
2. Kéo thả (hoặc chọn file) đúng 4 file ở bảng trên từ `data/processed/`.
3. Đặt tên dataset, ví dụ `cies-processed-v2` (đặt **Private** nếu không muốn công khai dữ liệu thô).
4. Create → đợi upload xong.
5. Mở `notebooks/kaggle_retune_benchmark_cies.ipynb` trên Kaggle (Import Notebook từ GitHub) →
   **Add Input** → chọn dataset vừa tạo (`cies-processed-v2`).
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
