# Kết quả tham chiếu mẫu SHAP 20% (Sparkov)

Thư mục này lưu trữ bản sao lưu kết quả Sparkov chạy lại 10 seed với mẫu SHAP phân tầng 20% test (111.144 dòng / 429 fraud) đã hoàn thành trước ngày 2026-11-08.

## Danh sách file:
- `sparkov_20pct.json`: Toàn bộ 25 tổ hợp (5 model × 5 kỹ thuật) × 10 seed trên mẫu 20% test (sao lưu từ `results/rerun/main.json`).
- `cies_rerun_summary_20pct.json`: Bảng tổng hợp CIES và PR-AUC của 25 tổ hợp trên mẫu 20%.
- `cies_rerun_summary_20pct.csv`: Bảng CSV tổng hợp tương ứng.

## Cách Rollback / Sử dụng lại mẫu 20%:
1. **Dòng lệnh:** Thêm cờ `--shap-rows 20%` (hoặc `--shap-rows 111144`):
   ```bash
   python scripts/run_rerun.py train --shap-rows 20%
   ```
2. **Cấu hình mặc định:** Trong `src/config.py`, đổi lại:
   ```python
   RERUN_SHAP_TEST_FRACTION = {"sparkov": 0.20, "ulb": None}
   ```
3. **Khôi phục kết quả đã có mà không cần chạy lại:**
   Chỉ cần copy `sparkov_20pct.json` ngược lại vào `results/rerun/main.json`.
