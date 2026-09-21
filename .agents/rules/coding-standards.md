# Coding Standards

## Python General
- Viết code theo PEP 8
- Thêm docstring (triple-quoted) cho mọi function, class, module
- Comment bằng tiếng Việt hoặc tiếng Anh đều được
- Luôn xử lý exception và logging

## Python Style
- Dùng type hints cho function parameters và return values
- Dùng `pathlib.Path` thay vì `os.path`
- Dùng f-string thay vì `.format()` hoặc `%`
- Import thứ tự: stdlib → third-party → local

## Config & Paths
- Mọi đường dẫn, hằng số đều khai báo trong `src/config.py`
- KHÔNG hardcode path trong notebook hay script
- Dùng `PROJECT_ROOT`, `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, ... từ config

## Data Processing
- Dùng pandas cho data manipulation
- Validate data trước khi xử lý
- Log các bước xử lý quan trọng
- Dữ liệu gốc đặt trong `data/raw/`, kết quả xử lý đặt trong `data/processed/`

## Machine Learning
- Chia train/test set đúng cách, tránh data leakage
- Track experiments với metrics rõ ràng
- Lưu model artifacts vào thư mục riêng
- Target column: `is_fraud` (Sparkov), `Class` (ULB)

## Các bẫy đã gặp (đừng lặp lại)
- **Không import torch và xgboost cùng một process** (xung đột OpenMP). Model chạy qua `src/utils/isolation.py::run_isolated`; import thư viện model là lazy trong `build_model`. Test cần torch phải chạy trong subprocess.
- **Scale input** cho model dựa trên gradient/tuyến tính (Logistic Regression, ANN); cây thì không cần. Scaler được lưu cùng model và phải áp lại ở predict và SHAP.
- **Căn hàng theo vị trí, không theo nhãn index** khi gán Series vào DataFrame đã `reset_index` (dùng `.to_numpy()`).
- **Hình dạng đầu ra của thư viện đổi theo version** (vd. `shap.TreeExplainer` trả mảng 3 chiều cho RandomForest ở shap 0.52): luôn chuẩn hoá về `(n_mẫu, n_feature)` và có test.
- **Trọng số lớp**: `scale_pos_weight` của XGBoost = trọng số lớp dương / trọng số lớp âm (đã từng viết ngược).
- **Metric ổn định phải chống suy biến**: model dự đoán hằng số cho SHAP toàn 0 và thứ hạng hoà nhau nên cho CIES giả bằng 1.0; run như vậy bị loại.
- **`.gitignore`**: dùng `/data/`, không dùng `data/` trần (sẽ nuốt `src/data/`).
- **Thư viện lấy mẫu có hoàn lại** (bootstrap) tạo dòng trùng: khi làm out-of-fold phải gom bản sao vào cùng fold (`groups`).
- **Notebook Kaggle**: `!pip install "pkg>=x"` phải có ngoặc kép; luôn có `timeout` hợp lý hoặc chủ động chấp nhận `None`.
