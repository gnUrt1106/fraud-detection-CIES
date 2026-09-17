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
- Target column: `is_fraud`
