"""
config.py — Đường dẫn và hằng số dùng chung cho toàn bộ project.
"""

from pathlib import Path

# ===== Đường dẫn gốc =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ===== Dữ liệu =====
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ===== Báo cáo =====
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PROFILING_DIR = REPORTS_DIR / "profiling"

# ===== Kaggle Dataset =====
KAGGLE_DATASET = "kartik2112/fraud-detection"

# ===== Tên file dữ liệu Sparkov =====
TRAIN_FILE = "fraudTrain.csv"
TEST_FILE = "fraudTest.csv"

# ===== Cột mục tiêu =====
TARGET_COL = "is_fraud"

# ===== Các cột numerical chính =====
NUMERICAL_COLS = [
    "amt", "lat", "long", "merch_lat", "merch_long",
    "city_pop", "unix_time",
]

# ===== Các cột categorical chính =====
CATEGORICAL_COLS = [
    "merchant", "category", "gender", "job", "city", "state",
]

# ===== Đảm bảo các thư mục tồn tại =====
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, PROFILING_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
