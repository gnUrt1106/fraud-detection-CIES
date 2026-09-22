"""
config.py — Đường dẫn, hằng số, và cấu hình dùng chung cho toàn bộ project.

Tập trung mọi config ở 1 nơi — KHÔNG hardcode path/hằng số ở nơi khác.
"""

from pathlib import Path

# ===== Reproducibility =====
SEED = 42
N_RUNS = 20  # Số lần lặp cho thí nghiệm CIES (tối thiểu 20, có thể tăng 30)

# ===== Đường dẫn gốc =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ===== Dữ liệu =====
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ===== Results =====
RESULTS_DIR = PROJECT_ROOT / "results"

# ===== Báo cáo =====
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PROFILING_DIR = REPORTS_DIR / "profiling"

# ===== Kaggle Datasets =====
KAGGLE_DATASET_SPARKOV = "kartik2112/fraud-detection"
KAGGLE_DATASET_ULB = "mlg-ulb/creditcardfraud"

# ===== Tên file dữ liệu Sparkov =====
TRAIN_FILE = "fraudTrain.csv"
TEST_FILE = "fraudTest.csv"

# ===== Tên file dữ liệu ULB =====
ULB_FILE = "creditcard.csv"

# ===== Cột mục tiêu =====
TARGET_COL = "is_fraud"          # Sparkov
ULB_TARGET_COL = "Class"         # ULB

# ===== Các cột numerical chính (Sparkov) =====
NUMERICAL_COLS = [
    "amt", "lat", "long", "merch_lat", "merch_long",
    "city_pop", "unix_time",
]

# ===== Encoding strategy theo cardinality (Sparkov) =====
# Spec mục 3.1 — KHÔNG one-hot cho high-cardinality
ONEHOT_COLS = ["gender", "category", "state"]
TARGET_ENCODE_COLS = ["merchant", "city", "job"]

# Tất cả categorical columns (union)
CATEGORICAL_COLS = ONEHOT_COLS + TARGET_ENCODE_COLS

# ===== ULB Features =====
# ULB đã PCA sẵn → V1..V28 + Amount + Time, không cần encoding
ULB_FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
# Time bỏ ra vì chỉ là thứ tự giao dịch, không phải feature

# ===== Danh sách models =====
MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "catboost",
    "ann",
]

# ===== Xử lý dòng tổng hợp của các kỹ thuật resample trên cột one-hot =====
# SMOTE/ADASYN/Borderline-SMOTE nội suy tuyến tính nên sinh cột one-hot PHÂN SỐ (đo trên
# Sparkov: 90% dòng tổng hợp có ít nhất 1 cột phân số; 81% "bật" >1 category cùng lúc — điều
# không bao giờ xảy ra ở dữ liệu thật). True = ép mỗi nhóm one-hot của dòng tổng hợp về 1
# category hợp lệ (argmax). LƯU Ý đánh đổi đã đo (XGBoost, Sparkov 300k): PR-AUC SMOTE gốc
# 0.887 → 0.766 khi ép hợp lệ (SMOTENC độc lập cho 0.733; lấy mẫu theo trọng số 0.736) —
# giá trị phân số vô tình cô lập dòng tổng hợp khỏi vùng dữ liệu thật. Đặt False để quay về
# SMOTE thuần (one-hot phân số) nếu muốn so sánh với cách làm phổ biến trong literature.
# MẶC ĐỊNH False (SMOTE thuần, như literature/spec): trên xgboost x smote, 100k dòng, CIES gần như
# không đổi giữa hai chế độ (0.957 khi False, 0.959 khi True) nhưng PR-AUC tụt mạnh khi True,
# nên chọn cách so sánh được với các bài dùng SMOTE chuẩn. Đặt True để chạy ablation.
SNAP_SYNTHETIC_ONEHOT = False

# ===== Danh sách imbalance techniques =====
IMBALANCE_TECHNIQUES = [
    "smote",
    "smote_enn",
    "adasyn",
    "borderline_smote",
    "class_weighting",
]

# ===== Đảm bảo các thư mục tồn tại =====
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, PROFILING_DIR, RESULTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
