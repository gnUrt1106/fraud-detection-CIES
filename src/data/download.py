"""
download.py — Tải dataset Sparkov (kartik2112/fraud-detection) từ Kaggle
về thư mục data/raw/ sử dụng kagglehub.

Cách chạy:
    python -m src.data.download

Yêu cầu:
    - Đã cài kagglehub (pip install kagglehub)
    - Đã cấu hình Kaggle credentials (xem README.md)
"""

import sys
import shutil
from pathlib import Path

# Thêm project root vào sys.path để import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import RAW_DATA_DIR, KAGGLE_DATASET_SPARKOV, TRAIN_FILE, TEST_FILE


def _check_existing_data() -> bool:
    """Kiểm tra xem dữ liệu đã tồn tại ở data/raw/ chưa."""
    train_path = RAW_DATA_DIR / TRAIN_FILE
    test_path = RAW_DATA_DIR / TEST_FILE
    if train_path.exists() and test_path.exists():
        return True
    return False


def _format_size(size_bytes: int) -> str:
    """Chuyển byte sang đơn vị dễ đọc."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def _print_data_info():
    """In thông tin cơ bản về dữ liệu đã tải."""
    files = list(RAW_DATA_DIR.glob("*"))
    csv_files = [f for f in files if f.suffix == ".csv"]

    print(f"\n{'='*60}")
    print(f"📁 Đường dẫn dữ liệu: {RAW_DATA_DIR}")
    print(f"📄 Tổng số file: {len(files)}")
    print(f"📄 Số file CSV: {len(csv_files)}")
    print(f"{'-'*60}")

    total_size = 0
    for f in sorted(csv_files):
        size = f.stat().st_size
        total_size += size
        print(f"   {f.name:30s} {_format_size(size):>12s}")

    print(f"{'-'*60}")
    print(f"   {'Tổng dung lượng':30s} {_format_size(total_size):>12s}")
    print(f"{'='*60}\n")


def download_dataset():
    """
    Tải dataset Sparkov từ Kaggle về data/raw/.

    Sử dụng kagglehub để tải. Nếu dữ liệu đã tồn tại, bỏ qua.
    """
    # Kiểm tra dữ liệu đã tồn tại chưa
    if _check_existing_data():
        print("✅ Dữ liệu đã tồn tại ở data/raw/, bỏ qua việc tải lại.")
        _print_data_info()
        return

    # Tải dataset bằng kagglehub
    try:
        import kagglehub
    except ImportError:
        print(
            "❌ Chưa cài đặt kagglehub!\n\n"
            "   Chạy lệnh sau để cài đặt:\n"
            "   .venv/bin/pip install kagglehub\n"
        )
        sys.exit(1)

    print(f"⬇️  Đang tải dataset '{KAGGLE_DATASET_SPARKOV}' từ Kaggle...")
    print("   (Lần đầu có thể mất vài phút tùy tốc độ mạng)\n")

    try:
        # kagglehub.dataset_download tải về cache và trả lại đường dẫn
        cached_path = kagglehub.dataset_download(KAGGLE_DATASET_SPARKOV)
        cached_path = Path(cached_path)
        print(f"📦 Dataset đã tải về cache: {cached_path}")

    except Exception as e:
        error_msg = str(e).lower()

        if "401" in error_msg or "unauthorized" in error_msg or "credentials" in error_msg:
            print(
                "❌ Lỗi xác thực Kaggle API!\n\n"
                "   Hãy cấu hình Kaggle credentials theo một trong các cách sau:\n\n"
                "   Cách 1 — File kaggle.json:\n"
                "     1. Truy cập https://www.kaggle.com/settings\n"
                "     2. Mục 'API' → nhấn 'Create New Token'\n"
                "     3. File kaggle.json sẽ tự tải về\n"
                "     4. Di chuyển file vào: ~/.kaggle/kaggle.json\n"
                "     5. Đặt quyền: chmod 600 ~/.kaggle/kaggle.json\n\n"
                "   Cách 2 — Biến môi trường:\n"
                "     1. Copy .env.example thành .env\n"
                "     2. Điền KAGGLE_USERNAME và KAGGLE_KEY\n"
                "     3. Hoặc export trực tiếp:\n"
                "        export KAGGLE_USERNAME=your_username\n"
                "        export KAGGLE_KEY=your_api_key\n"
            )
        else:
            print(f"❌ Lỗi khi tải dataset: {e}")

        sys.exit(1)

    # Copy file CSV từ cache sang data/raw/
    print(f"\n📂 Đang copy dữ liệu sang {RAW_DATA_DIR}...")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    for csv_file in cached_path.rglob("*.csv"):
        dest = RAW_DATA_DIR / csv_file.name
        if not dest.exists():
            shutil.copy2(csv_file, dest)
            print(f"   ✅ {csv_file.name} → {dest}")
            copied_count += 1
        else:
            print(f"   ⏭️  {csv_file.name} đã tồn tại, bỏ qua.")

    if copied_count == 0 and not _check_existing_data():
        print(
            "⚠️  Không tìm thấy file CSV trong cache!\n"
            f"   Kiểm tra thủ công tại: {cached_path}"
        )
        sys.exit(1)

    print("\n🎉 Tải dữ liệu thành công!")
    _print_data_info()


if __name__ == "__main__":
    # Load biến môi trường từ .env (nếu có)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    download_dataset()
