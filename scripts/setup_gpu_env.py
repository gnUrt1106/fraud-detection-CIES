"""
setup_gpu_env.py — Cài môi trường cho máy có GPU NVIDIA. Chạy bằng python của .venv.

    1. Cài mọi thư viện trong requirements-lock.txt TRỪ torch (đúng phiên bản trong lock).
    2. Cài torch bản CUDA từ download.pytorch.org. Trên Windows `pip install torch` mặc định chỉ ra bản CPU nên
       phải chỉ rõ chỉ mục CUDA; thử lần lượt theo đời driver NVIDIA (nvidia-smi), ưu tiên đúng phiên bản trong lock.
    3. Không được bản CUDA nào → cài torch bản CPU đúng như lock để phần chạy CPU (ANN song song...) vẫn dùng được.

Mã thoát: 0 = xong và torch thấy CUDA | 3 = môi trường dùng được nhưng torch KHÔNG thấy CUDA | 1 = lỗi cài đặt.

    python scripts/setup_gpu_env.py --dry-run    # chỉ in kế hoạch, không cài gì
"""

import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
LOCK_FILE = ROOT / "requirements-lock.txt"
INDEX_URL = "https://download.pytorch.org/whl/{}"
LIBS_CHECK = "import xgboost, catboost, shap, imblearn, pyarrow, sklearn, pandas, optuna"
CUDA_CHECK = ("import sys, torch; "
              "ok = torch.cuda.is_available() and float(torch.ones(4, device='cuda').sum().item()) == 4.0; "
              "print('torch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(0) if ok else 'khong co GPU'); "
              "sys.exit(0 if ok else 1)")


def split_lock(text: str) -> Tuple[List[str], Optional[str]]:
    """(các dòng của lock KHÔNG phải torch, dòng yêu cầu torch hoặc None). torchvision... không bị coi là torch."""
    keep, torch_spec = [], None
    for line in text.splitlines():
        spec = line.split("#")[0].strip()
        if re.match(r"^torch\s*([=<>!~]|$)", spec):
            torch_spec = spec.replace(" ", "")
        else:
            keep.append(line)
    return keep, torch_spec


def cuda_plans(driver_major: Optional[int], torch_spec: Optional[str]) -> List[Tuple[str, str]]:
    """Danh sách (chỉ mục CUDA, yêu cầu torch) sẽ thử theo thứ tự. Rỗng nếu không có driver NVIDIA đủ mới."""
    if driver_major is None or driver_major < 528:        # CUDA 12.x cần driver >= 528 (Windows)
        return []
    pinned = torch_spec or "torch"
    if driver_major >= 580:                                 # CUDA 13.0
        order = [("cu130", pinned), ("cu130", "torch"), ("cu128", pinned), ("cu128", "torch")]
    else:
        order = [("cu128", pinned), ("cu128", "torch"), ("cu126", pinned), ("cu126", "torch")]
    seen, plans = set(), []
    for item in order:
        if item not in seen:
            seen.add(item)
            plans.append(item)
    return plans


def driver_major() -> Optional[int]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60).stdout
        return int(out.strip().splitlines()[0].strip().split(".")[0])
    except Exception:  # noqa: BLE001
        return None


def _py(code: str) -> bool:
    return subprocess.run([sys.executable, "-c", code]).returncode == 0


def _quiet_py(code: str) -> bool:
    return subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


def _pip(*args: str) -> bool:
    return subprocess.run([sys.executable, "-m", "pip", *args]).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lock", type=Path, default=LOCK_FILE)
    ap.add_argument("--dry-run", action="store_true", help="Chỉ in kế hoạch")
    a = ap.parse_args()

    if not a.lock.exists():
        print(f"[LOI] Khong thay {a.lock}")
        return 1
    others, torch_spec = split_lock(a.lock.read_text(encoding="utf-8-sig"))
    drv = driver_major()
    plans = cuda_plans(drv, torch_spec)
    print(f"Python {sys.version.split()[0]} | driver NVIDIA: {drv if drv is not None else 'khong thay nvidia-smi'} | torch trong lock: {torch_spec}")
    if a.dry_run:
        print("Ke hoach torch CUDA:", plans or "khong co (khong du driver)")
        return 0

    # 1) Thư viện (trừ torch)
    if _quiet_py(LIBS_CHECK):
        print("Thu vien trong requirements-lock.txt da co - bo qua buoc cai.")
    else:
        print("Cai thu vien tu requirements-lock.txt (tru torch), mat vai phut...")
        with tempfile.TemporaryDirectory() as td:
            req = Path(td) / "requirements-khong-torch.txt"
            req.write_text("\n".join(others) + "\n", encoding="utf-8")
            if not _pip("install", "-r", str(req)):
                print("[LOI] pip install that bai (kiem tra mang), roi chay lai.")
                return 1

    # 2) torch bản CUDA
    if _py(CUDA_CHECK):
        return 0
    if not plans:
        print("[CANH BAO] Khong thay driver NVIDIA du moi (can driver >= 528). Cap nhat driver tu nvidia.com roi chay lai de dung GPU.")
    for idx, spec in plans:
        print(f"\n--- Cai {spec} ban CUDA tu chi muc {idx} (khoang 2-3 GB, mat vai phut) ---")
        if importlib.util.find_spec("torch") is not None:
            _pip("uninstall", "-y", "torch")
        if _pip("install", "--index-url", INDEX_URL.format(idx), spec) and _py(CUDA_CHECK):
            return 0
        print(f"[!] {spec} tu {idx} khong dung duoc, thu phuong an tiep theo.")

    # 3) Không được CUDA → ít nhất có torch CPU
    if importlib.util.find_spec("torch") is None or not _quiet_py("import torch"):
        print("\nCai torch ban CPU (dung phien ban trong lock)...")
        if not _pip("install", torch_spec or "torch"):
            print("[LOI] Khong cai duoc torch.")
            return 1
    print("[CANH BAO] torch KHONG thay CUDA: SMOTE-ENN tren GPU se bi bo qua, ANN chay CPU.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
