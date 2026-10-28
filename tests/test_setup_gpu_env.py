"""
Test `scripts/setup_gpu_env.py` (không cần GPU, không cần mạng): chọn chỉ mục CUDA, lọc requirements-lock.
Nạp script theo đường dẫn vì scripts/ không phải package.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_split_lock_drops_only_torch_and_keeps_comments():
    m = _load("setup_gpu_env")
    text = "# chú thích tiếng Việt\r\npandas==2.3.3\r\ntorch==2.14.0\r\ntorchvision==0.30.0\r\nxgboost==3.4.1  # ghi chú\r\n"
    keep, spec = m.split_lock(text)
    assert spec == "torch==2.14.0"
    assert "torchvision==0.30.0" in keep and "pandas==2.3.3" in keep and "xgboost==3.4.1  # ghi chú" in keep
    assert not any(l.startswith("torch==") for l in keep)


def test_real_lock_file_has_a_torch_pin():
    m = _load("setup_gpu_env")
    lock = ROOT / "requirements-lock.txt"
    if not lock.exists():
        return
    keep, spec = m.split_lock(lock.read_text(encoding="utf-8-sig"))
    assert spec and spec.startswith("torch==")
    assert len(keep) > 5


def test_cuda_plans_follow_driver_generation_and_prefer_pinned_version():
    m = _load("setup_gpu_env")
    assert m.cuda_plans(None, "torch==2.14.0") == []
    assert m.cuda_plans(500, "torch==2.14.0") == []
    new = m.cuda_plans(581, "torch==2.14.0")
    assert new[0] == ("cu130", "torch==2.14.0") and ("cu128", "torch") in new
    mid = m.cuda_plans(572, "torch==2.14.0")
    assert mid[0] == ("cu128", "torch==2.14.0") and all(i != "cu130" for i, _ in mid) and ("cu126", "torch") in mid
    assert len(set(m.cuda_plans(581, None))) == len(m.cuda_plans(581, None))      # không trùng lặp khi không có pin


def test_dry_run_prints_plan_without_installing():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "setup_gpu_env.py"), "--dry-run"],
                       capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "Ke hoach torch CUDA" in r.stdout
