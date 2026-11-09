"""CIES_FORCE_CPU=1 phải tắt nhánh GPU của build_model dù máy có nvidia-smi (tune chạy ở tiến trình con nên cần biến môi trường)."""

import importlib


def test_cies_force_cpu_env_disables_gpu_detection(monkeypatch):
    import src.models.train as train

    monkeypatch.setenv("CIES_FORCE_CPU", "1")
    monkeypatch.setattr(train, "_has_gpu_cache", None)
    assert train._has_gpu() is False


def test_without_env_detection_is_unchanged(monkeypatch):
    import shutil
    import src.models.train as train

    monkeypatch.delenv("CIES_FORCE_CPU", raising=False)
    monkeypatch.setattr(train, "_has_gpu_cache", None)
    monkeypatch.setattr(shutil, "which", lambda name: None)      # không có nvidia-smi => False như cũ
    assert train._has_gpu() is False
    monkeypatch.setattr(train, "_has_gpu_cache", None)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="GPU 0: NVIDIA A100", stderr=""))
    assert train._has_gpu() is True
