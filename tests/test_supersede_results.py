"""scripts/supersede_results.py: chuyển tổ hợp cũ ra khỏi file kết quả, có sao lưu, mặc định chỉ in."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("supersede_results", Path(__file__).resolve().parents[1] / "scripts" / "supersede_results.py")
sr = importlib.util.module_from_spec(_spec)
sys.modules["supersede_results"] = sr
_spec.loader.exec_module(sr)


def _file(tmp_path):
    p = tmp_path / "colab_full_gpu.json"
    res = [{"combination": f"{m}__{t}", "model_name": m, "complete": True} for m in ("xgboost", "catboost") for t in ("smote", "adasyn")]
    p.write_text(json.dumps({"design": "d", "results": res, "errors": []}), encoding="utf-8")
    return p


def test_dry_run_changes_nothing(tmp_path):
    p = _file(tmp_path)
    before = p.read_text(encoding="utf-8")
    n, kept, names = sr.supersede(p, ["catboost"], tmp_path / "old", apply=False)
    assert (n, kept) == (2, 2) and names == ["catboost__smote", "catboost__adasyn"]
    assert p.read_text(encoding="utf-8") == before and not (tmp_path / "old").exists()


def test_apply_moves_entries_and_keeps_backup(tmp_path):
    p = _file(tmp_path)
    original = json.loads(p.read_text(encoding="utf-8"))
    sr.supersede(p, ["catboost"], tmp_path / "old", apply=True)
    left = json.loads(p.read_text(encoding="utf-8"))
    assert [r["model_name"] for r in left["results"]] == ["xgboost", "xgboost"] and left["design"] == "d"
    moved = json.loads((tmp_path / "old" / "colab_full_gpu_superseded_catboost.json").read_text(encoding="utf-8"))
    assert [r["combination"] for r in moved["results"]] == ["catboost__smote", "catboost__adasyn"]
    assert json.loads((tmp_path / "old" / "colab_full_gpu.json.before_supersede").read_text(encoding="utf-8")) == original


def test_refuses_when_nothing_matches_or_backup_exists(tmp_path):
    p = _file(tmp_path)
    with pytest.raises(ValueError):
        sr.supersede(p, ["ann"], tmp_path / "old", apply=True)
    sr.supersede(p, ["catboost"], tmp_path / "old", apply=True)
    p2 = _file(tmp_path)
    with pytest.raises(FileExistsError):
        sr.supersede(p2, ["catboost"], tmp_path / "old", apply=True)
