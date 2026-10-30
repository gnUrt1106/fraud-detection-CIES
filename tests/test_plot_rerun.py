"""
test_plot_rerun.py — scripts/plot_rerun.py vẽ được từ bảng tổng hợp của `run_rerun.py merge` (kể cả khi thiếu tổ hợp).
"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import IMBALANCE_TECHNIQUES, MODEL_NAMES  # noqa: E402


def _load_script():
    spec = importlib.util.spec_from_file_location("plot_rerun", ROOT / "scripts" / "plot_rerun.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _summary(path: Path, skip=()):
    results = []
    for i, m in enumerate(MODEL_NAMES):
        for j, t in enumerate(IMBALANCE_TECHNIQUES):
            if (m, t) in skip:
                continue
            results.append({
                "model_name": m, "imbalance_technique": t,
                "cies_metrics": {"cies_score": 0.85 + 0.03 * i + 0.002 * j},
                "pr_auc_summary": {"mean": 0.70 + 0.02 * i + 0.003 * j, "std": 0.002},
            })
    path.write_text(json.dumps({"design": "seed_rerun_full_train", "results": results, "problems": []}), encoding="utf-8")


def test_plot_rerun_writes_both_figures(tmp_path):
    mod = _load_script()
    _summary(tmp_path / "cies_rerun_summary.json")
    _summary(tmp_path / "cies_rerun_summary_ulb.json", skip={("random_forest", "adasyn")})  # ô thiếu vẫn vẽ được
    out = tmp_path / "figs"
    assert mod.main(["--results-dir", str(tmp_path), "--out-dir", str(out)]) == 0
    for name in ("rerun_cies_heatmap.png", "rerun_prauc_vs_cies.png"):
        assert (out / name).stat().st_size > 10_000


def test_plot_rerun_without_summaries_fails_cleanly(tmp_path):
    mod = _load_script()
    assert mod.main(["--results-dir", str(tmp_path), "--out-dir", str(tmp_path / "figs")]) == 1


def test_load_summary_and_number_format(tmp_path):
    mod = _load_script()
    _summary(tmp_path / "s.json")
    d = mod.load_summary(tmp_path / "s.json")
    assert len(d) == 25 and set(d[("ann", "smote")]) == {"cies", "pr_mean", "pr_std"}
    assert mod._fmt(-0.617, 2) == "−0,62" and mod._fmt(0.9695) == "0,970"
