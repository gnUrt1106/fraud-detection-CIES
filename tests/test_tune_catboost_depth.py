"""Khoảng depth CatBoost của Optuna: mặc định (4, 12), giới hạn (4, 7) hoặc ép đúng một giá trị (7, 7); depth luôn nằm trong trial.params."""

import pytest

optuna = pytest.importorskip("optuna")

from src.models.tune import sample_hyperparameters  # noqa: E402


def _depths(model_name, **kwargs):
    study = optuna.create_study(direction="maximize")
    out = []
    for _ in range(40):
        trial = study.ask()
        out.append(sample_hyperparameters(trial, model_name, **kwargs).get("depth"))
        study.tell(trial, 0.5)
    return out, study


def test_default_range_is_4_to_12():
    depths, _ = _depths("catboost")
    assert min(depths) >= 4 and max(depths) <= 12 and max(depths) > 7


def test_capped_range_never_exceeds_the_cap():
    depths, _ = _depths("catboost", catboost_depth=(4, 7))
    assert min(depths) >= 4 and max(depths) <= 7


def test_fixed_depth_is_recorded_in_trial_params():
    depths, study = _depths("catboost", catboost_depth=(7, 7))
    assert set(depths) == {7}
    assert all(t.params["depth"] == 7 for t in study.trials)  # để best_params có depth


def test_other_models_ignore_the_catboost_range():
    study = optuna.create_study()
    params = sample_hyperparameters(study.ask(), "xgboost", catboost_depth=(7, 7))
    assert "max_depth" in params and "depth" not in params
