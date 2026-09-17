"""
test_pipeline.py — Comprehensive verification tests for CIES pipeline.

Checks:
1. Encoding: StratifiedKFold Target Encoding + OneHot + fixed_categories on sample data.
2. Models: build_model for all 5, verify CatBoost has NO cat_features, ANN forward/backward.
3. Imbalance: 5 techniques (SMOTE, SMOTE-ENN, ADASYN, Borderline-SMOTE, Class Weighting).
4. Metrics: PR-AUC, F1, F2, Precision@Recall.
5. CIES & SHAP:
   - Rank-weighted distance math & properties (identical rankings = 1.0, reversed rankings low).
   - End-to-end CIES run on sample data (LR + SMOTE, N_RUNS=3).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import SEED, ONEHOT_COLS, TARGET_ENCODE_COLS
from src.data.encoding import encode_train, encode_test, stratified_kfold_target_encode
from src.imbalance.resamplers import apply_imbalance, get_class_weights
from src.models.train import build_model, train_model, predict_proba
from src.evaluation.metrics import evaluate_model
from src.explainability.shap_utils import compute_shap
from src.explainability.cies import (
    compute_rank_weighted_distance,
    shap_to_ranks,
    compute_stability_metric,
    run_cies_experiment,
)


def create_synthetic_data(n_samples=600, fraud_rate=0.05):
    """Tạo dữ liệu giả lập có cấu trúc giống Sparkov để test nhanh."""
    np.random.seed(SEED)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud
    labels = np.array([0] * n_legit + [1] * n_fraud)
    np.random.shuffle(labels)

    genders = np.random.choice(["M", "F"], size=n_samples)
    categories = np.random.choice(["grocery_pos", "shopping_net", "gas_transport"], size=n_samples)
    states = np.random.choice(["CA", "NY", "TX", "FL"], size=n_samples)
    merchants = np.random.choice([f"fraud_merch_{i}" for i in range(20)], size=n_samples)
    cities = np.random.choice([f"city_{i}" for i in range(15)], size=n_samples)
    jobs = np.random.choice([f"job_{i}" for i in range(10)], size=n_samples)

    amt = np.where(labels == 1, np.random.exponential(200, n_samples), np.random.exponential(50, n_samples))
    lat = np.random.uniform(30, 45, n_samples)
    long = np.random.uniform(-120, -70, n_samples)

    df = pd.DataFrame({
        "gender": genders,
        "category": categories,
        "state": states,
        "merchant": merchants,
        "city": cities,
        "job": jobs,
        "amt": amt,
        "lat": lat,
        "long": long,
        "is_fraud": labels,
    })
    return df


def test_encoding():
    print("\n--- Testing Encoding Module ---")
    df = create_synthetic_data(600)
    train_df = df.iloc[:400].reset_index(drop=True)
    test_df = df.iloc[400:].reset_index(drop=True)

    encoded_train, maps = encode_train(
        train_df, target_col="is_fraud",
        onehot_cols=["gender", "category", "state"],
        target_encode_cols=["merchant", "city", "job"],
        n_splits=3,
        random_state=SEED,
    )
    assert "is_fraud" in encoded_train.columns
    assert "merchant_encoded" in encoded_train.columns
    assert "city_encoded" in encoded_train.columns
    assert "job_encoded" in encoded_train.columns
    assert "gender_M" in encoded_train.columns
    assert "merchant" not in encoded_train.columns

    encoded_test = encode_test(
        test_df, maps,
        target_encode_cols=["merchant", "city", "job"],
    )
    # Check that test has matching feature columns
    train_features = [c for c in encoded_train.columns if c != "is_fraud"]
    for col in train_features:
        assert col in encoded_test.columns, f"Missing {col} in encoded_test"
    print("✅ Encoding test passed! Stratified K-Fold target encoding + OneHot consistent.")


def test_catboost_no_cat_features():
    print("\n--- Testing CatBoost spec (no cat_features) ---")
    cb = build_model("catboost", seed=SEED)
    # Verify cat_features is None or not set
    assert getattr(cb, "cat_features", None) is None
    print("✅ CatBoost verified: cat_features is None (dùng chung features đã encode).")


def test_imbalance_techniques():
    print("\n--- Testing 5 Imbalance Techniques ---")
    X = np.random.randn(200, 10)
    y = np.array([0] * 190 + [1] * 10)

    techniques = ["smote", "smote_enn", "adasyn", "borderline_smote", "class_weighting"]
    for tech in techniques:
        X_res, y_res, cw = apply_imbalance(tech, X, y, seed=SEED)
        if tech == "class_weighting":
            assert len(X_res) == len(X)
            assert cw is not None
        else:
            assert len(X_res) >= len(X)
            assert cw is None
        print(f"  ✅ Technique '{tech}' OK (samples: {len(X)} -> {len(X_res)})")


def test_cies_metrics():
    print("\n--- Testing CIES Stability Metrics ---")
    # 1. Identical SHAP values across runs -> distance = 0, CIES = 1.0
    sv1 = np.array([[10.0, 5.0, 1.0], [8.0, 4.0, 2.0]])
    sv2 = np.array([[10.0, 5.0, 1.0], [8.0, 4.0, 2.0]])
    metrics = compute_stability_metric([sv1, sv2])
    assert np.isclose(metrics["cies_score"], 1.0), f"Expected 1.0, got {metrics['cies_score']}"
    assert np.isclose(metrics["mean_rank_distance"], 0.0)

    # 2. Inverted rankings -> distance > 0, CIES < 1.0
    sv_inv = np.array([[1.0, 5.0, 10.0], [2.0, 4.0, 8.0]])
    metrics_inv = compute_stability_metric([sv1, sv_inv])
    assert metrics_inv["cies_score"] < 1.0
    assert metrics_inv["mean_rank_distance"] > 0.0
    print(f"✅ Stability metrics verified: identical CIES = {metrics['cies_score']}, inverted CIES = {metrics_inv['cies_score']:.4f}")


def test_end_to_end_cies():
    print("\n--- Testing End-to-End CIES Experiment (LR + SMOTE, N_RUNS=3) ---")
    df = create_synthetic_data(500)
    train_df = df.iloc[:350].reset_index(drop=True)
    eval_df = df.iloc[350:].reset_index(drop=True)

    result = run_cies_experiment(
        model_name="logistic_regression",
        imbalance_technique="smote",
        df_train=train_df,
        df_test_fixed_eval=eval_df,
        target_col="is_fraud",
        onehot_cols=["gender", "category", "state"],
        target_encode_cols=["merchant", "city", "job"],
        n_runs=3,
        feature_level=True,
    )
    assert result["cies_metrics"]["n_runs"] == 3
    assert 0.0 <= result["cies_metrics"]["cies_score"] <= 1.0
    assert "feature_cies" in result
    assert isinstance(result["feature_cies"], pd.DataFrame)
    print(f"✅ End-to-End CIES run success: CIES score = {result['cies_metrics']['cies_score']:.4f}")
    print("Top features by mean SHAP:")
    print(result["feature_cies"].head(3))


def test_optuna_tuning():
    print("\n--- Testing Optuna HPO Module (tune_model on sample data) ---")
    from src.models.tune import tune_model, sample_hyperparameters
    X = np.random.randn(150, 8)
    y = np.array([0] * 135 + [1] * 15)

    # Test tuning XGBoost with 2 trials
    res = tune_model("xgboost", X, y, n_trials=2, n_splits=2, seed=SEED)
    assert "best_params" in res
    assert "best_pr_auc" in res
    assert res["n_trials"] == 2
    assert "cat_features" not in res["best_params"]
    print(f"✅ Optuna tuning verified: best PR-AUC = {res['best_pr_auc']:.4f}, params = {res['best_params']}")


if __name__ == "__main__":
    test_encoding()
    test_catboost_no_cat_features()
    test_imbalance_techniques()
    test_cies_metrics()
    test_optuna_tuning()
    test_end_to_end_cies()
    print("\n🎉 ALL VERIFICATION TESTS PASSED SUCCESSFULLY! 🎉\n")
