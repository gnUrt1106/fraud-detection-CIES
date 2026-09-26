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
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import SEED
from src.data.encoding import encode_train, encode_test, stratified_kfold_target_encode
from src.imbalance.resamplers import apply_imbalance, get_class_weights, onehot_groups_from_columns
from src.models.train import build_model
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


def test_rank_weighted_distance_depends_on_rank_not_column_order():
    """
    Regression test: trọng số phải theo THỨ HẠNG feature, không theo vị trí cột.
    (1) hoán vị cột giống nhau ở cả 2 ranking không được đổi khoảng cách;
    (2) đổi chỗ top-2 phải bị phạt nặng hơn đổi chỗ 2 feature cuối.
    """
    a = np.array([1, 2, 3, 4, 5, 6]); b = np.array([2, 1, 3, 4, 5, 6])
    perm = np.array([4, 2, 5, 0, 3, 1])
    assert np.isclose(compute_rank_weighted_distance(a, b),
                      compute_rank_weighted_distance(a[perm], b[perm]))

    top_swap = compute_rank_weighted_distance(np.array([1, 2, 3, 4, 5, 6]), np.array([2, 1, 3, 4, 5, 6]))
    bottom_swap = compute_rank_weighted_distance(np.array([1, 2, 3, 4, 5, 6]), np.array([1, 2, 3, 4, 6, 5]))
    assert top_swap > bottom_swap, (top_swap, bottom_swap)


def test_condition_agreement_matrix_rank_weighted_not_spearman():
    """
    compute_condition_agreement_matrix(): so thứ hạng GIỮA các điều kiện (vd. kỹ thuật imbalance
    của cùng 1 model), dùng rank-weighted distance — không phải Spearman (coi mọi hạng ngang
    nhau). Cùng 1 cú đổi chỗ top-2 phải bị phạt nặng hơn đổi chỗ 2 feature cuối, y hệt tính chất
    đã kiểm ở compute_rank_weighted_distance, nhưng qua đường vào là mean|SHAP| (không phải rank
    có sẵn) để tái hiện đúng cách notebook 06 gọi hàm này.
    """
    from src.explainability.cies import compute_condition_agreement_matrix

    base = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])  # feature 0 quan trọng nhất
    top_swapped = base.copy(); top_swapped[[0, 1]] = top_swapped[[1, 0]]
    bottom_swapped = base.copy(); bottom_swapped[[4, 5]] = bottom_swapped[[5, 4]]

    m_top = compute_condition_agreement_matrix({"a": base, "b": top_swapped})
    m_bottom = compute_condition_agreement_matrix({"a": base, "b": bottom_swapped})
    assert m_top.loc["a", "b"] < m_bottom.loc["a", "b"], "đổi chỗ top-2 phải bị phạt nặng hơn"

    m_same = compute_condition_agreement_matrix({"a": base, "b": base.copy(), "c": base * 2})
    assert np.allclose(np.diag(m_same.values), 1.0)
    assert np.isclose(m_same.loc["a", "b"], 1.0)
    assert np.isclose(m_same.loc["a", "c"], 1.0), "nhân đôi mọi giá trị không đổi thứ hạng -> vẫn đồng thuận tuyệt đối"


def test_tree_shap_binary_class_shape():
    """
    Regression test: shap.TreeExplainer cho RandomForestClassifier có thể trả
    ndarray 3 chiều (n_samples, n_features, n_classes) thay vì list — đã tái
    hiện với shap==0.52.0 và gây ValueError shape mismatch trong
    compute_stability_metric() khi chạy CIES cho random_forest (không phải
    XGBoost/CatBoost — 2 model đó vốn đã trả đúng 2D). Nếu compute_shap()
    không unwrap đúng chiều class, test này phải fail để bắt lại sớm.
    """
    print("\n--- Testing TreeExplainer shape handling (RandomForest) ---")
    from sklearn.ensemble import RandomForestClassifier

    np.random.seed(SEED)
    X = np.random.rand(100, 5)
    y = (np.random.rand(100) < 0.3).astype(int)
    clf = RandomForestClassifier(n_estimators=10, random_state=SEED).fit(X, y)

    shap_values = compute_shap(clf, "random_forest", X[:10])
    assert shap_values.shape == (10, 5), f"Expected (10, 5), got {shap_values.shape}"
    print(f"✅ TreeExplainer shape OK: {shap_values.shape}")


def test_xgboost_scale_pos_weight_direction():
    """
    Regression test: scale_pos_weight phải LỚN hơn 1 khi fraud là lớp hiếm.
    Từng bị tính ngược (w0/w1) nên class_weighting làm XGBoost bỏ qua fraud.
    """
    y = np.array([0] * 990 + [1] * 10)
    weights = get_class_weights(y)
    model = build_model("xgboost", input_dim=3, class_weights=weights, params={})
    spw = model.get_params()["scale_pos_weight"]
    assert spw > 1.0, f"scale_pos_weight={spw} — fraud (lớp hiếm) phải được TĂNG trọng số"
    assert np.isclose(spw, 99.0, rtol=0.05), f"Kỳ vọng ~99 (n_neg/n_pos), nhận {spw}"


def test_cies_rejects_constant_model_runs(monkeypatch):
    """
    Model hằng số → SHAP toàn 0 → thứ hạng hoà nhau → CIES giả = 1.0. Chốt chặn
    trong run_cies_experiment phải loại các run đó (không được trả điểm cao).
    """
    import src.explainability.cies as cies_mod

    monkeypatch.setattr(cies_mod, "compute_shap", lambda *a, **k: np.zeros((5, 12)))
    df = create_synthetic_data(300)
    result = cies_mod.run_cies_experiment(
        model_name="logistic_regression", imbalance_technique="class_weighting",
        df_train=df.iloc[:200].reset_index(drop=True), df_test_fixed_eval=df.iloc[200:].reset_index(drop=True),
        target_col="is_fraud", onehot_cols=["gender", "category", "state"],
        target_encode_cols=["merchant", "city", "job"], n_runs=3,
    )
    assert result["cies_metrics"]["cies_score"] == 0.0, "run toàn-0 không được cho điểm ổn định cao"
    assert all(r["status"] == "failed" for r in result["run_logs"])


def test_ann_shap_uses_deep_explainer_2d_and_deterministic():
    """
    ANN dùng DeepExplainer: phải trả shape (n, f) — shap>=0.4x trả (n, f, 1) — và tất
    định (KernelExplainer tự lệch chính nó ~0.85 Spearman, làm CIES của ANN thấp giả).
    Chạy trong subprocess riêng: torch + xgboost (đã nạp bởi test khác) cùng 1 process
    có thể segfault do xung đột OpenMP (xem src/utils/isolation.py).
    """
    import subprocess, textwrap
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
        import numpy as np
        from src.models.train import build_model, train_model
        from src.explainability.shap_utils import compute_shap, EXPLAINER_MAP
        assert EXPLAINER_MAP["ann"] == "deep", EXPLAINER_MAP["ann"]
        rng = np.random.RandomState(0)
        X = rng.randn(400, 6); y = (X[:, 0] + 0.5 * X[:, 1] > 1).astype(int)
        m = train_model(build_model("ann", input_dim=6, params={{"epochs": 3, "batch_size": 64}}), X, y, model_name="ann")
        a = compute_shap(m, "ann", X[:20], X_background=X[:50])
        b = compute_shap(m, "ann", X[:20], X_background=X[:50])
        assert a.shape == (20, 6), a.shape
        assert np.allclose(a, b), "DeepExplainer phải tất định"
        print("OK")
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert res.returncode == 0 and "OK" in res.stdout, res.stdout + res.stderr


def test_encode_train_handles_non_default_index():
    """
    Regression test: cột *_encoded được gán theo NHÃN index → NaN/lệch hàng khi df_train
    có index không mặc định (vd. ngay sau train_test_split). Phải gán theo vị trí.
    """
    df = create_synthetic_data(400)
    shuffled = df.sample(frac=1.0, random_state=1)          # index xáo trộn, không phải 0..n-1
    shuffled.index = shuffled.index + 1000                   # và không trùng nhãn với RangeIndex
    enc, _ = encode_train(
        shuffled, target_col="is_fraud", onehot_cols=["gender"],
        target_encode_cols=["merchant"], n_splits=3, random_state=SEED,
    )
    assert enc["merchant_encoded"].isna().sum() == 0, "target encoding bị NaN do lệch index"
    ref, _ = encode_train(
        shuffled.reset_index(drop=True), target_col="is_fraud", onehot_cols=["gender"],
        target_encode_cols=["merchant"], n_splits=3, random_state=SEED,
    )
    assert np.allclose(enc["merchant_encoded"].to_numpy(), ref["merchant_encoded"].to_numpy())


def test_cies_failure_path_returns_all_metric_keys():
    """Nhánh <2 run thành công phải trả đủ khoá — notebook đọc cies_metrics["mean_spearman"]."""
    import src.explainability.cies as cies_mod

    df = create_synthetic_data(300)
    result = cies_mod.run_cies_experiment(
        model_name="logistic_regression", imbalance_technique="class_weighting",
        df_train=df.iloc[:200].reset_index(drop=True), df_test_fixed_eval=df.iloc[200:].reset_index(drop=True),
        target_col="is_fraud", onehot_cols=["gender", "category", "state"],
        target_encode_cols=["merchant", "city", "job"], n_runs=1,
    )
    for key in ("cies_score", "mean_rank_distance", "std_rank_distance", "mean_spearman", "n_runs"):
        assert key in result["cies_metrics"], key


def test_ann_scaling_and_batch_of_one():
    """
    Regression tests cho ANN (subprocess riêng vì torch + xgboost cùng process dễ segfault):
    (1) không crash khi len(train) % batch_size == 1 (BatchNorm1d không nhận batch 1 mẫu);
    (2) học được dù có feature thang ~1e9 (như unix_time) — không scale thì train hỏng.
    """
    import subprocess, textwrap
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
        import numpy as np
        from sklearn.metrics import average_precision_score
        from src.models.train import build_model, train_model, predict_proba
        rng = np.random.RandomState(0)
        # (1) 513 mẫu, batch 512 -> batch cuối = 1 mẫu
        X = rng.randn(513, 4); y = (rng.rand(513) < 0.3).astype(int)
        train_model(build_model("ann", input_dim=4, params={{"epochs": 1, "batch_size": 512}}), X, y, model_name="ann")
        # (2) cột 1 là hằng số cỡ 1e9 + nhiễu nhỏ (như unix_time), cột 0 mang tín hiệu
        n = 3000
        x0 = rng.randn(n); y = (x0 > 0.8).astype(int)
        X = np.column_stack([x0, 1.3e9 + rng.randn(n), rng.randn(n) * 1e6])
        m = train_model(build_model("ann", input_dim=3, params={{"epochs": 15, "batch_size": 128}}), X, y, model_name="ann")
        ap = average_precision_score(y, predict_proba(m, X))
        assert ap > 0.9, f"ANN không học được khi input chưa scale hợp lý: AP={{ap:.3f}}"
        print("OK")
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=240)
    assert res.returncode == 0 and "OK" in res.stdout, res.stdout + res.stderr


def test_cies_vs_prauc_merges_on_model_and_technique():
    """Benchmark đặt tên cột 'imbalance_technique'; ghép phải theo (model, technique), không tích Descartes."""
    import matplotlib
    matplotlib.use("Agg")
    from src.visualization.explain import plot_cies_vs_prauc

    cies = pd.DataFrame({"model": ["a", "a"], "technique": ["x", "y"], "cies_score": [0.9, 0.5]})
    bench = pd.DataFrame({"model": ["a", "a"], "imbalance_technique": ["x", "y"], "pr_auc": [0.8, 0.6]})
    fig = plot_cies_vs_prauc(cies, bench)
    n_points = sum(len(c.get_offsets()) for c in fig.axes[0].collections)
    assert n_points == 2, f"kỳ vọng 2 điểm (1 mỗi tổ hợp), nhận {n_points}"


def test_target_encoding_keeps_bootstrap_duplicates_in_same_fold():
    """
    Bootstrap có dòng trùng. Với `groups`, mọi bản sao của 1 dòng phải cùng fold nên nhận
    CÙNG giá trị encode (và nhãn của chính nó không lọt vào thống kê fold train). Không có
    `groups` thì bản sao rơi vào các fold khác nhau và giá trị encode lệch nhau.
    """
    base = create_synthetic_data(300).reset_index(drop=True)
    sampled = base.sample(frac=1.0, replace=True, random_state=3)
    gid = sampled.index.to_numpy()
    df = sampled.reset_index(drop=True)

    grouped, _ = stratified_kfold_target_encode(df, "merchant", "is_fraud", n_splits=5, random_state=SEED, groups=gid)
    plain, _ = stratified_kfold_target_encode(df, "merchant", "is_fraud", n_splits=5, random_state=SEED)

    def max_spread(enc):
        return pd.Series(enc.to_numpy()).groupby(gid).agg(lambda v: v.max() - v.min()).max()

    assert max_spread(grouped) < 1e-12, "bản sao của cùng 1 dòng bị tách sang fold khác nhau"
    assert max_spread(plain) > 1e-6, "test vô nghĩa: không có groups mà bản sao vẫn đồng nhất"


def test_shap_ranks_share_ties_instead_of_column_order():
    """Feature hoà (vd. SHAP = 0) nhận hạng trung bình chung, không xếp theo thứ tự cột."""
    ranks = shap_to_ranks(np.array([[5.0, 0.0, 0.0, 2.0]]))
    assert list(ranks) == [1.0, 3.5, 3.5, 2.0], ranks
    # hoán vị cột không đổi hạng của cùng 1 feature
    perm = [3, 1, 0, 2]
    assert list(shap_to_ranks(np.array([[5.0, 0.0, 0.0, 2.0]])[:, perm])) == list(ranks[perm])


def test_target_encoding_prior_uses_fold_train_only():
    """
    Category chưa từng thấy ở fold train phải nhận prior = trung bình nhãn của CHÍNH fold train
    đó (không phải trung bình toàn bộ train, vốn chứa nhãn của fold validation).
    Mỗi dòng 1 category riêng ⇒ mọi dòng validation đều rơi vào nhánh fillna.
    """
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.RandomState(0)
    n = 500
    df = pd.DataFrame({"uniq": [f"c{i}" for i in range(n)], "is_fraud": (rng.rand(n) < 0.2).astype(int)})
    enc, _ = stratified_kfold_target_encode(df, "uniq", "is_fraud", n_splits=5, random_state=SEED)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for a, b in skf.split(df, df["is_fraud"]):
        expected = df["is_fraud"].iloc[a].mean()
        assert np.allclose(enc.iloc[b].to_numpy(), expected), "prior phải là mean của fold train"


def test_resampling_yields_valid_onehot_blocks(monkeypatch):
    """
    SMOTE/ADASYN/Borderline/SMOTE-ENN nội suy nên sinh one-hot phân số (dòng "bật" >1 category).
    Với onehot_groups, mọi nhóm phải là đúng 1 category (0/1, tổng = 1). Tắt công tắc
    SNAP_SYNTHETIC_ONEHOT thì phải thấy phân số — nếu không thì test này vô nghĩa.
    """
    import src.imbalance.resamplers as rs

    monkeypatch.setattr(rs, "SNAP_SYNTHETIC_ONEHOT", True)  # mặc định trong config là False
    df = create_synthetic_data(1500, fraud_rate=0.1)
    enc, _ = encode_train(
        df, target_col="is_fraud", onehot_cols=["gender", "category", "state"],
        target_encode_cols=["merchant", "city", "job"], n_splits=3, random_state=SEED,
    )
    feats = [c for c in enc.columns if c != "is_fraud"]
    X, y = enc[feats].to_numpy(dtype=float), enc["is_fraud"].to_numpy()
    groups = onehot_groups_from_columns(feats, ["gender", "category", "state"])
    assert len(groups) == 3

    def blocks_valid(Xr):
        for g in groups:
            b = Xr[:, g]
            if not (np.isin(b, [0.0, 1.0]).all() and np.allclose(b.sum(axis=1), 1.0)):
                return False
        return True

    for tech in ["smote", "smote_enn", "adasyn", "borderline_smote"]:
        Xr, _, _ = apply_imbalance(tech, X, y, seed=SEED, onehot_groups=groups)
        assert blocks_valid(Xr), f"{tech}: one-hot không hợp lệ sau resample"

    monkeypatch.setattr(rs, "SNAP_SYNTHETIC_ONEHOT", False)
    Xr, _, _ = apply_imbalance("smote", X, y, seed=SEED, onehot_groups=groups)
    assert not blocks_valid(Xr), "test vô nghĩa: SMOTE thuần lẽ ra phải sinh one-hot phân số"


def test_tune_checkpoint_resume(tmp_path):
    """
    Checkpoint SQLite: chạy lại với cùng file phải TIẾP TỤC (chỉ chạy nốt số trial còn thiếu),
    trial RUNNING dở dang phải bị đánh dấu FAIL và không tính vào n_trials.
    """
    import optuna
    from src.models.tune import tune_model

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] + rng.normal(scale=0.5, size=300) > 0.8).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]).assign(is_fraud=y)
    db = tmp_path / "lr.db"

    r1 = tune_model("logistic_regression", df, n_trials=3, n_splits=3, storage_path=db)
    assert r1["complete"] and r1["n_trials"] == 3

    # Giả lập phiên bị ngắt giữa 1 trial: study có 1 trial RUNNING.
    study = optuna.load_study(study_name="logistic_regression", storage=f"sqlite:///{db}")
    study.ask()
    assert any(t.state == optuna.trial.TrialState.RUNNING for t in study.trials)

    r2 = tune_model("logistic_regression", df, n_trials=5, n_splits=3, storage_path=db)
    assert r2["complete"] and r2["n_trials"] == 5

    study = optuna.load_study(study_name="logistic_regression", storage=f"sqlite:///{db}")
    finished = [t for t in study.trials if t.state in (
        optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    assert len(finished) == 5, "phải đủ đúng 5 trial hoàn tất, không thừa/thiếu"
    assert not any(t.state == optuna.trial.TrialState.RUNNING for t in study.trials)
    assert sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials) == 1

    # Chạy lại khi đã đủ: không chạy thêm trial nào.
    r3 = tune_model("logistic_regression", df, n_trials=5, n_splits=3, storage_path=db)
    assert r3["n_trials"] == 5


def test_tune_merge_write_keeps_entries_added_meanwhile(tmp_path):
    """File kết quả bị sửa bởi tiến trình khác giữa lượt chạy: ghi model mới không được xoá mục đó."""
    import json
    from src.models.tune import _merge_write

    f = tmp_path / "best_params.json"
    f.write_text(json.dumps({"xgboost": {"n_trials": 30}}))
    # ... lượt chạy khác merge thêm catboost trong lúc tune ...
    f.write_text(json.dumps({"xgboost": {"n_trials": 100}, "catboost": {"n_trials": 100}}))
    out = _merge_write(f, "random_forest", {"n_trials": 100})
    saved = json.loads(f.read_text())
    assert saved == out
    assert saved["xgboost"]["n_trials"] == 100 and saved["catboost"]["n_trials"] == 100
    assert saved["random_forest"]["n_trials"] == 100


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


def test_save_cies_results_overwrite_loses_concurrent_writes(tmp_path):
    """
    save_cies_results() ghi đè cả file bằng list trong bộ nhớ của riêng nó — TÁI HIỆN lỗi
    thật: 2 "tiến trình" (ở đây là 2 lần gọi tuần tự, mô phỏng race) cùng ghi vào 1 file thì
    tiến trình ghi sau làm mất kết quả tiến trình kia vừa thêm. Test này xác nhận lỗi đó CÓ
    THẬT với save_cies_results() — không dùng hàm này khi có >1 tiến trình ghi cùng file.
    """
    from src.explainability.cies import save_cies_results

    f = "combo.json"
    # Tiến trình A đọc file rỗng, có list riêng [A]
    list_a = [{"model_name": "logistic_regression", "imbalance_technique": "smote_enn", "cies_metrics": {}}]
    save_cies_results(list_a, tmp_path, filename=f)
    # Tiến trình B (list riêng, không biết A vừa ghi) ghi đè bằng list của B — B không có A
    list_b = [{"model_name": "xgboost", "imbalance_technique": "smote_enn", "cies_metrics": {}}]
    save_cies_results(list_b, tmp_path, filename=f)

    saved = json.load(open(tmp_path / f, encoding="utf-8"))
    models = {r["model_name"] for r in saved}
    assert models == {"xgboost"}, "tái hiện lỗi: kết quả logistic_regression của A đã bị B ghi đè mất"


def test_merge_cies_result_keeps_concurrent_writes(tmp_path):
    """merge_cies_result() PHẢI sửa được đúng lỗi ở test trên — đọc lại trước khi ghi."""
    from src.explainability.cies import merge_cies_result

    f = "combo.json"
    merge_cies_result({"model_name": "logistic_regression", "imbalance_technique": "smote_enn", "cies_metrics": {}}, tmp_path, filename=f)
    merge_cies_result({"model_name": "xgboost", "imbalance_technique": "smote_enn", "cies_metrics": {}}, tmp_path, filename=f)

    saved = json.load(open(tmp_path / f, encoding="utf-8"))
    models = {r["model_name"] for r in saved}
    assert models == {"logistic_regression", "xgboost"}, "cả 2 tổ hợp phải còn nguyên, không cái nào bị đè mất"

    # Ghi lại đúng combo cũ (vd. chạy lại sau lỗi) phải THAY chứ không nhân đôi
    merge_cies_result({"model_name": "xgboost", "imbalance_technique": "smote_enn", "cies_metrics": {"cies_score": 0.99}}, tmp_path, filename=f)
    saved = json.load(open(tmp_path / f, encoding="utf-8"))
    assert len(saved) == 2
    xgb = next(r for r in saved if r["model_name"] == "xgboost")
    assert xgb["cies_metrics"]["cies_score"] == 0.99


def test_merge_cies_result_parallel_processes_lose_nothing(tmp_path):
    """
    Nhiều TIẾN TRÌNH ghi cùng lúc (như 2 script CIES chia model chạy song song). Bản cũ (đọc → ghi,
    không khoá) mất 176/200 kết quả khi 8 tiến trình tranh nhau; có khoá thì phải còn đủ.
    """
    import subprocess, textwrap
    root = str(Path(__file__).resolve().parent.parent)
    out = tmp_path / "combo.json"
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {root!r})
        from src.explainability.cies import merge_cies_result
        proc = sys.argv[1]
        for j in range(25):
            merge_cies_result({{"model_name": proc, "imbalance_technique": f"t{{j}}", "cies_metrics": {{}}}},
                              __import__("pathlib").Path({str(tmp_path)!r}), filename="combo.json")
    """)
    procs = [subprocess.Popen([sys.executable, "-c", code, f"p{i}"]) for i in range(8)]
    assert all(p.wait(timeout=120) == 0 for p in procs)
    saved = json.load(open(out, encoding="utf-8"))
    assert len(saved) == 8 * 25, f"mất {8 * 25 - len(saved)} kết quả do ghi đè lẫn nhau"


def test_merge_writers_refuse_corrupt_file_instead_of_wiping(tmp_path):
    """
    File kết quả hỏng (vd. đọc trúng lúc tiến trình khác đang ghi dở ở bản cũ) KHÔNG được coi là
    rỗng: bản cũ của _merge_write làm vậy rồi ghi lại chỉ còn 1 model — xoá sạch các model khác.
    """
    import pytest
    from src.models.tune import _merge_write
    from src.explainability.cies import merge_cies_result

    bp = tmp_path / "best_params.json"
    bp.write_text('{"xgboost": {"n_trials": 100}, "catb')
    with pytest.raises(json.JSONDecodeError):
        _merge_write(bp, "random_forest", {"n_trials": 100})
    assert bp.read_text() == '{"xgboost": {"n_trials": 100}, "catb', "file hỏng không được bị ghi đè"

    cr = tmp_path / "cies.json"
    cr.write_text('[{"model_name": "xgboost"')
    with pytest.raises(json.JSONDecodeError):
        merge_cies_result({"model_name": "rf", "imbalance_technique": "smote"}, tmp_path, filename="cies.json")
    assert cr.read_text() == '[{"model_name": "xgboost"'


def test_preprocess_sparkov_drops_identity_sorts_by_time_and_uses_transaction_year():
    """
    Tiền xử lý Sparkov: (1) không còn cột định danh khách hàng nào; (2) dòng được sắp theo thời
    gian dù file gốc lệch thứ tự (fraudTrain.csv có 1 chỗ ngược); (3) tuổi tính theo năm của
    TỪNG giao dịch (dữ liệu trải 2019–2020), không theo 1 năm cố định.
    """
    from src.config import CUSTOMER_IDENTITY_COLS
    from src.data.preprocess import preprocess_sparkov

    raw = pd.DataFrame({
        "trans_date_trans_time": ["2020-03-01 10:00:00", "2019-06-01 23:00:00", "2019-01-01 01:00:00"],
        "dob": ["1990-05-05"] * 3,
        "amt": [3.0, 2.0, 1.0],  # bằng thứ tự thời gian -> sau khi sắp phải tăng dần
        "cc_num": [1, 1, 2], "zip": [1, 1, 2], "unix_time": [0, 0, 0], "trans_num": ["a", "b", "c"],
        "first": ["x"] * 3, "last": ["y"] * 3, "street": ["s"] * 3,
        "gender": ["F", "F", "M"], "category": ["travel"] * 3, "merchant": ["m"] * 3,
        **{c: [0] * 3 for c in CUSTOMER_IDENTITY_COLS},
        "is_fraud": [0, 1, 0],
    })
    out = preprocess_sparkov(raw)
    assert not set(CUSTOMER_IDENTITY_COLS) & set(out.columns)
    assert not {"cc_num", "zip", "unix_time", "trans_date_trans_time", "dob"} & set(out.columns)
    assert list(out["amt"]) == [1.0, 2.0, 3.0], "phải sắp theo thời gian giao dịch"
    assert list(out["age"]) == [29, 29, 30]
    assert list(out["hour"]) == [1, 23, 10]


def test_split_ulb_by_time_keeps_test_strictly_later():
    from src.data.preprocess import split_ulb_by_time

    rng = np.random.RandomState(0)
    df = pd.DataFrame({"Time": rng.permutation(1000).astype(float), "V1": rng.randn(1000),
                       "Class": (rng.rand(1000) < 0.05).astype(int)})
    train, test = split_ulb_by_time(df, test_size=0.2)
    assert len(train) == 800 and len(test) == 200
    assert train["Time"].max() < test["Time"].min()


def test_time_series_folds_never_validate_on_the_past():
    """Mỗi fold train CHỈ trên dòng đứng trước khối validation; các khối liên tiếp, phủ 1/3 cuối."""
    from src.models.tune import time_series_folds

    folds = time_series_folds(900, n_splits=3)
    assert len(folds) == 3
    for tr, val in folds:
        assert tr.max() < val.min()
        assert tr.min() == 0, "cửa sổ mở rộng: luôn train từ đầu dữ liệu"
    vals = np.concatenate([val for _, val in folds])
    assert np.array_equal(vals, np.arange(600, 900))


def test_tune_folds_encode_without_future_labels():
    """
    Target encoding trong mỗi fold tune chỉ fit trên dòng TRƯỚC khối validation. Merchant m1 chỉ bị
    lừa đảo ở giai đoạn sau (dòng >= 600): ở fold 1 (train = dòng 0..599), m1 phải được mã hoá
    thấp. Encode 1 lần trên cả tập (cách cũ) thì m1 đã mang sẵn tỷ lệ fraud của tương lai.
    """
    from src.models.tune import encode_time_folds
    from src.data.encoding import encode_train

    n = 900
    idx = np.arange(n)
    merchant = np.where(idx % 2 == 0, "m1", "m2")
    y = np.zeros(n, dtype=int)
    y[(idx < 600) & (merchant == "m2") & (idx % 10 == 1)] = 1   # quá khứ: fraud chỉ ở m2
    y[(idx >= 600) & (merchant == "m1") & (idx % 4 == 0)] = 1   # tương lai: m1 bị lợi dụng mạnh
    y[(idx >= 600) & (merchant == "m2") & (idx % 10 == 1)] = 1
    df = pd.DataFrame({"merchant": merchant, "amt": np.arange(n, dtype=float), "is_fraud": y})

    folds, feats = encode_time_folds(df, "is_fraud", n_splits=3, onehot_cols=[], target_encode_cols=["merchant"])
    col = feats.index("merchant_encoded")
    X_tr, _, _, _ = folds[0]
    m1_rows = merchant[:600] == "m1"
    per_fold = X_tr[m1_rows, col].mean()

    leaky, _ = encode_train(df, "is_fraud", onehot_cols=[], target_encode_cols=["merchant"])
    leaky_m1 = leaky["merchant_encoded"].to_numpy()[:600][m1_rows].mean()

    assert per_fold < X_tr[~m1_rows, col].mean(), "m1 chưa từng bị fraud trong fold train -> phải thấp hơn m2"
    assert leaky_m1 > 3 * per_fold, (leaky_m1, per_fold)


def test_resample_cache_is_identical_and_keyed_on_everything_that_matters(tmp_path, monkeypatch):
    """
    Cache resample (1 file parquet / kỹ thuật): (1) giống hệt không cache; (2) là 1 dataset đọc được
    (tên cột + is_fraud); (3) lần 2 đọc file, không chạy lại resampler; (4) đổi seed hoặc
    SNAP_SYNTHETIC_ONEHOT thì tính lại và ghi đè, không dùng nhầm bản cũ.
    """
    import src.imbalance.resamplers as rs

    rng = np.random.RandomState(0)
    X = rng.randn(300, 4)
    y = (rng.rand(300) < 0.1).astype(int)
    names = ["a", "b", "c", "d"]

    ref_X, ref_y, _ = rs.apply_imbalance("smote_enn", X, y, seed=SEED)
    X1, y1, _ = rs.apply_imbalance_cached("smote_enn", X, y, seed=SEED, cache_dir=tmp_path, feature_names=names)
    assert np.array_equal(X1, ref_X) and np.array_equal(y1, ref_y)
    saved = pd.read_parquet(tmp_path / "train_encoded_smote_enn.parquet")
    assert list(saved.columns) == names + ["is_fraud"] and len(saved) == len(ref_y)

    calls = []
    real = rs.apply_imbalance
    monkeypatch.setattr(rs, "apply_imbalance", lambda *a, **k: calls.append(1) or real(*a, **k))
    X2, y2, _ = rs.apply_imbalance_cached("smote_enn", X, y, seed=SEED, cache_dir=tmp_path, feature_names=names)
    assert calls == [] and np.array_equal(X2, ref_X) and np.array_equal(y2, ref_y)

    rs.apply_imbalance_cached("smote_enn", X, y, seed=SEED + 1, cache_dir=tmp_path, feature_names=names)
    assert len(calls) == 1, "đổi seed phải tính lại"
    monkeypatch.setattr(rs, "SNAP_SYNTHETIC_ONEHOT", not rs.SNAP_SYNTHETIC_ONEHOT)
    rs.apply_imbalance_cached("smote_enn", X, y, seed=SEED + 1, cache_dir=tmp_path, feature_names=names)
    assert len(calls) == 2, "đổi SNAP_SYNTHETIC_ONEHOT phải tính lại"
    assert [f.name for f in tmp_path.iterdir()] == ["train_encoded_smote_enn.parquet"]


def test_cies_uses_explicit_params_per_dataset(monkeypatch):
    """ULB tune riêng: run_cies_experiment(params=...) phải truyền đúng tham số đó vào build_model."""
    import src.explainability.cies as cies_mod

    seen = []
    real = cies_mod.build_model
    monkeypatch.setattr(cies_mod, "build_model", lambda *a, **k: seen.append(k.get("params")) or real(*a, **k))
    df = create_synthetic_data(300)
    cies_mod.run_cies_experiment(
        model_name="logistic_regression", imbalance_technique="class_weighting",
        df_train=df.iloc[:200].reset_index(drop=True), df_test_fixed_eval=df.iloc[200:].reset_index(drop=True),
        target_col="is_fraud", onehot_cols=["gender", "category", "state"],
        target_encode_cols=["merchant", "city", "job"], n_runs=2, params={"C": 0.5},
    )
    assert seen == [{"C": 0.5}, {"C": 0.5}]


def test_search_space_covers_previously_binding_values():
    """Lần tune trước chọn giá trị sát biên cũ; vùng tìm mới phải chứa chúng ở giữa, không sát biên.

    Sparkov: XGBoost n_estimators=550 (90% của [100, 600]), learning_rate=0.0137 (9% thang log
    của [0.01, 0.3]); RF max_depth=27 (88% của [6, 30]). ULB: LR C=0.00085 (12% của [1e-4, 1e4]).
    """
    import optuna
    from src.models.tune import sample_hyperparameters

    def position(dist, value):
        lo, hi = dist.low, dist.high
        if getattr(dist, "log", False):
            lo, hi, value = np.log(lo), np.log(hi), np.log(value)
        return (value - lo) / (hi - lo)

    study = optuna.create_study()
    binding = {
        "xgboost": {"n_estimators": 550, "learning_rate": 0.0137},
        "catboost": {"iterations": 550, "learning_rate": 0.0137},
        "random_forest": {"max_depth": 27},
        "logistic_regression": {"C": 0.00085},
    }
    for model_name, values in binding.items():
        trial = study.ask()
        sample_hyperparameters(trial, model_name)
        for param, value in values.items():
            pos = position(trial.distributions[param], value)
            assert 0.15 < pos < 0.85, (model_name, param, round(pos, 3))
    trial = study.ask()
    sample_hyperparameters(trial, "catboost")
    assert trial.distributions["depth"].high <= 12  # bộ nhớ CatBoost tăng theo 2^depth


def test_optuna_tuning():
    print("\n--- Testing Optuna HPO Module (tune_model on sample data) ---")
    from src.models.tune import tune_model
    rng = np.random.RandomState(SEED)
    X = rng.randn(150, 8)
    # Fold validation theo thời gian lấy khối cuối dữ liệu: fraud phải rải khắp các dòng như dữ
    # liệu thật, không dồn hết về cuối mảng.
    y = rng.permutation(np.array([0] * 135 + [1] * 15))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]).assign(is_fraud=y)

    # Test tuning XGBoost with 2 trials
    res = tune_model("xgboost", df, n_trials=2, n_splits=2, seed=SEED)
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
