"""
encoding.py — Encoding pipeline cho categorical features.

Thứ tự pipeline (spec mục 3.3):
    1. Split train/test THEO THỜI GIAN (src/data/preprocess.py)
    2. Encoding: fit trên train → transform test
    3. Imbalance handling: CHỈ áp dụng lên train, SAU khi encode

Encoding strategy (spec mục 3.1, cột theo config.ONEHOT_COLS / TARGET_ENCODE_COLS):
    - One-hot: gender, category (low cardinality)
    - Stratified K-fold Target Encoding: merchant (high cardinality)
    (state, city, job đã bị bỏ khỏi feature — là cột định danh khách hàng, xem AGENT_SPEC §8)

RÀNG BUỘC:
    - PHẢI dùng StratifiedKFold, KHÔNG KFold thường
    - Fit chỉ trên train; test dùng thống kê toàn train để transform
    - Mỗi CIES run phải tính lại encoding riêng (không dùng lại từ run trước)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from typing import Dict, Tuple, Optional

from src.config import (
    TARGET_COL, ONEHOT_COLS, TARGET_ENCODE_COLS, SEED
)


def stratified_kfold_target_encode(
    df_train: pd.DataFrame,
    col: str,
    target_col: str,
    n_splits: int = 5,
    smoothing: int = 10,
    random_state: int = SEED,
    groups: Optional[np.ndarray] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Stratified K-fold Target Encoding cho 1 cột categorical.

    Pseudo-code bắt buộc tuân thủ từ spec mục 3.2.
    Dùng StratifiedKFold (KHÔNG KFold thường) để đảm bảo mỗi fold
    có tỷ lệ fraud/legit cân bằng.

    Args:
        df_train: DataFrame train
        col: Tên cột categorical cần encode
        target_col: Tên cột target (is_fraud)
        n_splits: Số fold cho StratifiedKFold
        smoothing: Hệ số smoothing — category hiếm bị kéo về global_mean
        random_state: Seed cho reproducibility
        groups: (tuỳ chọn) id nhóm cho từng dòng — các dòng cùng nhóm luôn nằm CÙNG 1 fold.
            Dùng cho bootstrap (có dòng trùng): nếu bản sao của 1 dòng rơi vào fold train còn
            dòng đó ở fold validation thì nhãn của chính nó lọt vào giá trị encode của nó
            (rò rỉ). Khi có `groups` dùng StratifiedGroupKFold (vẫn stratified theo target).

    Returns:
        encoded: Series encoded values cho train (out-of-fold)
        full_mapping: Series mapping tính trên toàn bộ train (dùng cho test)
    """
    # Đảm bảo index tuần tự tránh lỗi reindexing
    df_reset = df_train.reset_index(drop=True)
    if groups is None:
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        splits = skf.split(df_reset, df_reset[target_col])
    else:
        skf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        splits = skf.split(df_reset, df_reset[target_col], groups=np.asarray(groups))
    encoded = pd.Series(index=df_reset.index, dtype=float)
    global_mean = df_reset[target_col].mean()

    for train_idx, val_idx in splits:
        fold_train = df_reset.iloc[train_idx]
        # Prior làm mịn tính CHỈ từ fold train. Dùng trung bình toàn bộ train (bản cũ, theo
        # pseudo-code gốc của spec) thì nhãn của chính các dòng validation lọt vào prior —
        # rò rỉ nhỏ nhưng có thật, và dễ loại bỏ hoàn toàn.
        fold_mean = fold_train[target_col].mean()
        stats = fold_train.groupby(col)[target_col].agg(["mean", "count"])
        # Smoothing: category hiếm bị kéo về fold_mean
        smoothed = (
            stats["count"] * stats["mean"] + smoothing * fold_mean
        ) / (stats["count"] + smoothing)
        encoded.iloc[val_idx] = (
            df_reset[col].iloc[val_idx].map(smoothed).fillna(fold_mean)
        )

    # Full mapping trên toàn bộ train — dùng cho test set
    full_stats = df_reset.groupby(col)[target_col].agg(["mean", "count"])
    full_mapping = (
        full_stats["count"] * full_stats["mean"] + smoothing * global_mean
    ) / (full_stats["count"] + smoothing)

    encoded.index = df_train.index
    return encoded, full_mapping


def encode_train(
    df_train: pd.DataFrame,
    target_col: str = TARGET_COL,
    onehot_cols: Optional[list] = None,
    target_encode_cols: Optional[list] = None,
    n_splits: int = 5,
    smoothing: int = 10,
    random_state: int = SEED,
    fixed_categories: Optional[Dict[str, list]] = None,
    groups: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode toàn bộ categorical features cho tập train.

    Args:
        df_train: DataFrame train (raw features + target)
        target_col: Tên cột target
        onehot_cols: Danh sách cột one-hot (mặc định theo config)
        target_encode_cols: Danh sách cột target encoding (mặc định theo config)
        n_splits: Số fold cho StratifiedKFold
        smoothing: Hệ số smoothing
        random_state: Seed
        fixed_categories: Dict danh sách categories cố định cho từng cột onehot (đảm bảo đồng nhất số feature qua bootstrap runs)
        groups: id nhóm mỗi dòng (vd. id dòng gốc của bootstrap) — xem stratified_kfold_target_encode

    Returns:
        df_encoded: DataFrame đã encode (không chứa cột categorical gốc)
        encoding_maps: Dict chứa thông tin encoding để tái lập cho test
    """
    if onehot_cols is None:
        onehot_cols = ONEHOT_COLS
    if target_encode_cols is None:
        target_encode_cols = TARGET_ENCODE_COLS

    # Reset index đảm bảo sạch sẽ cho concat dummy và target encoding
    df_encoded = df_train.reset_index(drop=True)
    encoding_maps = {
        "global_mean": df_encoded[target_col].mean(),
        "target_mappings": {},
        "onehot_cols": onehot_cols,
        "onehot_categories": {},
    }

    # --- One-hot encoding ---
    for col in onehot_cols:
        if col in df_encoded.columns:
            if fixed_categories and col in fixed_categories:
                cats = fixed_categories[col]
                series_cat = pd.Categorical(df_encoded[col], categories=cats)
                dummies = pd.get_dummies(series_cat, prefix=col, dtype=float)
            else:
                dummies = pd.get_dummies(df_encoded[col], prefix=col, dtype=float)
            encoding_maps["onehot_categories"][col] = list(dummies.columns)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(columns=[col], inplace=True)

    # --- Stratified K-fold Target Encoding ---
    for col in target_encode_cols:
        if col in df_encoded.columns:
            encoded_values, full_mapping = stratified_kfold_target_encode(
                df_train, col, target_col,
                n_splits=n_splits,
                smoothing=smoothing,
                random_state=random_state,
                groups=groups,
            )
            # .to_numpy(): gán theo VỊ TRÍ. encoded_values mang index gốc của df_train,
            # còn df_encoded đã reset_index — gán thẳng Series sẽ căn theo NHÃN index và
            # sinh NaN/lệch hàng khi df_train có index không mặc định (vd. ngay sau
            # train_test_split hoặc .sample()).
            df_encoded[f"{col}_encoded"] = encoded_values.to_numpy()
            encoding_maps["target_mappings"][col] = full_mapping
            df_encoded.drop(columns=[col], inplace=True)

    return df_encoded, encoding_maps


def encode_test(
    df_test: pd.DataFrame,
    encoding_maps: Dict,
    target_encode_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Encode tập test dùng thống kê đã tính trên TOÀN BỘ train.

    RÀNG BUỘC (spec mục 3.2):
    - Test dùng full mapping từ toàn bộ train (không k-fold)
    - KHÔNG fit lại trên test

    Args:
        df_test: DataFrame test (raw features)
        encoding_maps: Dict encoding maps từ encode_train()
        target_encode_cols: Danh sách cột target encoding

    Returns:
        df_encoded: DataFrame test đã encode
    """
    if target_encode_cols is None:
        target_encode_cols = TARGET_ENCODE_COLS

    # reset_index: cột one-hot dựng từ pd.Categorical luôn có index 0..n-1, ghép (concat) theo NHÃN
    # index — df_test có index khác (vd. 1 lát cắt .iloc[...] không reset) sẽ bị ghép lệch hàng,
    # nhân đôi số dòng và sinh NaN. Giống encode_train: kết quả luôn có index 0..n-1.
    df_encoded = df_test.reset_index(drop=True)
    global_mean = encoding_maps["global_mean"]

    # --- One-hot encoding ---
    for col in encoding_maps["onehot_cols"]:
        if col in df_encoded.columns:
            expected_cols = encoding_maps["onehot_categories"][col]
            prefix = f"{col}_"
            cats = [c[len(prefix):] for c in expected_cols if c.startswith(prefix)]
            if cats:
                series_cat = pd.Categorical(df_encoded[col], categories=cats)
                dummies = pd.get_dummies(series_cat, prefix=col, dtype=float)
            else:
                dummies = pd.get_dummies(df_encoded[col], prefix=col, dtype=float)
                for expected_col in expected_cols:
                    if expected_col not in dummies.columns:
                        dummies[expected_col] = 0.0
                dummies = dummies[expected_cols]
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(columns=[col], inplace=True)

    # --- Target Encoding (dùng full mapping từ train) ---
    for col in target_encode_cols:
        if col in df_encoded.columns:
            full_mapping = encoding_maps["target_mappings"][col]
            df_encoded[f"{col}_encoded"] = (
                df_encoded[col].map(full_mapping).fillna(global_mean)
            )
            df_encoded.drop(columns=[col], inplace=True)

    return df_encoded
