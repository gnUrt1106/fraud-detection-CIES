"""Bỏ dòng trùng ULB và chia theo thời gian (src/data/preprocess.py::prepare_ulb)."""

import numpy as np
import pandas as pd
import pytest

from src.config import ULB_FEATURE_COLS, ULB_TARGET_COL
from src.data.preprocess import dedupe_ulb, prepare_ulb


def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = rng.uniform(0, 100, n).round(2)
    df["Time"] = np.arange(n, dtype=float)
    df[ULB_TARGET_COL] = 0
    df.loc[::10, ULB_TARGET_COL] = 1          # 20 fraud rải đều theo thời gian
    return df[["Time"] + ULB_FEATURE_COLS + [ULB_TARGET_COL]]


def test_dedupe_removes_only_fully_identical_rows_and_keeps_first():
    df = _toy()
    dup = pd.concat([df, df.iloc[[3, 3, 10]]], ignore_index=True)       # 3 bản sao
    out = dedupe_ulb(dup)
    assert len(out) == len(df)
    pd.testing.assert_frame_equal(out, df)                               # thứ tự và nhãn không đổi


def test_rows_equal_on_features_but_different_time_or_label_are_kept():
    df = _toy()
    same_feat_new_time = df.iloc[[5]].copy()
    same_feat_new_time["Time"] += 0.5
    same_feat_new_label = df.iloc[[7]].copy()
    same_feat_new_label[ULB_TARGET_COL] = 1 - same_feat_new_label[ULB_TARGET_COL]
    out = dedupe_ulb(pd.concat([df, same_feat_new_time, same_feat_new_label], ignore_index=True))
    assert len(out) == len(df) + 2


def test_prepare_report_counts_and_time_split():
    df = _toy()
    dup = pd.concat([df, df.iloc[[0, 10, 11]]], ignore_index=True)       # 0 và 10 là fraud, 11 thì không
    train, test, rep = prepare_ulb(dup)
    assert rep["rows_before"] == 203 and rep["duplicate_rows"] == 3 and rep["duplicate_fraud_rows"] == 2
    assert rep["rows_after"] == 200 and rep["fraud_after"] == 20
    assert len(train) + len(test) == 200 and len(test) == 40
    assert rep["train_rows"] == len(train) and rep["test_rows"] == len(test)
    assert train["Time"].max() <= test["Time"].min()                     # train = quá khứ
    assert train["Time"].is_monotonic_increasing and test["Time"].is_monotonic_increasing
    assert train[ULB_TARGET_COL].sum() + test[ULB_TARGET_COL].sum() == 20


def test_prepare_does_not_mutate_input():
    df = _toy()
    before = df.copy()
    prepare_ulb(df)
    pd.testing.assert_frame_equal(df, before)


def test_cross_split_feature_duplicates_are_reported_not_removed():
    df = _toy()
    leak = df.iloc[[2]].copy()                                           # y hệt dòng train về 29 đặc trưng
    leak["Time"] = df["Time"].max() + 1                                  # nhưng nằm ở cuối => thuộc test
    train, test, rep = prepare_ulb(pd.concat([df, leak], ignore_index=True))
    assert rep["test_rows_equal_train_on_29_features"] == 1
    assert len(train) + len(test) == len(df) + 1                         # không bị loại


def test_prepare_raises_when_a_side_has_no_fraud():
    df = _toy()
    df[ULB_TARGET_COL] = 0
    df.loc[0, ULB_TARGET_COL] = 1                                        # fraud chỉ ở train
    with pytest.raises(ValueError):
        prepare_ulb(df)
