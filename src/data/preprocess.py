"""
preprocess.py — Làm sạch feature Sparkov và chia train/test theo thời gian (Sparkov + ULB).

Chia theo THỜI GIAN (train = quá khứ, test = tương lai), như cách đánh giá chuẩn trong literature
fraud (Le Borgne et al., Fraud Detection Handbook; Dal Pozzolo et al., TNNLS 2018) và đúng cách
Sparkov được phát hành (fraudTrain.csv → fraudTest.csv). Dữ liệu trả về luôn đã sắp theo thời gian:
`tune.py` chia fold validation theo thứ tự dòng nên dựa vào điều này.
"""

import pandas as pd

from src.config import ULB_TARGET_COL, CUSTOMER_IDENTITY_COLS

TIMESTAMP_COL = "trans_date_trans_time"

# Cột mã/định danh không phải feature.
ID_COLS = ["Unnamed: 0", "trans_num", "cc_num", "first", "last", "street", "zip", "unix_time"]


def preprocess_sparkov(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sắp theo thời gian, thêm `hour`, `day_of_week`, `age`, rồi bỏ cột ID và mọi cột định danh
    khách hàng (`config.CUSTOMER_IDENTITY_COLS`).

    Returns:
        DataFrame chỉ còn feature + `is_fraud`, thứ tự dòng theo thời gian giao dịch.
    """
    data = df.sort_values(TIMESTAMP_COL, kind="mergesort").reset_index(drop=True)
    ts = pd.to_datetime(data[TIMESTAMP_COL])
    data["hour"] = ts.dt.hour
    data["day_of_week"] = ts.dt.dayofweek
    # Tuổi tại thời điểm giao dịch: dữ liệu trải 2019–2020 nên lấy năm của từng dòng.
    data["age"] = ts.dt.year - pd.to_datetime(data["dob"]).dt.year

    drop = ID_COLS + CUSTOMER_IDENTITY_COLS + [TIMESTAMP_COL, "dob"]
    return data.drop(columns=[c for c in drop if c in data.columns])


ULB_EXPECTED_ROWS = 284_807  # creditcard.csv gốc, trước khi bỏ dòng trùng


def dedupe_ulb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bỏ dòng trùng KHỚP HOÀN TOÀN ở mọi cột (kể cả `Time` và `Class`), giữ dòng đầu tiên.

    ULB gốc có 1.081 dòng như vậy (19 fraud). Nếu giữ lại, cùng một giao dịch có thể rơi vào cả
    train lẫn test (rò rỉ) hoặc bị đếm đôi. Thứ tự các dòng còn lại giữ nguyên; không đổi nhãn nào.
    """
    return df.loc[~df.duplicated(keep="first")].reset_index(drop=True)


def prepare_ulb(df: pd.DataFrame, test_size: float = 0.20):
    """
    ULB gốc -> (train, test, report): bỏ dòng trùng, rồi chia theo thời gian (`split_ulb_by_time`).

    `report` lưu số liệu để truy vết: số dòng trùng, kích thước train/test, và số dòng test trùng
    29 đặc trưng với một dòng train (chỉ để báo cáo, KHÔNG loại bỏ).
    """
    from src.config import ULB_FEATURE_COLS

    n0, f0 = len(df), int(df[ULB_TARGET_COL].sum())
    dup = df.duplicated(keep="first")
    n_dup, f_dup = int(dup.sum()), int(df.loc[dup, ULB_TARGET_COL].sum())
    deduped = dedupe_ulb(df)
    train, test = split_ulb_by_time(deduped, test_size=test_size)

    h_train = set(pd.util.hash_pandas_object(train[ULB_FEATURE_COLS], index=False).tolist())
    h_test = pd.util.hash_pandas_object(test[ULB_FEATURE_COLS], index=False)
    report = {
        "rows_before": n0, "fraud_before": f0,
        "duplicate_rows": n_dup, "duplicate_fraud_rows": f_dup,
        "rows_after": len(deduped), "fraud_after": int(deduped[ULB_TARGET_COL].sum()),
        "train_rows": len(train), "train_fraud": int(train[ULB_TARGET_COL].sum()),
        "test_rows": len(test), "test_fraud": int(test[ULB_TARGET_COL].sum()),
        "train_time_max": float(train["Time"].max()), "test_time_min": float(test["Time"].min()),
        "test_rows_equal_train_on_29_features": int(h_test.isin(h_train).sum()),
    }
    return train, test, report


def split_ulb_by_time(df: pd.DataFrame, test_size: float = 0.20):
    """
    ULB không có sẵn file train/test: sắp theo `Time` (giây từ giao dịch đầu) và lấy phần
    `test_size` cuối làm test. Giữ cột `Time` (notebook 05 bỏ nó khỏi feature qua `ULB_FEATURE_COLS`).
    """
    data = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    cut = int(round(len(data) * (1 - test_size)))
    train, test = data.iloc[:cut].reset_index(drop=True), data.iloc[cut:].reset_index(drop=True)
    if train[ULB_TARGET_COL].sum() == 0 or test[ULB_TARGET_COL].sum() == 0:
        raise ValueError("Một phía của phép chia theo thời gian không có fraud nào")
    return train, test
