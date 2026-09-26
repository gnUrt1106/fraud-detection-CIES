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
