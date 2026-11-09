"""
prepare_ulb.py - Bỏ dòng trùng của ULB gốc rồi chia train/test theo thời gian (80/20).

    python scripts/prepare_ulb.py --raw path/creditcard.csv            # hoặc thư mục chứa nó
    python scripts/prepare_ulb.py --raw path/ --out data/processed     # mặc định --out là data/processed

Đầu vào chấp nhận: `creditcard.csv` (284.807 dòng) hoặc cặp `ulb_train.parquet` + `ulb_test.parquet`
GỐC (chưa bỏ trùng; ghép lại phải đủ 284.807 dòng). Đầu ra: `ulb_train.parquet`, `ulb_test.parquet`
và `bao_cao_bo_trung.json` trong --out. Số kỳ vọng (đã kiểm): 1.081 dòng trùng (19 fraud),
283.726 dòng, train 226.981 (399 fraud), test 56.745 (74 fraud).

Không ghi đè file đã có nếu chưa thêm --force (tránh làm lệch kết quả đang chạy).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import ULB_EXPECTED_ROWS, prepare_ulb  # noqa: E402

EXPECTED = {"duplicate_rows": 1081, "duplicate_fraud_rows": 19, "rows_after": 283_726,
            "train_rows": 226_981, "train_fraud": 399, "test_rows": 56_745, "test_fraud": 74}


def load_raw(raw: Path):
    """Trả (DataFrame ULB gốc, tên nguồn). Thư mục thì tìm creditcard.csv rồi tới cặp parquet."""
    if raw.is_file():
        return pd.read_csv(raw), raw.name
    csv = raw / "creditcard.csv"
    if csv.exists():
        return pd.read_csv(csv), csv.name
    tr, te = raw / "ulb_train.parquet", raw / "ulb_test.parquet"
    if tr.exists() and te.exists():
        return pd.concat([pd.read_parquet(tr), pd.read_parquet(te)], ignore_index=True), "ulb_train+ulb_test.parquet"
    raise SystemExit(f"[LỖI] Không thấy creditcard.csv hoặc ulb_train/ulb_test.parquet trong {raw}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="creditcard.csv hoặc thư mục chứa dữ liệu ULB gốc")
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--force", action="store_true", help="ghi đè ulb_train/test.parquet đã có")
    args = ap.parse_args(argv)

    targets = [args.out / n for n in ("ulb_train.parquet", "ulb_test.parquet", "bao_cao_bo_trung.json")]
    if not args.force and any(t.exists() for t in targets):
        print(f"[BỎ QUA] {args.out} đã có ulb_train/test/bao_cao; thêm --force nếu thật sự muốn làm lại.")
        return 0

    df, source = load_raw(args.raw)
    if len(df) != ULB_EXPECTED_ROWS:
        raise SystemExit(f"[LỖI] ULB gốc phải có {ULB_EXPECTED_ROWS:,} dòng, đang có {len(df):,} "
                         f"(có thể đã bỏ trùng rồi). Dùng creditcard.csv gốc.")
    train, test, report = prepare_ulb(df)
    report = {"created": datetime.now().isoformat(timespec="seconds"), "source": source, **report}
    mism = {k: (report[k], v) for k, v in EXPECTED.items() if report[k] != v}
    if mism:
        print(f"[CẢNH BÁO] Lệch số kỳ vọng (thực tế, kỳ vọng): {mism}")

    args.out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(targets[0], index=False)
    test.to_parquet(targets[1], index=False)
    targets[2].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
