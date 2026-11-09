"""
supersede_results.py - Chuyển các tổ hợp của một (hoặc vài) model ra khỏi file kết quả `results/rerun/<tên>.json` sang thư mục lưu trữ, có sao lưu.

Dùng khi tổ hợp cũ không còn hợp lệ (vd. 5 mục CatBoost chạy GPU + SHAP Approximate trong colab_full_gpu.json) và phải chạy lại: `train` coi tổ hợp có
trong file là "đã xong" và bỏ qua, `merge` thì báo trùng. Mặc định CHỈ IN những gì sẽ chuyển; thêm --apply mới ghi.

    python scripts/supersede_results.py --file results/rerun/colab_full_gpu.json --models catboost --dest-dir results/rerun_superseded
    python scripts/supersede_results.py --file results/rerun/colab_full_gpu.json --models catboost --dest-dir results/rerun_superseded --apply

Ghi: <dest-dir>/<tên file>.before_supersede (bản gốc nguyên vẹn), <dest-dir>/<tên>_superseded_<models>.json (các mục đã chuyển), rồi ghi lại file gốc
chỉ còn phần còn lại (atomic). Không ghi đè bản sao lưu đã có.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def supersede(path: Path, models, dest_dir: Path, apply: bool):
    """Trả (số mục chuyển, số mục còn lại, danh sách tổ hợp chuyển). Ném ValueError nếu không có mục nào khớp."""
    path, dest_dir = Path(path), Path(dest_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{path.name} không có danh sách 'results'")
    models = set(models)
    moved = [r for r in results if r.get("model_name") in models]
    kept = [r for r in results if r.get("model_name") not in models]
    if not moved:
        raise ValueError(f"Không có tổ hợp nào của {sorted(models)} trong {path.name}")
    names = [r.get("combination") for r in moved]
    if not apply:
        return len(moved), len(kept), names

    dest_dir.mkdir(parents=True, exist_ok=True)
    backup = dest_dir / f"{path.name}.before_supersede"
    out = dest_dir / f"{path.stem}_superseded_{'_'.join(sorted(models))}.json"
    for p in (backup, out):
        if p.exists():
            raise FileExistsError(f"{p} đã tồn tại; không ghi đè. Chuyển nó đi hoặc dùng --dest-dir khác")
    shutil.copy2(path, backup)
    out.write_text(json.dumps({**{k: v for k, v in data.items() if k != "results"}, "results": moved,
                               "note": f"đã chuyển khỏi {path.name}: tổ hợp cũ không còn hợp lệ, cần chạy lại"},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({**data, "results": kept}, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return len(moved), len(kept), names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--dest-dir", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    try:
        n_moved, n_kept, names = supersede(a.file, a.models, a.dest_dir, a.apply)
    except (ValueError, FileExistsError) as e:
        print(f"[LỖI] {e}")
        return 2
    print(f"{'ĐÃ CHUYỂN' if a.apply else 'SẼ CHUYỂN (chưa ghi, thêm --apply)'} {n_moved} tổ hợp, còn lại {n_kept} trong {a.file.name}:")
    for n in names:
        print("  -", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
