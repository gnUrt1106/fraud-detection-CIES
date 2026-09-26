"""
jsonio.py — Cập nhật file JSON kết quả an toàn khi nhiều tiến trình cùng ghi.

Dùng cho `results/best_params.json` (tune) và `results/cies_summary_results*.json` (CIES): các
file này từng được nhiều tiến trình ghi song song (vd. 2 script CIES chia model, hoặc tune RF local
trong lúc merge kết quả Kaggle vào).
"""

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional


@contextmanager
def _exclusive_lock(path: Path):
    lock_path = path.with_name(path.name + ".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def update_json(
    path: Path,
    update: Callable[[Any], Any],
    default: Callable[[], Any],
    serializer: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """
    Đọc → `update(nội dung)` → ghi, giữ khoá độc quyền suốt quá trình và ghi atomic (file tạm +
    os.replace), nên không tiến trình nào đọc phải bản ghi dở hay ghi đè mất cập nhật của tiến trình
    khác. File chưa tồn tại thì bắt đầu từ `default()`. File HỎNG (JSON lỗi) thì báo lỗi thay vì coi
    như rỗng — coi như rỗng rồi ghi lại sẽ xoá sạch kết quả của mọi model/tổ hợp khác.

    Returns:
        Nội dung sau khi ghi.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(path):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                current = json.load(f)
        else:
            current = default()
        new = update(current)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(new, f, default=serializer, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    return new
