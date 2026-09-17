"""
isolation.py — Chạy 1 hàm trong process con (subprocess) riêng biệt.

Lý do tồn tại module này:
    torch (dùng cho ANN) và xgboost cùng nạp vào 1 process có thể
    segfault hoặc treo (hang) vô thời hạn do xung đột OpenMP runtime
    (2 bản libomp khác nhau cùng tồn tại trong 1 process — đã xác nhận
    tái hiện được trên môi trường dev của project này).

    notebooks/03_train_models.ipynb và notebooks/04_cies_experiment.ipynb
    đều lặp qua cả 5 model (bao gồm 'ann' và 'xgboost') trong CÙNG 1
    kernel/process, nên mỗi lần train 1 model PHẢI được cô lập trong
    process con riêng — dùng run_isolated() thay vì gọi hàm train trực
    tiếp trong vòng lặp.

Dùng context 'spawn' (không dùng 'fork', kể cả trên Linux nơi 'fork' là
mặc định): 'fork' nhân bản toàn bộ memory image của process cha — nếu
torch/xgboost đã được nạp ở process cha thì process con vẫn kế thừa
nguyên trạng thái đó, không giải quyết được xung đột. 'spawn' khởi động
1 interpreter Python hoàn toàn mới, chỉ import đúng những gì cần cho
func — đảm bảo cô lập thật sự. Cách này hoạt động nhất quán trên cả máy
local (macOS) lẫn Kaggle/Colab (Linux).

RÀNG BUỘC: func phải là 1 hàm top-level trong 1 module import được (không
dùng lambda/closure), và mọi tham số + giá trị trả về phải pickle được.
"""

import multiprocessing as mp
import queue as queue_module
import traceback
from typing import Any, Callable, Optional

DEFAULT_TIMEOUT_SECONDS = 3600  # 1 giờ — đủ cho 1 tổ hợp model×technique
POLL_INTERVAL_SECONDS = 0.5


def _subprocess_entry(func: Callable, args: tuple, kwargs: dict, result_queue: mp.Queue) -> None:
    try:
        result = func(*args, **kwargs)
        result_queue.put(("ok", result))
    except Exception as e:  # noqa: BLE001 — cố ý bắt mọi lỗi để chuyển ngược về process cha
        result_queue.put(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def run_isolated(
    func: Callable,
    *args: Any,
    timeout: Optional[float] = DEFAULT_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> Any:
    """
    Chạy func(*args, **kwargs) trong 1 process con (spawn) riêng biệt.

    Args:
        func: Hàm top-level cần chạy cô lập (vd. train_and_evaluate_combo,
            run_cies_experiment)
        *args, **kwargs: Tham số truyền cho func
        timeout: Số giây tối đa chờ kết quả trước khi coi là bị treo (hang)
            và huỷ process con. None = chờ vô hạn (không khuyến khích).

    Returns:
        Giá trị trả về từ func (đã đi qua subprocess).

    Raises:
        RuntimeError: Nếu process con crash (segfault, exception) trước
            khi trả kết quả.
        TimeoutError: Nếu vượt quá `timeout` giây mà chưa có kết quả.
    """
    func_name = getattr(func, "__name__", str(func))
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    process = ctx.Process(
        target=_subprocess_entry,
        args=(func, args, kwargs, result_queue),
        daemon=True,
    )
    process.start()

    waited = 0.0
    while True:
        try:
            status, payload = result_queue.get(timeout=POLL_INTERVAL_SECONDS)
            break
        except queue_module.Empty:
            if not process.is_alive():
                process.join()
                raise RuntimeError(
                    f"Subprocess cho '{func_name}' bị crash (exit code "
                    f"{process.exitcode}) trước khi trả kết quả — có thể do xung "
                    f"đột thư viện native (torch/xgboost OpenMP) hoặc lỗi khác "
                    f"không bắt được ở tầng Python."
                )
            waited += POLL_INTERVAL_SECONDS
            if timeout is not None and waited >= timeout:
                process.terminate()
                process.join()
                raise TimeoutError(
                    f"Subprocess cho '{func_name}' vượt quá timeout={timeout}s "
                    f"— có thể bị treo (hang)."
                )

    process.join()
    if status == "error":
        raise RuntimeError(f"Lỗi bên trong subprocess '{func_name}':\n{payload}")
    return payload
