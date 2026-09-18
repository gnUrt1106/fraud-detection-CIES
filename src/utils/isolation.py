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

import logging
import multiprocessing as mp
import queue as queue_module
import traceback
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 3600  # 1 giờ — đủ cho 1 tổ hợp model×technique
POLL_INTERVAL_SECONDS = 0.5
KILL_GRACE_SECONDS = 10  # thời gian chờ sau SIGTERM/SIGKILL trước khi coi là D-state


def _subprocess_entry(func: Callable, args: tuple, kwargs: dict, result_queue: mp.Queue) -> None:
    try:
        result = func(*args, **kwargs)
        result_queue.put(("ok", result))
    except Exception as e:  # noqa: BLE001 — cố ý bắt mọi lỗi để chuyển ngược về process cha
        result_queue.put(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def _terminate_hard(process: mp.Process) -> None:
    """
    Buộc dừng 1 process con, có escalation SIGTERM -> SIGKILL.

    LÝ DO: `process.join()` KHÔNG có timeout mặc định — nếu process con đang
    treo trong 1 syscall không thể ngắt bằng SIGTERM (vd. driver GPU/CUDA bị
    deadlock ở tầng kernel), `terminate()` (SIGTERM) không giết được nó, và
    `join()` ngay sau đó sẽ đợi VÔ HẠN — vô hiệu hoá toàn bộ cơ chế timeout
    của run_isolated() (đã tái hiện: 1 job Optuna trên Kaggle "Running" hơn
    5 giờ dù timeout=3600s, vì join() bị treo ở bước terminate này).
    """
    process.terminate()
    process.join(timeout=KILL_GRACE_SECONDS)
    if process.is_alive():
        process.kill()  # SIGKILL — escalation, không thể bị process con chặn
        process.join(timeout=KILL_GRACE_SECONDS)
    if process.is_alive():
        logger.error(
            f"Process con (pid={process.pid}) vẫn sống sau SIGKILL — có thể "
            f"đang ở trạng thái D (uninterruptible, thường do driver GPU/kernel "
            f"treo). Không thể ép dừng từ tầng Python; có thể cần khởi động lại "
            f"container/kernel."
        )


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
        # daemon=False (KHÔNG phải True): process daemon bị Python cấm tự
        # sinh process con — RandomForestClassifier(n_jobs=-1) cần joblib
        # spawn worker (backend "loky") để chạy song song, nên với
        # daemon=True nó bị ép về n_jobs=1 (rất chậm). Ép joblib dùng
        # backend "threading" thay thế KHÔNG ổn định giữa các version
        # sklearn/joblib (đã tái hiện: hoạt động ở local nhưng KHÔNG hoạt
        # động trên container Kaggle — có lẽ version khác không tôn trọng
        # context manager parallel_backend() cùng cách). daemon=False loại
        # bỏ hẳn giới hạn gốc, không phụ thuộc version. Đánh đổi: nếu
        # process CHA (kernel Jupyter/Kaggle) bị kill đột ngột ngoài tầm
        # kiểm soát code này, process con daemon=False có thể không tự
        # chết theo — chấp nhận được vì (1) timeout ở dưới vẫn tự
        # terminate/kill process con trong mọi đường thoát do CODE NÀY quản
        # lý, (2) trên Kaggle cả container bị dọn sạch khi session dừng.
        daemon=False,
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
                _terminate_hard(process)
                raise TimeoutError(
                    f"Subprocess cho '{func_name}' vượt quá timeout={timeout}s "
                    f"— có thể bị treo (hang)."
                )

    process.join(timeout=KILL_GRACE_SECONDS)
    if status == "error":
        raise RuntimeError(f"Lỗi bên trong subprocess '{func_name}':\n{payload}")
    return payload
