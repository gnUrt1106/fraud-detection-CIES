"""
deeplift_native.py — DeepLIFT (quy tắc Rescale) theo lô cho ANN dạng MLP, thay cho vòng lặp từng dòng của shap.DeepExplainer.

Cùng thuật toán với shap.DeepExplainer (PyTorch): với mỗi dòng x và mỗi dòng nền r, hệ số nhân của từng ReLU là
    m = (ReLU(a_x) - ReLU(a_r)) / (a_x - a_r)          nếu |a_x - a_r| >= 1e-6,
    m = ReLU'(a_r)                                      nếu không (giống nonlinear_1d của shap),
các lớp tuyến tính (Linear, BatchNorm1d ở chế độ eval) giữ nguyên gradient, Dropout là phép đồng nhất. Giá trị SHAP của x là
    phi(x) = trung bình_r [ g(x, r) * (x - r) ],  g = gradient "đã nhân hệ số" của logit theo đầu vào, lan truyền ngược
    trên cặp (x, r).
Khác biệt duy nhất: shap lan truyền ngược từng dòng x một (mỗi lần trên toàn bộ nền), còn đây tính cả một lô dòng x × toàn
bộ nền bằng phép nhân ma trận trên GPU → nhanh hơn hàng trăm lần mà kết quả giống đến sai số làm tròn float32
(đã so khớp với shap.DeepExplainer trong tests/test_deeplift_native.py).

Chỉ hỗ trợ Sequential gồm Linear / ReLU / BatchNorm1d / Dropout và đầu ra 1 neuron (logit) — đúng FraudDetectorANN.
Kiến trúc khác → báo lỗi, KHÔNG lùi âm thầm.
"""

from typing import Callable, List, Tuple

import numpy as np

_EPS_DELTA = 1e-6          # ngưỡng của shap.nonlinear_1d


def _parse_mlp(network) -> List[Tuple]:
    import torch.nn as nn

    ops: List[Tuple] = []
    for layer in network:
        if isinstance(layer, nn.Linear):
            ops.append(("linear", layer.weight.detach(), layer.bias.detach()))
        elif isinstance(layer, nn.ReLU):
            ops.append(("relu",))
        elif isinstance(layer, nn.BatchNorm1d):
            if layer.running_mean is None:
                raise ValueError("BatchNorm1d không có thống kê chạy (track_running_stats=False) — không hỗ trợ")
            inv = (layer.running_var.detach() + layer.eps).rsqrt()
            scale = inv * (layer.weight.detach() if layer.weight is not None else 1.0)
            shift = (layer.bias.detach() if layer.bias is not None else 0.0) - layer.running_mean.detach() * scale
            ops.append(("affine", scale, shift))
        elif isinstance(layer, nn.Dropout):
            continue                                  # eval: phép đồng nhất
        else:
            raise ValueError(f"Lớp {type(layer).__name__} chưa được hỗ trợ bởi DeepLIFT native")
    if not ops or ops[-1][0] != "linear" or ops[-1][1].shape[0] != 1:
        raise ValueError("Mạng phải kết thúc bằng Linear(…, 1) (logit)")
    return ops


def _forward_pre_relu(ops, Z):
    """Chạy xuôi, trả (danh sách đầu vào của từng ReLU theo thứ tự, logit)."""
    import torch

    pres, h = [], Z
    for op in ops:
        if op[0] == "linear":
            h = h @ op[1].T + op[2]
        elif op[0] == "relu":
            pres.append(h)
            h = torch.relu(h)
        else:
            h = h * op[1] + op[2]
    return pres, h


def make_deeplift_fn(ann, background, chunk: int = 0) -> Callable[[np.ndarray], np.ndarray]:
    """
    Trả hàm f(X_đã_scale) -> ndarray (n, n_features): SHAP (DeepLIFT Rescale, trung bình trên nền) của LOGIT.

    Args:
        ann: nn.Module (FraudDetectorANN) đã train, có thuộc tính `.network`.
        background: tensor (R, d) đã scale, nằm trên cùng thiết bị với ann.
        chunk: số dòng x mỗi lô (0 = tự chọn theo thiết bị: GPU 128, CPU 16). Chỉ ảnh hưởng RAM/tốc độ.
    """
    import torch

    ann.eval()
    device = background.device
    ops = _parse_mlp(ann.network)
    ops = [tuple(t.to(device=device, dtype=torch.float32) if torch.is_tensor(t) else t for t in op) for op in ops]
    bg = background.to(dtype=torch.float32)
    R, d = bg.shape
    if ops[0][0] != "linear" or ops[0][1].shape[1] != d:
        raise ValueError(f"Lớp đầu của mạng không khớp số feature ({d})")
    pre_r, _ = _forward_pre_relu(ops, bg)                 # (R, n_k) cho từng ReLU: tính MỘT lần
    relu_r = [torch.relu(p) for p in pre_r]
    grad_r = [(p > 0).to(torch.float32) for p in pre_r]    # ReLU'(a_r), dùng khi |Δ| nhỏ
    step = chunk if chunk > 0 else (128 if device.type == "cuda" else 16)

    def fn(X) -> np.ndarray:
        x_all = X if torch.is_tensor(X) else torch.as_tensor(np.asarray(X), dtype=torch.float32)
        x_all = x_all.to(device=device, dtype=torch.float32)
        out = np.empty((x_all.shape[0], d), dtype=np.float32)
        with torch.no_grad():
            for a in range(0, x_all.shape[0], step):
                x = x_all[a:a + step]                                   # (B, d)
                pre_x, _ = _forward_pre_relu(ops, x)
                mult = []                                               # hệ số nhân từng ReLU, (B, R, n_k)
                for k in range(len(pre_x)):
                    px = pre_x[k][:, None, :]
                    d_in = px - pre_r[k][None, :, :]
                    d_out = torch.relu(px) - relu_r[k][None, :, :]
                    ratio = d_out / torch.where(d_in == 0, torch.ones_like(d_in), d_in)
                    mult.append(torch.where(d_in.abs() < _EPS_DELTA, grad_r[k][None, :, :].expand_as(d_in), ratio))
                    del d_in, d_out, ratio
                # Lan truyền ngược trên từng cặp (x, r): bắt đầu từ ∂logit/∂logit = 1.
                g = None
                k = len(mult)
                for op in reversed(ops):
                    if op[0] == "linear":
                        g = op[1].expand(1, 1, -1) if g is None else g @ op[1]     # (·,·,out) @ (out,in)
                    elif op[0] == "relu":
                        k -= 1
                        g = g * mult[k]
                    else:
                        g = g * op[1]
                # g: (B, R, d) → phi = trung bình_r g * (x - r)
                phi = (g * (x[:, None, :] - bg[None, :, :])).mean(dim=1)
                out[a:a + x.shape[0]] = phi.cpu().numpy()
        return out

    return fn
