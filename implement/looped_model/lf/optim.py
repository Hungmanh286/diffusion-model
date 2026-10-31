"""M3 — Optimizer Adam-atan2 (DEV-01 phương án B) và EMA (PLAN 2.3, 2.5).

Adam-atan2 bản PyTorch thuần, khớp đúng công thức kernel CUDA của gói
`adam-atan2==0.0.3` (csrc/adam_atan2.cu, a=b=1):
    w <- w (1 - lr*wd)
    m <- lerp(m, g, 1-b1);  v <- lerp(v, g^2, 1-b2)
    w <- w - (lr / (1-b1^t)) * atan2(m, sqrt(v) / sqrt(1-b2^t))
Kiểm bằng số: bước đầu với g>0 có độ lớn ~0.9967*lr (PLAN 2.5), test trong
tests/test_optim.py. Nếu phương án A (gói CUDA) cài được trên server thì so khớp.

EMA khớp TRM (ref/models/ema.py): chỉ tham số requires_grad, khởi tạo từ trọng
số bước 0, KHÔNG hiệu chỉnh bias, cập nhật sau mỗi batch.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim.optimizer import Optimizer


class AdamATan2(Optimizer):
    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
    ):
        if lr <= 0:
            raise ValueError("lr phải > 0")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas phải trong [0,1)")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("AdamATan2 không hỗ trợ gradient sparse")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                m, v = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # Decoupled weight decay (nhân, như kernel).
                if wd != 0:
                    p.mul_(1 - lr * wd)

                m.lerp_(g, 1 - b1)
                v.lerp_(g.square(), 1 - b2)

                bc1 = 1 - b1**t
                bc2 = 1 - b2**t
                denom = (v.sqrt() / math.sqrt(bc2))
                p.add_(torch.atan2(m, denom), alpha=-lr / bc1)
        return loss


class EMA:
    """Trung bình trượt mũ của tham số (khớp TRM ema.py). Không hiệu chỉnh bias."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: p.detach().clone().float()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(p.detach().float(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        """Nạp trọng số EMA vào model (dùng khi đánh giá)."""
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].to(p.dtype))

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.shadow = sd["shadow"]


def build_lr_lambda(warmup_steps: int, schedule: str = "constant"):
    """Hệ số LR theo bước: warmup tuyến tính tới 1.0, sau đó giữ hằng (NQ)."""
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0

    if schedule != "constant":
        raise NotImplementedError("NQ dùng constant; schedule khác chưa cần")
    return fn
