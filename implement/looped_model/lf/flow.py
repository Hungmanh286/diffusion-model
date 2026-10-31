"""M4 — Lõi flow: interpolant, bộ lấy mẫu thời gian, StableMax CE, nhãn ACT,
exploration (PLAN_TABLE2.md M4, SPEC §3.1-3.3).

Quy ước: x one-hot trên |V| lớp. Ô clue (D4) chỉ gồm ô có quân clue, luôn giữ
đúng giá trị clue ở mọi t; interpolant và loss chỉ áp dụng lên ô còn lại.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_LABEL_ID = -100


# --------------------------------------------------------------------------- #
# Interpolant và nhiễu
# --------------------------------------------------------------------------- #
def sample_noise(shape, sigma: float, generator=None, device=None, dtype=torch.float32):
    """x0 ~ N(0, sigma^2 I)."""
    return sigma * torch.randn(*shape, generator=generator, device=device, dtype=dtype)


def interpolant(x0, x1, t, clue_mask=None):
    """I_t = (1-t) x0 + t x1 trên ô không clue; ô clue giữ đúng x1 (D4).

    x0, x1: (B, L, V); t: (B,) hoặc scalar; clue_mask: (B, L) bool True=cố định.
    """
    t = _broadcast_t(t, x1)
    it = (1 - t) * x0 + t * x1
    if clue_mask is not None:
        it = torch.where(clue_mask.unsqueeze(-1), x1, it)
    return it


def _broadcast_t(t, ref):
    t = torch.as_tensor(t, dtype=ref.dtype, device=ref.device)
    if t.ndim == 0:
        return t
    return t.view(-1, *([1] * (ref.ndim - 1)))


# --------------------------------------------------------------------------- #
# Bộ lấy mẫu thời gian (SPEC 3.3 / App. E.1)
# --------------------------------------------------------------------------- #
def sorted_time_sampler(k: int, size: int, generator=None, device=None):
    """`sorted`: sắp k+1 giá trị độc lập U[0,1] -> t_0 < ... < t_k."""
    u = torch.rand(size, k + 1, generator=generator, device=device)
    return torch.sort(u, dim=1).values


def random_start_time_sampler(k: int, size: int, generator=None, device=None):
    """`random-start`: t0 ~ U[0,1], rồi sắp t0 cùng k giá trị U[t0,1]."""
    t0 = torch.rand(size, 1, generator=generator, device=device)
    rest = t0 + (1 - t0) * torch.rand(size, k, generator=generator, device=device)
    allt = torch.cat([t0, rest], dim=1)
    return torch.sort(allt, dim=1).values


def make_time_sampler(name: str):
    if name == "sorted":
        return sorted_time_sampler
    if name == "random-start":
        return random_start_time_sampler
    raise ValueError(f"time sampler không rõ: {name}")


# --------------------------------------------------------------------------- #
# StableMax cross-entropy (khớp ref/models/losses.py)
# --------------------------------------------------------------------------- #
def _stablemax(x):
    return torch.where(x < 0, 1.0 / (1.0 - x + 1e-30), x + 1.0)


def log_stablemax(x, dim=-1):
    s = _stablemax(x)
    return torch.log(s / s.sum(dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, valid_mask):
    """CE theo StableMax, tính ở float64; token không hợp lệ cho 0 (losses.py:24-32).

    logits: (..., V); labels: (...) chỉ số lớp; valid_mask: (...) bool.
    Trả về loss theo từng token (cùng shape labels).
    """
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    safe_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
    picked = torch.gather(logprobs, dim=-1, index=safe_labels.long().unsqueeze(-1)).squeeze(-1)
    return -torch.where(valid_mask, picked, torch.zeros_like(picked))


def mean_ce_per_sequence(logits, labels, valid_mask):
    """Mean StableMax CE mỗi chuỗi: tổng trên token hợp lệ / số token hợp lệ."""
    per_tok = stablemax_cross_entropy(logits, labels, valid_mask)
    counts = valid_mask.sum(-1).clamp_min(1)
    return per_tok.sum(-1) / counts


def loss_valid_mask(labels, clue_mask):
    """Ô tính loss: có nhãn và KHÔNG phải ô clue (D4)."""
    mask = labels != IGNORE_LABEL_ID
    if clue_mask is not None:
        mask = mask & (~clue_mask)
    return mask


# --------------------------------------------------------------------------- #
# Nhãn ACT và exploration (SPEC 3.2 / trm.py)
# --------------------------------------------------------------------------- #
def act_label(logits, labels, valid_mask):
    """q = 1 nếu mọi ô hợp lệ (không clue) được dự đoán đúng (losses.py:70-71)."""
    preds = logits.argmax(-1)
    correct = valid_mask & (preds == labels)
    return correct.sum(-1) == valid_mask.sum(-1)


def exploration_min_halt_steps(size, k, prob, generator=None, device=None):
    """Với xác suất `prob` ép chạy ít nhất m_min ~ U{2..k} bước (trm.py:286-287).

    Trả về tensor int (size,): m_min cho vị trí bị ép, 0 cho vị trí không ép.
    """
    forced = torch.rand(size, generator=generator, device=device) < prob
    m_min = torch.randint(2, k + 1, (size,), generator=generator, device=device)
    return torch.where(forced, m_min, torch.zeros_like(m_min))
