"""Test M4 — lõi flow (PLAN_TABLE2.md M4). CPU, numpy/torch."""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lf import flow  # noqa: E402

torch.manual_seed(0)
V = 3


def _onehot(ids, v=V):
    return torch.nn.functional.one_hot(ids, v).float()


def test_interpolant_endpoints():
    B, L = 2, 8
    x0 = torch.randn(B, L, V)
    x1 = _onehot(torch.randint(1, V, (B, L)))
    i0 = flow.interpolant(x0, x1, torch.zeros(B))
    i1 = flow.interpolant(x0, x1, torch.ones(B))
    assert torch.allclose(i0, x0, atol=1e-6)
    assert torch.allclose(i1, x1, atol=1e-6)


def test_interpolant_clue_fixed_all_t():
    B, L = 2, 8
    x0 = torch.randn(B, L, V)
    x1 = _onehot(torch.randint(1, V, (B, L)))
    clue = torch.zeros(B, L, dtype=torch.bool)
    clue[:, 0] = True
    for t in (0.0, 0.3, 0.7, 1.0):
        it = flow.interpolant(x0, x1, torch.full((B,), t), clue_mask=clue)
        assert torch.allclose(it[:, 0], x1[:, 0], atol=1e-6)


def test_sorted_sampler_increasing():
    t = flow.sorted_time_sampler(16, 100)
    assert t.shape == (100, 17)
    assert torch.all(t[:, 1:] >= t[:, :-1])
    assert torch.all((t >= 0) & (t <= 1))


def test_random_start_sampler():
    t = flow.random_start_time_sampler(16, 200)
    assert t.shape == (200, 17)
    assert torch.all(t[:, 1:] >= t[:, :-1])
    t0 = t[:, 0]
    # mọi giá trị còn lại >= t0
    assert torch.all(t >= t0.unsqueeze(1) - 1e-6)


def test_stablemax_ce_hand_value():
    # logits (1,0,-1) -> s=(2,1,0.5), tổng 3.5. Nhãn 0 -> -log(2/3.5).
    logits = torch.tensor([[1.0, 0.0, -1.0]])
    labels = torch.tensor([0])
    mask = torch.tensor([True])
    ce = flow.stablemax_cross_entropy(logits, labels, mask)
    assert abs(ce.item() - (-math.log(2 / 3.5))) < 1e-9
    # Nhãn 2 -> -log(0.5/3.5)
    ce2 = flow.stablemax_cross_entropy(logits, torch.tensor([2]), mask)
    assert abs(ce2.item() - (-math.log(0.5 / 3.5))) < 1e-9


def test_stablemax_ce_ignores_invalid():
    logits = torch.randn(1, 4, V)
    labels = torch.randint(0, V, (1, 4))
    mask = torch.tensor([[True, False, True, False]])
    per = flow.stablemax_cross_entropy(logits, labels, mask)
    assert per[0, 1].item() == 0.0 and per[0, 3].item() == 0.0


def test_loss_mask_excludes_clue():
    labels = torch.tensor([[1, 2, 1, 2]])
    clue = torch.tensor([[True, False, False, True]])
    mask = flow.loss_valid_mask(labels, clue)
    assert mask.tolist() == [[False, True, True, False]]


def test_act_label():
    logits = torch.zeros(2, 4, V)
    labels = torch.tensor([[1, 2, 1, 2], [1, 2, 1, 2]])
    # chuỗi 0: dự đoán đúng hết; chuỗi 1: sai một ô
    for j, lab in enumerate(labels[0]):
        logits[0, j, lab] = 10.0
        logits[1, j, lab] = 10.0
    logits[1, 2] = 0.0
    logits[1, 2, 0] = 10.0  # sai ô 2
    mask = torch.ones(2, 4, dtype=torch.bool)
    q = flow.act_label(logits, labels, mask)
    assert q.tolist() == [True, False]


def test_act_label_ignores_clue():
    # Ô clue sai không làm hỏng nhãn q (chỉ xét ô không clue).
    logits = torch.zeros(1, 3, V)
    labels = torch.tensor([[1, 2, 1]])
    for j, lab in enumerate(labels[0]):
        logits[0, j, lab] = 10.0
    logits[0, 0] = 0.0  # làm "sai" ô 0...
    logits[0, 0, 0] = 10.0
    mask = torch.tensor([[False, True, True]])  # ...nhưng ô 0 là clue -> bỏ qua
    assert bool(flow.act_label(logits, labels, mask).item()) is True


def test_exploration_frequency_and_bounds():
    g = torch.Generator().manual_seed(1)
    m = flow.exploration_min_halt_steps(20000, 16, 0.1, generator=g)
    forced = m > 0
    freq = forced.float().mean().item()
    assert abs(freq - 0.1) < 0.01
    assert torch.all(m[forced] >= 2) and torch.all(m[forced] <= 16)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"\n{len(fns)} test đạt.")
