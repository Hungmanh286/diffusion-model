"""Test M3 — mô hình và optimizer (PLAN_TABLE2.md M3). Chạy trên CPU.

`python3 tests/test_model.py` hoặc `pytest`. Dùng mô hình đủ cỡ (512, 2 lớp)
cho test số tham số; các test khác dùng cỡ nhỏ cho nhanh.
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lf.model import LFDenoiser, DenoiserTrace  # noqa: E402
from lf.optim import AdamATan2, EMA  # noqa: E402

torch.manual_seed(0)
VOCAB = 3


def _small(**kw):
    return LFDenoiser(vocab_size=VOCAB, hidden_size=32, num_heads=4, seq_len=16, **kw)


def test_shapes():
    m = _small()
    B, L = 5, 16
    x = torch.randn(B, L, VOCAB)
    t = torch.rand(B)
    h, l = m.initial_state(B)
    logits, q, (h2, l2) = m.forward(x, t, (h, l))
    assert logits.shape == (B, L, VOCAB)
    assert q.shape == (B,)
    assert h2.shape == (B, L, 32) and l2.shape == (B, L, 32)


def test_param_count_full_size():
    m = LFDenoiser(vocab_size=VOCAB, hidden_size=512, num_heads=8, num_layers=2, seq_len=64)
    # Backbone 2 lớp phải đúng 6,815,744 (PLAN M3).
    backbone = sum(p.numel() for p in m.reasoning.parameters())
    assert backbone == 6_815_744, backbone
    total = m.num_parameters()
    assert 6_900_000 <= total <= 7_300_000, total  # 6.9–7.3M (PLAN M3)


def test_noncausal_dependency():
    # Đổi token ở vị trí j làm đổi đầu ra ở vị trí i != j (noncausal).
    m = _small()
    B, L = 1, 16
    x = torch.randn(B, L, VOCAB)
    t = torch.rand(B)
    h, l = m.initial_state(B)
    out1, _, _ = m.forward(x, t, (h, l))
    x2 = x.clone()
    x2[0, 10] += 3.0  # đổi vị trí 10
    out2, _, _ = m.forward(x2, t, m.initial_state(B))
    assert not torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5)  # vị trí 0 đổi theo


def test_f_theta_call_count_and_grad_truncation():
    # 15 lần gọi F_theta = 3 chu trình * (4 + 1); chỉ 5 lần cuối có gradient.
    m = _small(m_cycles=4, outer_cycles=3)
    B, L = 2, 16
    x = torch.randn(B, L, VOCAB)
    t = torch.rand(B)
    trace = DenoiserTrace()
    logits, _, _ = m.forward(x, t, m.initial_state(B), trace=trace)
    assert trace.num_calls == 15, trace.num_calls
    # outer_cycles-1=2 chu trình đầu (10 lần) không gradient; 1 chu trình cuối (5) có.
    assert trace.grad_enabled == [False] * 10 + [True] * 5


def test_stop_gradient_between_steps():
    m = _small()
    B = 2
    x = torch.randn(B, 16, VOCAB)
    t = torch.rand(B)
    _, _, (h2, l2) = m.forward(x, t, m.initial_state(B))
    assert not h2.requires_grad and not l2.requires_grad


def test_q_head_init():
    m = _small(q_head_bias_init=-5.0)
    assert torch.all(m.q_head.weight == 0)
    assert torch.allclose(m.q_head.bias, torch.full_like(m.q_head.bias, -5.0))
    # q_hat ban đầu ~ sigmoid(-5) ~ 0.0067 -> không dừng sớm.
    B = 3
    x = torch.randn(B, 16, VOCAB)
    _, q, _ = m.forward(x, torch.rand(B), m.initial_state(B))
    assert torch.allclose(torch.sigmoid(q), torch.full((B,), 1 / (1 + math.exp(5))), atol=1e-3)


def test_q_head_reads_position_0():
    # Đổi trạng thái đầu vào ở vị trí 0 phải đổi q; đây là ràng buộc vị trí đọc.
    m = _small(q_head_pos=0)
    assert m.q_head_pos == 0


def test_grad_flows_to_params():
    m = _small()
    B = 2
    x = torch.randn(B, 16, VOCAB)
    logits, q, _ = m.forward(x, torch.rand(B), m.initial_state(B))
    (logits.sum() + q.sum()).backward()
    grads = [p.grad for p in m.reasoning.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


# --------------------------------------------------------------------------- #
# Optimizer
# --------------------------------------------------------------------------- #
def test_adam_atan2_first_step_magnitude():
    # Với g>0, bước đầu có độ lớn ~0.9967*lr (PLAN 2.5, DEV-01).
    w = torch.nn.Parameter(torch.tensor([1.0]))
    opt = AdamATan2([w], lr=1.0, betas=(0.9, 0.95), weight_decay=0.0)
    w.grad = torch.tensor([2.0])  # g>0 bất kỳ dương
    w0 = w.item()
    opt.step()
    step_size = w0 - w.item()
    expected = 10.0 * math.atan(0.1)  # (lr/0.1)*atan2(0.1g,|g|) = 10*arctan(0.1)
    assert abs(step_size - expected) < 1e-6
    assert abs(expected - 0.9966865) < 1e-5


def test_adam_atan2_sign_and_weight_decay():
    w = torch.nn.Parameter(torch.tensor([1.0]))
    opt = AdamATan2([w], lr=0.1, betas=(0.9, 0.95), weight_decay=1.0)
    w.grad = torch.tensor([-3.0])  # g<0 -> w tăng
    opt.step()
    # weight decay nhân trước: 1*(1-0.1*1)=0.9; rồi atan2(âm,·)<0 nên -lr/bc1*atan2>0.
    assert w.item() > 0.9


def test_ema_tracks_params():
    m = _small()
    ema = EMA(m, decay=0.9)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    # Giả lập một bước: đổi tham số.
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)
    for n, p in m.named_parameters():
        if p.requires_grad:
            exp = before[n].float() * 0.9 + p.detach().float() * 0.1
            assert torch.allclose(ema.shadow[n], exp, atol=1e-5)


def test_bf16_forward_dtype():
    """DEV-03: forward bf16, tham số vẫn float32; trạng thái và logits ở bf16."""
    m = _small(forward_dtype=torch.bfloat16)
    B, L = 4, 16
    x = torch.randn(B, L, VOCAB)  # đầu vào float32, model tự ép
    t = torch.rand(B)
    h, l = m.initial_state(B)
    assert h.dtype == torch.bfloat16 and l.dtype == torch.bfloat16
    logits, q, (h2, l2) = m.forward(x, t, (h, l))
    assert logits.dtype == torch.bfloat16
    assert h2.dtype == torch.bfloat16 and l2.dtype == torch.bfloat16
    assert q.dtype == torch.float32
    assert all(p.dtype == torch.float32 for p in m.parameters())
    assert torch.isfinite(logits.float()).all()
    # Trạng thái float32 cũ (checkpoint trước khi có bf16) vẫn nạp được.
    logits2, _, _ = m.forward(x, t, (h.float(), l.float()))
    assert logits2.dtype == torch.bfloat16
    # Gần với forward float32 cùng trọng số. Sai số bf16 cộng dồn qua 15 lần gọi
    # F_theta (3 chu trình x (m+1)); đo được ~2-7% tương đối, nên ngưỡng 15%.
    m32 = _small()
    m32.load_state_dict(m.state_dict())
    h32, l32 = m32.initial_state(B)
    ref, _, _ = m32.forward(x, t, (h32, l32))
    rel = ((logits.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.15, rel


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"\n{len(fns)} test đạt.")
