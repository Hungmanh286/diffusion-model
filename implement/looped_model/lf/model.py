"""M3 — Denoiser Looped Flows có trạng thái (PLAN_TABLE2.md M3, SPEC §3.1).

Dựng trên TRM (ref/models: trm.py, layers.py). Khác TRM ở đầu vào: thay token
embedding rời rạc bằng
  - chiếu interpolant nhiễu I_t (|V| -> 512, không bias), nhân sqrt(512),
  - cộng embedding thời gian MLP(1->512->512, SiLU) vào mọi vị trí.
Phần hồi quy giữ nguyên TRM: z=(h, l); mỗi lần gọi denoiser lặp 3 chu trình
(outer_cycles), mỗi chu trình cập nhật l `m` lần rồi cập nhật h một lần; 2 chu
trình đầu không gradient (trm.py:208-216); trạng thái truyền ra bị detach
(trm.py:219). Đầu q đọc h ở vị trí 0 (trm.py:221; Q1/D5).

Đối chiếu tham số (khớp 6,815,744 backbone): attention mỗi lớp 4*512^2, SwiGLU
3*512*1536, hai lớp = 6,815,744 (PLAN M3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- #
# Lớp cơ bản (tái hiện từ ref/models/layers.py, commit c011037)
# --------------------------------------------------------------------------- #
def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Truncated LeCun normal, cắt ở ±2 độ lệch (như models/common.py của TRM)."""
    if std == 0:
        with torch.no_grad():
            tensor.zero_()
        return tensor
    with torch.no_grad():
        return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.to(torch.float32)
    var = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return x.to(dtype)


def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    orig = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    q = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q.to(orig), k.to(orig)


class Linear(nn.Module):
    """Linear với truncated LeCun normal init (như CastedLinear của TRM)."""

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty(out_features, in_features), std=1.0 / (in_features**0.5))
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_pos: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = nn.Buffer(emb.cos(), persistent=False)
        self.sin = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos, self.sin


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, causal: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal
        self.qkv_proj = Linear(hidden_size, 3 * num_heads * self.head_dim, bias=False)
        self.o_proj = Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, cos_sin, x):
        b, s, _ = x.shape
        qkv = self.qkv_proj(x).view(b, s, 3 * self.num_heads, self.head_dim)
        q = qkv[:, :, : self.num_heads]
        k = qkv[:, :, self.num_heads : 2 * self.num_heads]
        v = qkv[:, :, 2 * self.num_heads :]
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary(q, k, cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # B H S D
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.inter = inter
        self.gate_up_proj = Linear(hidden_size, inter * 2, bias=False)
        self.down_proj = Linear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Block(nn.Module):
    """Post-norm: RMSNorm sau mỗi phép cộng residual (trm.py:90-104)."""

    def __init__(self, hidden_size: int, num_heads: int, expansion: float, eps: float):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads, causal=False)
        self.mlp = SwiGLU(hidden_size, expansion)
        self.eps = eps

    def forward(self, cos_sin, x):
        x = rms_norm(x + self.attn(cos_sin, x), self.eps)
        x = rms_norm(x + self.mlp(x), self.eps)
        return x


class ReasoningModule(nn.Module):
    """Mạng dùng chung F_theta: thêm input injection rồi chạy các block."""

    def __init__(self, blocks: list[Block]):
        super().__init__()
        self.layers = nn.ModuleList(blocks)

    def forward(self, hidden, injection, cos_sin):
        hidden = hidden + injection
        for layer in self.layers:
            hidden = layer(cos_sin, hidden)
        return hidden


# --------------------------------------------------------------------------- #
# Denoiser Looped Flows
# --------------------------------------------------------------------------- #
@dataclass
class DenoiserTrace:
    """Ghi lại từng lần gọi F_theta để test (số lần gọi, có gradient hay không)."""

    grad_enabled: list[bool] = field(default_factory=list)

    @property
    def num_calls(self) -> int:
        return len(self.grad_enabled)


class LFDenoiser(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        expansion: float = 4.0,
        m_cycles: int = 4,
        outer_cycles: int = 3,
        seq_len: int = 64,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        q_head_pos: int = 0,
        q_head_bias_init: float = -5.0,
        z_init_std: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.m_cycles = m_cycles
        self.outer_cycles = outer_cycles
        self.seq_len = seq_len
        self.q_head_pos = q_head_pos
        self.dtype = dtype
        self.embed_scale = math.sqrt(hidden_size)

        # Đầu vào liên tục: chiếu interpolant |V| -> hidden (không bias) và MLP thời gian.
        self.input_proj = Linear(vocab_size, hidden_size, bias=False)
        self.time_mlp = nn.Sequential(
            Linear(1, hidden_size, bias=True),
            nn.SiLU(),
            Linear(hidden_size, hidden_size, bias=True),
        )

        self.reasoning = ReasoningModule(
            [Block(hidden_size, num_heads, expansion, rms_norm_eps) for _ in range(num_layers)]
        )
        self.rotary = RotaryEmbedding(hidden_size // num_heads, seq_len, rope_theta)

        self.lm_head = Linear(hidden_size, vocab_size, bias=False)
        self.q_head = Linear(hidden_size, 2, bias=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(q_head_bias_init)

        self.h_init = nn.Buffer(trunc_normal_init_(torch.empty(hidden_size), std=z_init_std), persistent=True)
        self.l_init = nn.Buffer(trunc_normal_init_(torch.empty(hidden_size), std=z_init_std), persistent=True)

    # -- trạng thái -------------------------------------------------------- #
    def initial_state(self, batch_size: int, device=None):
        device = device or self.h_init.device
        h = self.h_init.view(1, 1, -1).expand(batch_size, self.seq_len, self.hidden_size).to(device)
        l = self.l_init.view(1, 1, -1).expand(batch_size, self.seq_len, self.hidden_size).to(device)
        return h.contiguous(), l.contiguous()

    def _injection(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """e_t = sqrt(H) * input_proj(I_t) + time_emb(t), cộng vào mọi vị trí."""
        inp = self.embed_scale * self.input_proj(x_t)
        temb = self.time_mlp(t.reshape(-1, 1).to(x_t.dtype))  # (B, H)
        return inp + temb.unsqueeze(1)

    def _one_cycle(self, h, l, e, cos_sin):
        for _ in range(self.m_cycles):
            l = self.reasoning(l, h + e, cos_sin)  # l <- F(l + h + e)
        h = self.reasoning(h, l, cos_sin)          # h <- F(h + l)
        return h, l

    def forward(self, x_t, t, state, trace: DenoiserTrace | None = None):
        """Một lần gọi denoiser.

        x_t:  (B, L, |V|) interpolant nhiễu (liên tục).
        t:    (B,) hoặc scalar, thời gian trong [0, 1].
        state: (h, l), mỗi cái (B, L, H).
        Trả về: logits (B, L, |V|), (h, l) mới đã detach.
        """
        h, l = state
        cos_sin = self.rotary()
        e = self._injection(x_t, t)

        if trace is not None:
            # Bọc reasoning để đếm số lần gọi và cờ gradient.
            orig = self.reasoning.forward

            def wrapped(hidden, injection, cs):
                trace.grad_enabled.append(torch.is_grad_enabled())
                return orig(hidden, injection, cs)

            self.reasoning.forward = wrapped  # type: ignore[assignment]

        try:
            # outer_cycles-1 chu trình đầu không gradient (trm.py:208-216).
            with torch.no_grad():
                for _ in range(self.outer_cycles - 1):
                    h, l = self._one_cycle(h, l, e, cos_sin)
            # Chu trình cuối có gradient.
            h, l = self._one_cycle(h, l, e, cos_sin)
        finally:
            if trace is not None:
                self.reasoning.forward = orig  # type: ignore[assignment]

        logits = self.lm_head(h)
        q_halt_logit = self.q_head(h[:, self.q_head_pos]).to(torch.float32)[..., 0]
        new_state = (h.detach(), l.detach())  # stop-gradient giữa các bước (trm.py:219)
        return logits, q_halt_logit, new_state

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
