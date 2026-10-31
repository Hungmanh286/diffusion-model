"""M4 — Sampler suy luận (Algorithm 2, noise-backtracking) — PLAN M4, SPEC §4.1.

Lưới thời gian 0 = t_0 < ... < t_n = 1 (đều, D6). Với mỗi bước (App. C, Eq.18-19):
    a   = clip(1 - gamma (t_{i+1} - t_i), 0, 1)
    s   = a t_i
    eps ~ p0
    x_bar_s = a x_{t_i} + sqrt((1-s)^2 - (a-s)^2) eps          (Eq. 18)
    x_hat_s = D_s(x_bar_s, state)                              (denoiser)
    x_{t_{i+1}} = x_bar_s + (t_{i+1}-s) (x_hat_s - x_bar_s)/(1-s)  (Eq. 19)
gamma=0 suy biến về Euler (Eq. 8). Ô clue (D4) được giữ đúng giá trị clue ở mọi
bước: trong x_bar_s, trong x_hat_s và trong x_{t_{i+1}}.
"""
from __future__ import annotations

import torch

from .flow import sample_noise


def uniform_grid(n: int, device=None, dtype=torch.float32):
    """0 = t_0 < ... < t_n = 1, đều."""
    return torch.linspace(0.0, 1.0, n + 1, device=device, dtype=dtype)


def _fix_clue(x, clue_mask, x1_clue):
    if clue_mask is None:
        return x
    return torch.where(clue_mask.unsqueeze(-1), x1_clue, x)


@torch.no_grad()
def sample(
    denoise_fn,
    x0,
    n_steps: int,
    gamma: float,
    sigma: float,
    state0,
    clue_mask=None,
    x1_clue=None,
    generator=None,
    return_trajectory: bool = False,
):
    """Chạy sampler từ nhiễu x0.

    denoise_fn(x_bar, s, state) -> (x_hat, new_state): x_hat là dự đoán x1 (liên
        tục, (B, L, V)); dùng oracle khi test.
    x0: (B, L, V) nhiễu khởi đầu. state0: trạng thái ban đầu của denoiser.
    clue_mask: (B, L) bool; x1_clue: (B, L, V) giá trị clue cố định.
    Trả về x cuối (B, L, V); nếu return_trajectory thì kèm danh sách x theo bước.
    """
    device = x0.device
    grid = uniform_grid(n_steps, device=device, dtype=x0.dtype)
    x = _fix_clue(x0, clue_mask, x1_clue)
    state = state0
    traj = [x] if return_trajectory else None

    for i in range(n_steps):
        ti = grid[i]
        tnext = grid[i + 1]
        a = torch.clamp(1.0 - gamma * (tnext - ti), 0.0, 1.0)
        s = a * ti
        eps = sample_noise(x.shape, sigma, generator=generator, device=device, dtype=x.dtype)
        var = (1.0 - s) ** 2 - (a - s) ** 2
        coef = torch.sqrt(torch.clamp(var, min=0.0))
        x_bar = a * x + coef * eps
        x_bar = _fix_clue(x_bar, clue_mask, x1_clue)

        x_hat, state = denoise_fn(x_bar, s, state)
        x_hat = _fix_clue(x_hat, clue_mask, x1_clue)

        # Bước Euler lùi từ s về t_{i+1} (Eq. 19). 1 - s > 0 vì s = a t_i <= t_i < 1.
        x = x_bar + (tnext - s) * (x_hat - x_bar) / (1.0 - s)
        x = _fix_clue(x, clue_mask, x1_clue)
        if return_trajectory:
            traj.append(x)

    if return_trajectory:
        return x, traj
    return x


def round_to_tokens(x):
    """round(x̂)^l = onehot(argmax) -> trả về chỉ số token (cũng là token id)."""
    return x.argmax(-1)
