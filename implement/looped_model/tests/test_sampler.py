"""Test M4 — sampler Algorithm 2 với denoiser oracle (PLAN M4). CPU."""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lf import sampler as sp  # noqa: E402

torch.manual_seed(0)
V = 3


def _onehot(ids, v=V):
    return torch.nn.functional.one_hot(ids, v).float()


def _oracle(x1):
    def fn(x_bar, s, state):
        return x1, state
    return fn


def test_uniform_grid():
    g = sp.uniform_grid(128)
    assert g.shape == (129,)
    assert g[0] == 0.0 and abs(g[-1] - 1.0) < 1e-6
    d = g[1:] - g[:-1]
    assert torch.allclose(d, d[0] * torch.ones_like(d), atol=1e-6)


def test_oracle_reaches_x1_ode():
    B, L = 4, 8
    x1 = _onehot(torch.randint(1, V, (B, L)))
    x0 = torch.randn(B, L, V)
    x = sp.sample(_oracle(x1), x0, n_steps=128, gamma=0.0, sigma=1.0, state0=None)
    assert torch.equal(sp.round_to_tokens(x), x1.argmax(-1))


def test_oracle_reaches_x1_sde():
    B, L = 4, 8
    x1 = _onehot(torch.randint(1, V, (B, L)))
    x0 = torch.randn(B, L, V)
    x = sp.sample(_oracle(x1), x0, n_steps=128, gamma=5.0, sigma=1.0, state0=None)
    # Bước cuối (tnext=1) rơi đúng x1 với oracle, bất kể gamma.
    assert torch.equal(sp.round_to_tokens(x), x1.argmax(-1))


def test_clue_fixed_through_trajectory():
    B, L = 3, 8
    x1 = _onehot(torch.randint(1, V, (B, L)))
    x0 = torch.randn(B, L, V)
    clue = torch.zeros(B, L, dtype=torch.bool)
    clue[:, :2] = True
    _, traj = sp.sample(
        _oracle(x1), x0, n_steps=16, gamma=5.0, sigma=1.0, state0=None,
        clue_mask=clue, x1_clue=x1, return_trajectory=True,
    )
    for xt in traj:
        assert torch.allclose(xt[:, :2], x1[:, :2], atol=1e-6)


def test_gamma0_is_euler_interpolant():
    # gamma=0, oracle: x_{t_i} nằm đúng trên interpolant (1-t)x0 + t x1.
    B, L = 2, 4
    x1 = _onehot(torch.randint(1, V, (B, L)))
    x0 = torch.randn(B, L, V)
    n = 10
    _, traj = sp.sample(
        _oracle(x1), x0, n_steps=n, gamma=0.0, sigma=1.0, state0=None,
        return_trajectory=True,
    )
    grid = sp.uniform_grid(n)
    for i, xt in enumerate(traj):
        expected = (1 - grid[i]) * x0 + grid[i] * x1
        assert torch.allclose(xt, expected, atol=1e-5)


def test_backtrack_marginal_mean_std():
    # App. C: nếu x_{t_i} ~ marginal interpolant tại t_i (mean t_i x1, std sig(1-t_i)),
    # thì x_bar_s có mean s x1, std sig(1-s). Kiểm công thức Eq.18 bằng Monte Carlo.
    sigma = 1.0
    ti = 0.4
    gamma = 5.0
    dt = 0.05
    a = max(0.0, min(1.0, 1 - gamma * dt))
    s = a * ti
    x1 = torch.tensor([0.0, 1.0, 2.0])  # giá trị x1 theo chiều để kiểm mean
    N = 200000
    g = torch.Generator().manual_seed(7)
    xi = torch.randn(N, 3, generator=g)
    eta = torch.randn(N, 3, generator=g)
    x_ti = ti * x1 + sigma * (1 - ti) * xi            # marginal tại t_i
    eps = sigma * eta
    coef = (max(0.0, (1 - s) ** 2 - (a - s) ** 2)) ** 0.5
    x_bar = a * x_ti + coef * eps
    assert torch.allclose(x_bar.mean(0), s * x1, atol=0.02)
    assert torch.allclose(x_bar.std(0), torch.full((3,), sigma * (1 - s)), atol=0.02)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"\n{len(fns)} test đạt.")
