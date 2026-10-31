"""Test M7 — harness đánh giá với denoiser oracle (PLAN M7). CPU, không cần GPU."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lf import eval_table2 as ev  # noqa: E402
from lf.data_nqueens import build_dataset  # noqa: E402

DATA = build_dataset()
V = 3


def _solution_onehot(sol_tokens):
    return F.one_hot(torch.from_numpy(sol_tokens.astype(np.int64)), V).float()


def _pick_input(k):
    iid = int(np.flatnonzero(DATA.input_num_rows == k)[0])
    sols = DATA.solutions[DATA.input_solutions[iid]]  # (k, L) token ids
    return iid, sols


def test_oracle_covers_all_solutions():
    # Oracle trả lời giải thứ (j mod k) cho mẫu j -> phủ đủ -> coverage 1.0.
    iid, sols = _pick_input(3)
    sol_oh = _solution_onehot(sols)  # (3, L, V)

    def oracle(x_bar, s, state):
        B = x_bar.shape[0]
        idx = [j % sol_oh.shape[0] for j in range(B)]
        return sol_oh[idx], state

    out = ev.run_inferences(oracle, DATA, [iid], num_inferences=20, n_steps=16, gamma=5.0)
    assert abs(out["coverage_by_unique_input"] - 1.0) < 1e-9
    assert abs(out["coverage_by_row"] - 1.0) < 1e-9
    assert out["first_sample_accuracy"] == 1.0


def test_oracle_single_solution_partial_coverage():
    iid, sols = _pick_input(4)
    one = _solution_onehot(sols[:1])  # (1, L, V)

    def oracle(x_bar, s, state):
        B = x_bar.shape[0]
        return one.expand(B, -1, -1), state

    out = ev.run_inferences(oracle, DATA, [iid], num_inferences=20, n_steps=16, gamma=5.0)
    assert abs(out["coverage_by_unique_input"] - 0.25) < 1e-9  # 1/4
    assert out["first_sample_accuracy"] == 1.0


def test_oracle_invalid_gives_zero():
    iid, sols = _pick_input(3)
    # Oracle trả bàn rỗng (toàn empty) -> không hợp lệ -> coverage 0, accuracy 0.
    empty = torch.zeros(1, DATA.seq_len, V)
    empty[..., 1] = 1.0  # tất cả = empty (id 1)

    def oracle(x_bar, s, state):
        B = x_bar.shape[0]
        return empty.expand(B, -1, -1), state

    out = ev.run_inferences(oracle, DATA, [iid], num_inferences=20, n_steps=16, gamma=0.0)
    assert out["coverage_by_unique_input"] == 0.0
    assert out["first_sample_accuracy"] == 0.0


def test_model_eval_runs_end_to_end():
    # Mô hình nhỏ chưa huấn luyện: chỉ kiểm harness chạy và trả cấu trúc hợp lệ.
    from lf.train import Trainer, TrainConfig
    cfg = TrainConfig(hidden_size=32, num_heads=4, k_steps=4, batch_size=8,
                      warmup_steps=0, device="cpu")
    tr = Trainer(cfg, data=DATA, row_pool=DATA.train_rows[:8])
    out = ev.evaluate_model(tr, DATA.test_inputs[:3], num_inferences=4, n_steps=8,
                            gamma=5.0, use_ema=True)
    assert 0.0 <= out["coverage_by_unique_input"] <= 1.0
    assert 0.0 <= out["first_sample_accuracy"] <= 1.0
    assert out["weights"] == "ema" and out["num_inputs"] == 3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"\n{len(fns)} test đạt.")
