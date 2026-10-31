"""M7 — Đánh giá Table 2 cho N-Queens (PLAN M7, SPEC §4.3, §5).

Mỗi input test: 20 lần suy luận độc lập (nhiễu x0 riêng) qua sampler Algorithm 2;
coverage = |lời giải hợp lệ khác nhau| / |S(c)| (cả hai cách gộp, D2); accuracy
= tỉ lệ input mà mẫu đầu hợp lệ (§5.2). Báo theo input duy nhất làm số tiêu đề,
kèm theo dòng (Q13). Hỗ trợ trọng số EMA và thô (D7), gamma 5 (SDE) và 0 (ODE).

Lõi `run_inferences` nhận `denoise_fn` nên test được bằng oracle không cần GPU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from lf import metrics as mt
from lf import sampler as sp


def model_denoise_fn(model, device, dtype):
    """Bọc LFDenoiser thành denoise_fn(x_bar, s, state) -> (x_hat, state).

    x_hat = softmax(logits): dự đoán x1 như vector xác suất trên simplex (SPEC 2.1).
    """
    def fn(x_bar, s, state):
        if state is None:
            state = model.initial_state(x_bar.shape[0], device=device)
        t = torch.as_tensor(s, device=device, dtype=dtype)
        if t.ndim == 0:
            t = t.expand(x_bar.shape[0])
        logits, _, new_state = model(x_bar.to(dtype), t, state)
        return F.softmax(logits.float(), dim=-1).to(dtype), new_state
    return fn


@torch.no_grad()
def run_inferences(
    denoise_fn,
    data,
    input_ids,
    num_inferences: int = 20,
    n_steps: int = 128,
    gamma: float = 5.0,
    sigma: float = 1.0,
    device="cpu",
    dtype=torch.float32,
    generator=None,
):
    """Chạy đánh giá trên danh sách input. Trả về dict coverage + accuracy.

    data: NQueensData; input_ids: iterable id input (thường data.test_inputs).
    """
    device = torch.device(device)
    L = data.seq_len
    per_input_cov: dict[int, float] = {}
    first_valid: list[bool] = []
    clues_for_acc: list[np.ndarray] = []

    for iid in input_ids:
        iid = int(iid)
        clue_cells = data.input_clue_cells[iid]
        num_compat = len(data.input_solutions[iid])
        # x1_clue: one-hot của giá trị clue (quân=2) tại ô clue; chỗ khác không dùng.
        x1_clue = torch.zeros(num_inferences, L, 3, dtype=dtype, device=device)
        x1_clue[:, clue_cells, mt.QUEEN_ID] = 1.0
        clue_mask = torch.zeros(num_inferences, L, dtype=torch.bool, device=device)
        clue_mask[:, clue_cells] = True

        x0 = sigma * torch.randn(num_inferences, L, 3, generator=generator, dtype=dtype).to(device)
        x_final = sp.sample(
            denoise_fn, x0, n_steps=n_steps, gamma=gamma, sigma=sigma, state0=None,
            clue_mask=clue_mask, x1_clue=x1_clue, generator=generator,
        )
        tokens = sp.round_to_tokens(x_final).cpu().numpy()  # (num_inferences, L)

        cov = mt.coverage_for_input(tokens, clue_cells, num_compat, n=data.n)
        per_input_cov[iid] = cov
        first_valid.append(mt.is_valid_board(tokens[0], clue_cells, n=data.n))
        clues_for_acc.append(clue_cells)

    # Gộp coverage theo dòng: lặp theo số dòng test của mỗi input.
    row_input_ids = []
    for iid in input_ids:
        row_input_ids += [int(iid)] * int(data.input_num_rows[int(iid)])
    agg = mt.aggregate_coverage(per_input_cov, np.array(row_input_ids)) if row_input_ids else {
        "by_unique_input": float("nan"), "by_row": float("nan")}

    return {
        "coverage_by_unique_input": agg["by_unique_input"],
        "coverage_by_row": agg["by_row"],
        "first_sample_accuracy": float(np.mean(first_valid)) if first_valid else float("nan"),
        "num_inputs": len(per_input_cov),
    }


@torch.no_grad()
def evaluate_model(trainer, input_ids, num_inferences=20, n_steps=128, gamma=5.0,
                   use_ema=True, generator=None):
    """Đánh giá mô hình của một Trainer. Tạm nạp EMA nếu use_ema rồi khôi phục."""
    model = trainer.model
    device = trainer.device
    dtype = trainer.dtype
    backup = None
    if use_ema:
        backup = {n: p.detach().clone() for n, p in model.named_parameters()}
        trainer.ema.copy_to(model)
    model.eval()
    try:
        out = run_inferences(
            model_denoise_fn(model, device, dtype), trainer.data, input_ids,
            num_inferences=num_inferences, n_steps=n_steps, gamma=gamma,
            sigma=trainer.cfg.sigma, device=device, dtype=dtype, generator=generator,
        )
    finally:
        if backup is not None:
            for n, p in model.named_parameters():
                p.data.copy_(backup[n])
    out["weights"] = "ema" if use_ema else "raw"
    out["gamma"] = gamma
    return out
