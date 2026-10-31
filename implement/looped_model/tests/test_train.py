"""Test M5 — vòng lặp huấn luyện trên CPU (PLAN M5, thí nghiệm 1 + resume).

Mô hình nhỏ, overfit 32 dòng để kiểm wiring: loss giảm mạnh, độ chính xác 1 bước
ở t lớn đạt cao. Và checkpoint/resume tái lập đúng (PLAN 3A). Huấn luyện thật cỡ
7M là việc của GPU server (M5 thí nghiệm đầy đủ, M6).

Chậm hơn các test khác (~1 phút trên CPU). Chạy: `python3 tests/test_train.py`.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lf.data_nqueens import build_dataset  # noqa: E402
from lf.train import Trainer, TrainConfig  # noqa: E402

DATA = build_dataset()


def _cfg(**kw):
    base = dict(
        hidden_size=64, num_heads=4, k_steps=8, batch_size=32,
        warmup_steps=0, peak_lr=1e-3, weight_decay=0.0, seed=0, device="cpu",
    )
    base.update(kw)
    return TrainConfig(**base)


def test_overfit_one_batch():
    pool = DATA.train_rows[:32]
    tr = Trainer(_cfg(), data=DATA, row_pool=pool)
    acc0 = tr.one_step_accuracy(pool, t_value=0.9)
    losses = [tr.train_step()["loss"] for _ in range(250)]
    acc1 = tr.one_step_accuracy(pool, t_value=0.9)
    assert np.mean(losses[-20:]) < 0.6 * losses[0], (losses[0], np.mean(losses[-20:]))
    assert acc0 <= 0.2  # trước huấn luyện gần như không đúng
    assert acc1 >= 0.8, acc1  # sau overfit, dự đoán 1 bước ở t lớn gần 100%


def test_resume_reproducible(tmp_path=None):
    import tempfile
    d = tempfile.mkdtemp()
    pool = DATA.train_rows[:32]

    # Chạy liền 40 bước.
    a = Trainer(_cfg(), data=DATA, row_pool=pool)
    for _ in range(40):
        a.train_step()
    ref = {n: p.detach().clone() for n, p in a.model.named_parameters()}

    # Chạy 20 bước, lưu, nạp vào trainer mới, chạy tiếp 20 bước.
    b = Trainer(_cfg(), data=DATA, row_pool=pool)
    for _ in range(20):
        b.train_step()
    ckpt = os.path.join(d, "c.pt")
    b.save(ckpt)
    c = Trainer(_cfg(), data=DATA, row_pool=pool)
    c.load(ckpt)
    assert c.step == 20
    for _ in range(20):
        c.train_step()

    for n, p in c.model.named_parameters():
        assert torch.allclose(p, ref[n], atol=1e-5), f"resume lệch ở {n}"


if __name__ == "__main__":
    for fn in [test_overfit_one_batch, test_resume_reproducible]:
        fn()
        print("ok  ", fn.__name__)
    print("\n2 test đạt.")
