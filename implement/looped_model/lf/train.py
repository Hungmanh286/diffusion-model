"""M5 — Huấn luyện Looped Flows theo Algorithm 1 (PLAN M5, SPEC §3.2).

Một bước gradient cho mỗi bước denoising (Q3). Gộp batch kiểu TRM (D11): mỗi vị
trí batch mang một rollout riêng (trạng thái z, chỉ số bước, lịch t, nhiễu x0,
nhãn x1, mặt nạ clue); vị trí dừng được thay bằng dòng dữ liệu mới. Khớp vòng
ngoài TRM (ref/models/recursive_reasoning/trm.py:249-297), gồm exploration lấy
mẫu lại mỗi bước (trm.py:286).

Đầu vào denoiser là interpolant teacher-forced I_{t_i} (không phải backtrack của
sampler) đúng theo Eq. 9. Hỗ trợ checkpoint và --resume (PLAN 3A: job dài trên
máy dùng chung).

Chạy nhanh trên CPU với mô hình nhỏ để kiểm (test_train.py). Huấn luyện thật cỡ
7M cần GPU server.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lf import flow
from lf.data_nqueens import build_dataset
from lf.model import LFDenoiser
from lf.optim import AdamATan2, EMA, build_lr_lambda

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class TrainConfig:
    vocab_size: int = 3
    seq_len: int = 64
    hidden_size: int = 512
    num_layers: int = 2
    num_heads: int = 8
    m_cycles: int = 4
    outer_cycles: int = 3
    k_steps: int = 16
    sigma: float = 1.0
    time_sampler: str = "sorted"
    exploration_prob: float = 0.1
    act_lambda: float = 0.5
    peak_lr: float = 1e-4
    warmup_steps: int = 2000
    weight_decay: float = 1.0
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    batch_size: int = 768
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        if yaml is None:
            raise RuntimeError("cần pyyaml để đọc config")
        with open(path) as f:
            c = yaml.safe_load(f)
        return cls(
            vocab_size=c["data"]["vocab_size"],
            seq_len=c["data"]["seq_len"],
            hidden_size=c["model"]["hidden_size"],
            num_layers=c["model"]["num_layers"],
            num_heads=c["model"]["num_heads"],
            m_cycles=c["model"]["m_cycles"],
            outer_cycles=c["model"]["outer_cycles"],
            k_steps=c["flow"]["k_steps"],
            sigma=c["flow"]["sigma"],
            time_sampler=c["flow"]["time_sampler"],
            exploration_prob=c["loss"]["exploration_prob"],
            act_lambda=c["loss"]["act_lambda"],
            peak_lr=c["optim"]["peak_lr"],
            warmup_steps=c["optim"]["warmup_steps"],
            weight_decay=c["optim"]["weight_decay"],
            grad_clip=c["optim"]["grad_clip"],
            ema_decay=c["optim"]["ema_decay"],
            batch_size=c["optim"]["batch_size"],
        )


class Trainer:
    def __init__(self, cfg: TrainConfig, data=None, row_pool=None):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)
        torch.manual_seed(cfg.seed)
        self.gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

        self.data = data if data is not None else build_dataset()
        pool = row_pool if row_pool is not None else self.data.train_rows
        self.row_pool = np.asarray(pool, dtype=np.int64)

        # Tensor dùng chung, đưa lên thiết bị một lần (dữ liệu nhỏ, PLAN 3A).
        V = cfg.vocab_size
        x1_ids = torch.from_numpy(self.data.x1.astype(np.int64))      # (R, L)
        self.x1_onehot = F.one_hot(x1_ids, V).to(self.dtype).to(self.device)
        self.labels = x1_ids.to(self.device)
        self.clue_mask_all = torch.from_numpy(self.data.clue_mask).to(self.device)

        self.model = LFDenoiser(
            vocab_size=V, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
            num_heads=cfg.num_heads, m_cycles=cfg.m_cycles, outer_cycles=cfg.outer_cycles,
            seq_len=cfg.seq_len, dtype=self.dtype,
        ).to(self.device)
        self.opt = AdamATan2(
            self.model.parameters(), lr=cfg.peak_lr, betas=(0.9, 0.95),
            weight_decay=cfg.weight_decay,
        )
        self.lr_lambda = build_lr_lambda(cfg.warmup_steps, "constant")
        self.ema = EMA(self.model, decay=cfg.ema_decay)
        self.time_sampler = flow.make_time_sampler(cfg.time_sampler)
        self.step = 0
        self._init_carry()

    # -- carry ------------------------------------------------------------- #
    def _init_carry(self):
        B = self.cfg.batch_size
        h, l = self.model.initial_state(B, device=self.device)
        self.c_h, self.c_l = h.clone(), l.clone()
        self.c_rows = torch.zeros(B, dtype=torch.long, device=self.device)
        self.c_step = torch.zeros(B, dtype=torch.long, device=self.device)
        self.c_x0 = torch.zeros(B, self.cfg.seq_len, self.cfg.vocab_size, dtype=self.dtype, device=self.device)
        self.c_tsched = torch.zeros(B, self.cfg.k_steps, dtype=self.dtype, device=self.device)
        self.c_halted = torch.ones(B, dtype=torch.bool, device=self.device)  # ép reset toàn bộ ở bước đầu

    def _draw_rows(self, num):
        idx = torch.from_numpy(
            self.row_pool[torch.randint(len(self.row_pool), (num,), generator=self.gen).numpy()]
        ).to(self.device)
        return idx

    def _reset_halted(self):
        halted = self.c_halted
        n = int(halted.sum())
        if n == 0:
            return
        new_rows = self._draw_rows(n)
        self.c_rows[halted] = new_rows
        self.c_step[halted] = 0
        # Nhiễu mới và lịch thời gian mới cho vị trí được reset.
        self.c_x0[halted] = self.cfg.sigma * torch.randn(
            n, self.cfg.seq_len, self.cfg.vocab_size, generator=self.gen, dtype=self.dtype
        ).to(self.device)
        tsched = self.time_sampler(self.cfg.k_steps - 1, n, generator=self.gen).to(self.dtype).to(self.device)
        self.c_tsched[halted] = tsched  # (n, k) đã sắp tăng
        h0, l0 = self.model.initial_state(n, device=self.device)
        self.c_h[halted] = h0
        self.c_l[halted] = l0

    # -- một bước huấn luyện ---------------------------------------------- #
    def train_step(self) -> dict:
        cfg = self.cfg
        self.model.train()
        self._reset_halted()

        rows = self.c_rows
        x1 = self.x1_onehot[rows]
        labels = self.labels[rows]
        clue = self.clue_mask_all[rows]
        t_i = torch.gather(self.c_tsched, 1, self.c_step.clamp_max(cfg.k_steps - 1).unsqueeze(1)).squeeze(1)

        I_t = flow.interpolant(self.c_x0, x1, t_i, clue_mask=clue)
        logits, q_halt, new_state = self.model(I_t, t_i, (self.c_h, self.c_l))

        valid = flow.loss_valid_mask(labels, clue)
        lm_loss = flow.mean_ce_per_sequence(logits, labels, valid).mean()
        with torch.no_grad():
            q_label = flow.act_label(logits, labels, valid).to(torch.float32)
        q_loss = F.binary_cross_entropy_with_logits(q_halt, q_label, reduction="mean")
        loss = lm_loss + cfg.act_lambda * q_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        lr = cfg.peak_lr * self.lr_lambda(self.step)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.step()
        self.ema.update(self.model)

        # Cập nhật carry: trạng thái mới, tăng bước, tính halt (kiểu TRM).
        with torch.no_grad():
            self.c_h, self.c_l = new_state
            self.c_step = self.c_step + 1
            is_last = self.c_step >= cfg.k_steps
            halted = is_last | (q_halt > 0)
            # Exploration: lấy mẫu lại mỗi bước (trm.py:286).
            forced = (torch.rand(cfg.batch_size, generator=self.gen).to(self.device) < cfg.exploration_prob)
            m_min = torch.randint(2, cfg.k_steps + 1, (cfg.batch_size,), generator=self.gen).to(self.device)
            min_halt = forced * m_min
            halted = halted & (self.c_step >= min_halt)
            self.c_halted = halted

        self.step += 1
        return {
            "step": self.step, "loss": loss.item(), "lm_loss": lm_loss.item(),
            "q_loss": q_loss.item(), "lr": lr,
            "frac_halted": float(halted.float().mean()),
        }

    # -- đánh giá 1 bước (kiểm overfit) ----------------------------------- #
    @torch.no_grad()
    def one_step_accuracy(self, rows: np.ndarray, t_value: float = 0.9, use_ema: bool = False) -> float:
        """Độ chính xác dự đoán 1 bước ở thời gian t lớn (không chạy sampler đầy đủ)."""
        model = self.model
        if use_ema:
            backup = {n: p.detach().clone() for n, p in model.named_parameters()}
            self.ema.copy_to(model)
        model.eval()
        rows_t = torch.from_numpy(np.asarray(rows, dtype=np.int64)).to(self.device)
        x1 = self.x1_onehot[rows_t]
        labels = self.labels[rows_t]
        clue = self.clue_mask_all[rows_t]
        B = rows_t.shape[0]
        x0 = self.cfg.sigma * torch.randn(B, self.cfg.seq_len, self.cfg.vocab_size, generator=self.gen, dtype=self.dtype).to(self.device)
        t = torch.full((B,), t_value, dtype=self.dtype, device=self.device)
        I_t = flow.interpolant(x0, x1, t, clue_mask=clue)
        h, l = model.initial_state(B, device=self.device)
        logits, _, _ = model(I_t, t, (h, l))
        valid = flow.loss_valid_mask(labels, clue)
        preds = logits.argmax(-1)
        seq_ok = ((valid & (preds == labels)).sum(-1) == valid.sum(-1))
        if use_ema:
            for n, p in model.named_parameters():
                p.data.copy_(backup[n])
        return float(seq_ok.float().mean())

    # -- checkpoint -------------------------------------------------------- #
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "step": self.step,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "gen_rng": self.gen.get_state(),
            "cfg": self.cfg.__dict__,
            # Carry rollout đang dở của từng vị trí batch (D11) — cần cho resume
            # liền mạch; thiếu nó thì mọi rollout bị khởi động lại khi --resume.
            "carry": {
                "h": self.c_h, "l": self.c_l, "rows": self.c_rows,
                "step": self.c_step, "x0": self.c_x0, "tsched": self.c_tsched,
                "halted": self.c_halted,
            },
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.step = ck["step"]
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.ema.load_state_dict(ck["ema"])
        torch.set_rng_state(ck["torch_rng"])
        self.gen.set_state(ck["gen_rng"])
        cy = ck["carry"]
        self.c_h, self.c_l = cy["h"].to(self.device), cy["l"].to(self.device)
        self.c_rows = cy["rows"].to(self.device)
        self.c_step = cy["step"].to(self.device)
        self.c_x0 = cy["x0"].to(self.device)
        self.c_tsched = cy["tsched"].to(self.device)
        self.c_halted = cy["halted"].to(self.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nq8.yaml")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--out", default="runs/dev")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    cfg.device = args.device
    cfg.seed = args.seed
    trainer = Trainer(cfg)
    if args.resume:
        trainer.load(args.resume)
        print(f"resume từ bước {trainer.step}")

    t0 = time.time()
    while trainer.step < args.steps:
        m = trainer.train_step()
        if m["step"] % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {m['step']} loss {m['loss']:.4f} lm {m['lm_loss']:.4f} "
                  f"q {m['q_loss']:.4f} lr {m['lr']:.2e} halt {m['frac_halted']:.2f} "
                  f"{m['step']/max(dt,1e-9):.2f} it/s")
        if m["step"] % args.ckpt_every == 0:
            trainer.save(os.path.join(args.out, f"ckpt_{m['step']}.pt"))
    trainer.save(os.path.join(args.out, "ckpt_final.pt"))


if __name__ == "__main__":
    main()
