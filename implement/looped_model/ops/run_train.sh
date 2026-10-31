#!/usr/bin/env bash
# PLAN_TABLE2.md 3A.3 — chạy huấn luyện trong tmux, nice, log vào runs/.
# Gọi preflight trước; không chạy nếu preflight fail. Hỗ trợ --resume qua RESUME.
#
# Dùng (trên server, sau `source ops/env_server.sh`):
#   RUN=seed0 STEPS=20000 SEED=0 ops/run_train.sh
#   RESUME=runs/seed0/ckpt_8000.pt RUN=seed0 STEPS=20000 ops/run_train.sh
set -eu

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

RUN="${RUN:-dev}"
STEPS="${STEPS:?đặt STEPS=số bước}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
CKPT_EVERY="${CKPT_EVERY:-2000}"
PY="${PY:-.venv/bin/python}"
OUTDIR="runs/$RUN"
mkdir -p "$OUTDIR"

bash ops/preflight.sh || { echo "huỷ: preflight fail"; exit 1; }

resume_arg=""
[ -n "${RESUME:-}" ] && resume_arg="--resume $RESUME"

cmd="nice -n 10 $PY -m lf.train --config configs/nq8.yaml --steps $STEPS \
  --out $OUTDIR --device $DEVICE --seed $SEED --ckpt-every $CKPT_EVERY \
  $resume_arg 2>&1 | tee -a $OUTDIR/train.log"

session="lf_$RUN"
if command -v tmux >/dev/null; then
  echo "khởi chạy trong tmux session '$session'; xem: tmux attach -t $session"
  tmux new-session -d -s "$session" "$cmd"
else
  echo "không có tmux; chạy nền bằng nohup (log: $OUTDIR/train.log)"
  nohup bash -c "$cmd" >/dev/null 2>&1 &
  echo "PID $!"
fi
