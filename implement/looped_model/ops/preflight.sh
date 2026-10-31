#!/usr/bin/env bash
# PLAN_TABLE2.md 3A.4 — kiểm an toàn TRƯỚC khi chạy job trên server production.
# Thoát khác 0 nếu vi phạm; run_train.sh gọi nó và sẽ không chạy nếu fail.
# Quy tắc 1: VRAM trống >= trần train (12 GB) + 2 GB dự phòng = 14 GB.
set -u

NEED_FREE_MB="${LF_NEED_FREE_MB:-14000}"   # 12 GB trần + 2 GB dự phòng (3A.3, 3A.4)
fail() { echo "PREFLIGHT FAIL: $1" >&2; exit 1; }

command -v nvidia-smi >/dev/null || fail "không có nvidia-smi (đang chạy trên server?)"

free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ -n "$free_mb" ] || fail "không đọc được VRAM trống"
echo "VRAM trống: ${free_mb} MiB (cần >= ${NEED_FREE_MB})"
[ "$free_mb" -ge "$NEED_FREE_MB" ] || fail "VRAM trống không đủ; có thể dịch vụ khác đang dùng GPU"

# Cảnh báo nếu ổ / gần đầy (cache phải ở /dataD; xem env_server.sh).
root_avail_gb=$(df -BG / | tail -1 | awk '{gsub("G","",$4); print $4}')
echo "ổ / còn: ${root_avail_gb} GB"
[ "${root_avail_gb:-0}" -ge 5 ] || fail "ổ / còn dưới 5 GB; kiểm cache có trỏ về /dataD chưa"

# Không khởi chạy nếu đã có job train của ta đang chạy (tránh tranh VRAM).
if pgrep -f "lf.train" >/dev/null 2>&1; then
  fail "đã có tiến trình lf.train đang chạy"
fi

echo "PREFLIGHT OK"
