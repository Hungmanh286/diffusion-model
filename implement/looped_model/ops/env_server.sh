#!/usr/bin/env bash
# PLAN_TABLE2.md 3A.3 — đặt cache về /dataD TRƯỚC khi cài hoặc chạy bất cứ gì.
# Lý do: ổ / chỉ còn ~42 GB và 96% đầy; đầy ổ này có thể làm sập dịch vụ khác.
# Dùng: `source ops/env_server.sh` ở đầu mỗi phiên trên server.

export SRV_ROOT="${SRV_ROOT:-/dataD/AI_projects/hungmanh/diffusion-model/implement/looped_model}"
export LF_CACHE="$SRV_ROOT/cache"

mkdir -p "$LF_CACHE"/{hf,torch,xdg,tmp,uv,pip,triton,torch_ext}

export HF_HOME="$LF_CACHE/hf"
export TORCH_HOME="$LF_CACHE/torch"
export XDG_CACHE_HOME="$LF_CACHE/xdg"
export TMPDIR="$LF_CACHE/tmp"
export UV_CACHE_DIR="$LF_CACHE/uv"
export PIP_CACHE_DIR="$LF_CACHE/pip"
export TRITON_CACHE_DIR="$LF_CACHE/triton"
export TORCH_EXTENSIONS_DIR="$LF_CACHE/torch_ext"   # nơi dựng extension adam_atan2_backend

# Nhường CPU cho GitLab Runner và các API (3A.3, Q18).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

echo "SRV_ROOT=$SRV_ROOT"
echo "cache  =$LF_CACHE"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "df / :"; df -h / | tail -1
