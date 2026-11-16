#!/bin/bash
#SBATCH -A m4999_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH -J tool-idnoise
#SBATCH -o /pscratch/sd/r/rmf253/moe-exp/ab/tool/noise/%x-%j.out
# Compact HYB17 identity noise on debug QOS (faster than full regular study).
set -euo pipefail

AB_ROOT=${AB_ROOT:-/pscratch/sd/r/rmf253/moe-exp/ab/tool}
BIN_DIR=${BIN_DIR:-/pscratch/sd/r/rmf253/moe-exp/ab}
REPO_ENV=${REPO_ENV:-/pscratch/sd/r/rmf253/moe-exp/repo/project/scripts/env.sh}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HYB17=${HYB17:-"${BIN_DIR}/testFlashMoE.hyb17"}

[[ -x "$HYB17" ]] || { echo "missing $HYB17"; exit 2; }
mkdir -p "${AB_ROOT}/noise"
# shellcheck disable=SC1090
source "$REPO_ENV"
export NVSHMEM_REMOTE_TRANSPORT=none
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/comm_libs/12.9/nccl-2.26/lib:${LD_LIBRARY_PATH:-}

export AB_ROOT
export ROUNDS=${ID_ROUNDS:-10}
export WARMUP=${WARMUP:-2}
export RUNS=${RUNS:-16}
export ERROR_MAX=${ERROR_MAX:-0.25}
export SMOKE=1
export TIMEOUT_S=${TIMEOUT_S:-180}
export BASE_TAG=HYB17A
export CAND_TAG=HYB17B
export BINS="HYB17A=${HYB17} HYB17B=${HYB17}"
export SHAPES_OVERRIDE=$'4096 2048 2048 32 2\n4096 2048 1408 16 6\n16384 2048 2048 32 2\n4096 2048 8192 32 2'

echo "META compact_identity host=$(hostname) job=${SLURM_JOB_ID:-local} rounds=$ROUNDS"
bash <(grep -v '^#SBATCH' "${SCRIPT_DIR}/ab_bench.sbatch")
echo "IDENTITY_NOISE_DONE"
