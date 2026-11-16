#!/bin/bash
#SBATCH -A m4999_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH -J tool-noise
#SBATCH -o /pscratch/sd/r/rmf253/moe-exp/ab/tool/noise/%x-%j.out
# Noise floor study vs HYB17.
# Phase identity: same HYB17 binary tagged twice → measurement noise / geo≈1.
# Phase margin: BK64 + HYB2 + HYB17 on the standard shape suite.
set -euo pipefail

AB_ROOT=${AB_ROOT:-/pscratch/sd/r/rmf253/moe-exp/ab/tool}
BIN_DIR=${BIN_DIR:-/pscratch/sd/r/rmf253/moe-exp/ab}
REPO_ENV=${REPO_ENV:-/pscratch/sd/r/rmf253/moe-exp/repo/project/scripts/env.sh}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARNESS="${SCRIPT_DIR}/ab_bench.sbatch"

HYB17=${HYB17:-"${BIN_DIR}/testFlashMoE.hyb17"}
HYB2=${HYB2:-"${BIN_DIR}/testFlashMoE.hyb2"}
BK64=${BK64:-"${BIN_DIR}/testFlashMoE.bk64"}

for bin in "$HYB17" "$HYB2" "$BK64"; do
  [[ -x "$bin" ]] || { echo "missing $bin"; exit 2; }
done

mkdir -p "${AB_ROOT}/noise" "${AB_ROOT}/jobs"
# shellcheck disable=SC1090
source "$REPO_ENV"
export NVSHMEM_REMOTE_TRANSPORT=none
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/comm_libs/12.9/nccl-2.26/lib:${LD_LIBRARY_PATH:-}
export AB_ROOT ERROR_MAX=${ERROR_MAX:-0.25} TIMEOUT_S=${TIMEOUT_S:-180}

run_harness() {
  # Strip SBATCH directives so we can chain phases in one allocation.
  bash <(grep -v '^#SBATCH' "$HARNESS")
}

echo "META noise_study host=$(hostname) job=${SLURM_JOB_ID:-local} date=$(date -Is)"

# --- identity: high-round, compact shape set ---
export ROUNDS=${ID_ROUNDS:-12}
export WARMUP=${WARMUP:-2}
export RUNS=${RUNS:-16}
export SMOKE=1
export BASE_TAG=HYB17A
export CAND_TAG=HYB17B
export BINS="HYB17A=${HYB17} HYB17B=${HYB17}"
export SHAPES_OVERRIDE=$'4096 2048 2048 32 2\n4096 2048 1408 16 6\n16384 2048 2048 32 2\n4096 2048 8192 32 2'
echo "==== PHASE identity HYB17A vs HYB17B rounds=$ROUNDS ===="
run_harness

# --- margin: standard suite, fewer rounds ---
export ROUNDS=${MARGIN_ROUNDS:-5}
export SMOKE=1
export BASE_TAG=HYB17
export CAND_TAG=HYB17
export BINS="BK64=${BK64} HYB2=${HYB2} HYB17=${HYB17}"
unset SHAPES_OVERRIDE
echo "==== PHASE margin BK64/HYB2/HYB17 rounds=$ROUNDS ===="
run_harness

echo "NOISE_STUDY_DONE"
