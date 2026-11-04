#!/bin/bash
#SBATCH --job-name=moe-device-trace
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 01:00:00
#SBATCH --output=granularity-trace-%j.log
set -eo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
source project/scripts/expert_granularity/env.sh
export PATH="/global/homes/k/kc994/.conda/envs/sysml/bin:$PATH"
python3 project/scripts/expert_granularity/run.py \
  --binary "${BIN:-csrc/cmake-build-trace/testFlashMoE}" --device-trace "$@"
