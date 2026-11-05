#!/bin/bash
#SBATCH --job-name=moe-granularity
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 01:00:00
#SBATCH --output=granularity-%j.log
set -eo pipefail
# Submit from repository root; select your allocation with sbatch -A ACCOUNT.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
source project/scripts/expert_granularity/env.sh
export PATH="/global/homes/k/kc994/.conda/envs/sysml/bin:$PATH"

# run.py uses the shared NVSHMEM-compatible single-node launcher.
python3 project/scripts/expert_granularity/run.py "${@}"
