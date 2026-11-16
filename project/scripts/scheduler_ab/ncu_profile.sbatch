#!/bin/bash
#SBATCH -A m4999_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH -J tool-ncu
#SBATCH -o /pscratch/sd/r/rmf253/moe-exp/ab/tool/ncu/%x-%j.out
# Profile one or more TAG=path binaries on a short shape list with ncu_wrap.sh.
set -euo pipefail

AB_ROOT=${AB_ROOT:-/pscratch/sd/r/rmf253/moe-exp/ab/tool}
REPO_ENV=${REPO_ENV:-/pscratch/sd/r/rmf253/moe-exp/repo/project/scripts/env.sh}
# shellcheck disable=SC1090
source "$REPO_ENV"
export NVSHMEM_REMOTE_TRANSPORT=none
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/comm_libs/12.9/nccl-2.26/lib:${LD_LIBRARY_PATH:-}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WRAP=${WRAP:-"${SCRIPT_DIR}/ncu_wrap.sh"}
SET=${SET:-speedoflight}
read -r -a BIN_LIST <<< "${BINS:?set BINS=\"TAG=path ...\"}"

SHAPES=(
  "4096 2048 1408 16 6"
  "4096 2048 2048 32 2"
)
if [[ -n "${SHAPES_OVERRIDE:-}" ]]; then
  IFS=';' read -r -a SHAPES <<< "$SHAPES_OVERRIDE"
fi

mkdir -p "${AB_ROOT}/ncu"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "META host=$(hostname) job=${SLURM_JOB_ID:-local} set=$SET bins=$BINS"

for entry in "${BIN_LIST[@]}"; do
  tag=${entry%%=*}
  bin=${entry#*=}
  for shape in "${SHAPES[@]}"; do
    # shellcheck disable=SC2086
    LABEL="$tag" SET="$SET" AB_ROOT="$AB_ROOT" "$WRAP" "$bin" $shape
  done
done
echo DONE
