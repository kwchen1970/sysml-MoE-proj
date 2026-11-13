#!/bin/bash
#SBATCH -A m4999_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH -J fm-ab
#SBATCH -o /pscratch/sd/r/rmf253/moe-exp/ab/ab-%j.out
# Interleaved A/B of FlashMoE binaries on one GPU, same node.
# BINS is a space-separated list of TAG=path. Every shape runs ROUNDS times;
# the binary order rotates each round so drift on the node hits every binary.
# Each run is eager with WARMUP warmup and RUNS timed iterations (paper used 1 and 4).
set -euo pipefail
source /pscratch/sd/r/rmf253/moe-exp/repo/project/scripts/env.sh
export NVSHMEM_REMOTE_TRANSPORT=none
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/comm_libs/12.9/nccl-2.26/lib:${LD_LIBRARY_PATH:-}

ROUNDS=${ROUNDS:-3}
WARMUP=${WARMUP:-2}
RUNS=${RUNS:-16}
read -r -a BIN_LIST <<< "${BINS:?set BINS=\"TAG=path ...\"}"

# Paper sweeps (H=2048, k=2) plus the project anchor (I=1408, E=16, k=6).
SHAPES=(
  "1024 2048 2048 32 2"
  "4096 2048 2048 32 2"
  "8192 2048 2048 32 2"
  "16384 2048 2048 32 2"
  "4096 2048 2048 8 2"
  "4096 2048 2048 16 2"
  "4096 2048 2048 128 2"
  "4096 2048 1024 32 2"
  "4096 2048 4096 32 2"
  "4096 2048 8192 32 2"
  "4096 2048 1408 16 6"
)
if [[ -n "${SHAPES_OVERRIDE:-}" ]]; then
  IFS=';' read -r -a SHAPES <<< "$SHAPES_OVERRIDE"
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "tag,round,S,H,I,E,k,latency_ms,error_pct"
n=${#BIN_LIST[@]}
for shape in "${SHAPES[@]}"; do
  for ((r = 0; r < ROUNDS; ++r)); do
    for ((j = 0; j < n; ++j)); do
      entry=${BIN_LIST[$(( (j + r) % n ))]}
      tag=${entry%%=*}
      bin=${entry#*=}
      set +e
      out=$(timeout 180 "$bin" $shape 0 "$WARMUP" "$RUNS" 2>&1)
      rc=$?
      set -e
      line=$(grep -E '^0, ' <<< "$out" | head -1 || true)
      if [[ -z "$line" ]]; then
        echo "FAIL,$tag,$r,$shape,rc=$rc"
        continue
      fi
      IFS=',' read -r -a f <<< "${line// /}"
      echo "$tag,$r,${f[2]},${f[3]},${f[4]},${f[5]},${f[6]},${f[7]},${f[8]}"
    done
  done
done
echo DONE
