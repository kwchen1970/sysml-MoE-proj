#!/bin/bash
#SBATCH -A m4999_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH -c 16
#SBATCH -J epi0-fix
#SBATCH -o /pscratch/sd/r/rmf253/moe-exp/ab/epi/epi0-%j.out
set -euo pipefail
BASE=/pscratch/sd/r/rmf253/moe-exp
EPI=$BASE/ab/epi
REPO=$EPI/repo
BUILD=$EPI/build
mkdir -p "$BUILD"
source "$BASE/repo/project/scripts/env.sh"
# env.sh points at jbr266 mathdx (permission denied); use our pscratch copy.
export MATHDX_ROOT=/pscratch/sd/r/rmf253/mathdx/nvidia-mathdx-25.12.0-cuda12/nvidia/mathdx/25.12
export mathdx_DIR=$MATHDX_ROOT/lib/cmake/mathdx
export CMAKE_PREFIX_PATH=$MATHDX_ROOT:${CMAKE_PREFIX_PATH:-}
export NVSHMEM_REMOTE_TRANSPORT=none
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/comm_libs/12.9/nccl-2.26/lib:${LD_LIBRARY_PATH:-}
echo "=== build EPI0 (gmem staging fix + cosize, no gateCache) ==="
echo "MATHDX_ROOT=$MATHDX_ROOT"
rm -rf "$BUILD"
cmake -DCMAKE_BUILD_TYPE=Release -Wno-dev -G Ninja \
  -Dmathdx_DIR="$mathdx_DIR" \
  -S "$REPO/csrc" -B "$BUILD"
cmake --build "$BUILD" --target testFlashMoE --parallel 8
cp -f "$BUILD/testFlashMoE" "$EPI/testFlashMoE.epi0"
set +e
"$EPI/testFlashMoE.epi0" 4096 2048 1408 16 6 0 1 2; echo SMOKE_EPI0_ANCHOR_RC=$?
"$EPI/testFlashMoE.epi0" 4096 2048 2048 32 2 0 1 2; echo SMOKE_EPI0_MID_RC=$?
"$BASE/ab/testFlashMoE.hyb17" 4096 2048 2048 32 2 0 1 2; echo SMOKE_HYB17_MID_RC=$?
set -e
ROUNDS=${ROUNDS:-5}
WARMUP=2
RUNS=16
BIN_LIST=( "BK64=$BASE/ab/testFlashMoE.bk64" "HYB2=$BASE/ab/testFlashMoE.hyb2" "HYB17=$BASE/ab/testFlashMoE.hyb17" "EPI0=$EPI/testFlashMoE.epi0" )
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "tag,round,S,H,I,E,k,latency_ms,error_pct"
n=${#BIN_LIST[@]}
for shape in "${SHAPES[@]}"; do
  for ((r=0;r<ROUNDS;++r)); do
    for ((j=0;j<n;++j)); do
      entry=${BIN_LIST[$(( (j+r)%n ))]}
      tag=${entry%%=*}; bin=${entry#*=}
      set +e
      out=$(timeout 180 "$bin" $shape 0 "$WARMUP" "$RUNS" 2>&1); rc=$?
      set -e
      line=$(grep -E "^0, " <<< "$out" | head -1 || true)
      if [[ -z "$line" ]]; then echo "FAIL,$tag,$r,$shape,rc=$rc"; echo "$out" | tail -12; continue; fi
      IFS="," read -r -a f <<< "${line// /}"
      echo "$tag,$r,${f[2]},${f[3]},${f[4]},${f[5]},${f[6]},${f[7]},${f[8]}"
    done
  done
done
echo DONE
