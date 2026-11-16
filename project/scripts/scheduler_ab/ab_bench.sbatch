#!/bin/bash
#SBATCH -A m4999_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH -J tool-ab
#SBATCH -o /pscratch/sd/r/rmf253/moe-exp/ab/tool/jobs/%x-%j.out
# Hardened interleaved A/B of FlashMoE binaries on one GPU.
# BINS is a space-separated list of TAG=path. Every shape runs ROUNDS times;
# binary order rotates each round so node drift hits every binary.
#
# Lane-safe defaults write under ab/tool/. Override -J / -o for other lanes.
set -euo pipefail

AB_ROOT=${AB_ROOT:-/pscratch/sd/r/rmf253/moe-exp/ab/tool}
REPO_ENV=${REPO_ENV:-/pscratch/sd/r/rmf253/moe-exp/repo/project/scripts/env.sh}
# shellcheck disable=SC1090
source "$REPO_ENV"
export NVSHMEM_REMOTE_TRANSPORT=none
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/comm_libs/12.9/nccl-2.26/lib:${LD_LIBRARY_PATH:-}

ROUNDS=${ROUNDS:-3}
WARMUP=${WARMUP:-2}
RUNS=${RUNS:-16}
TIMEOUT_S=${TIMEOUT_S:-180}
ERROR_MAX=${ERROR_MAX:-0.25}   # fail shape if any timed run exceeds this error(%)
SMOKE=${SMOKE:-1}              # 1 = run mid shape once before the suite
CAND_TAG=${CAND_TAG:-}         # optional; used only in META for keep-gate hints
BASE_TAG=${BASE_TAG:-HYB17}

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

SMOKE_SHAPE=${SMOKE_SHAPE:-"4096 2048 2048 32 2"}
mkdir -p "${AB_ROOT}/jobs" "${AB_ROOT}/parsed"

nvidia-smi --query-gpu=name,memory.total,uuid --format=csv,noheader || true
echo "META host=$(hostname) job=${SLURM_JOB_ID:-local} date=$(date -Is)"
echo "META rounds=$ROUNDS warmup=$WARMUP runs=$RUNS timeout_s=$TIMEOUT_S error_max=$ERROR_MAX"
echo "META cand_tag=${CAND_TAG:-none} base_tag=$BASE_TAG ab_root=$AB_ROOT"
echo "META bins=${BINS}"
for entry in "${BIN_LIST[@]}"; do
  tag=${entry%%=*}
  bin=${entry#*=}
  if [[ ! -x "$bin" ]]; then
    echo "FAIL,missing_bin,$tag,$bin"
    exit 2
  fi
  sum=$(md5sum "$bin" | awk '{print $1}')
  echo "META bin_md5 tag=$tag path=$bin md5=$sum"
done

run_one() {
  local tag=$1 bin=$2 round=$3 shape=$4
  local out rc line
  set +e
  out=$(timeout "$TIMEOUT_S" "$bin" $shape 0 "$WARMUP" "$RUNS" 2>&1)
  rc=$?
  set -e
  line=$(grep -E '^0, ' <<< "$out" | head -1 || true)
  if [[ -z "$line" ]]; then
    echo "FAIL,$tag,$round,$shape,rc=$rc"
    # Keep a short diagnostic without dumping the whole NVSHMEM banner.
    grep -E 'error|Error|illegal|CUDA|FAIL|assert' <<< "$out" | head -5 || true
    return 1
  fi
  IFS=',' read -r -a f <<< "${line// /}"
  # f: rank,dtype,S,H,I,E,k,latency,error,...
  local err=${f[8]}
  echo "$tag,$round,${f[2]},${f[3]},${f[4]},${f[5]},${f[6]},${f[7]},${err}"
  # bc-less float compare via awk
  awk -v e="$err" -v m="$ERROR_MAX" 'BEGIN{exit !(e+0 > m+0)}' && {
    echo "FAIL,error_band,$tag,$round,$shape,error=$err,max=$ERROR_MAX"
    return 2
  }
  return 0
}

if [[ "$SMOKE" == "1" ]]; then
  echo "SMOKE_BEGIN $SMOKE_SHAPE"
  smoke_rc=0
  for entry in "${BIN_LIST[@]}"; do
    tag=${entry%%=*}
    bin=${entry#*=}
    run_one "$tag" "$bin" smoke "$SMOKE_SHAPE" || smoke_rc=$?
  done
  echo "SMOKE_END rc=$smoke_rc"
  if [[ "$smoke_rc" -ne 0 ]]; then
    echo "DONE_SMOKE_FAIL"
    exit 3
  fi
fi

echo "tag,round,S,H,I,E,k,latency_ms,error_pct"
n=${#BIN_LIST[@]}
fail_count=0
for shape in "${SHAPES[@]}"; do
  for ((r = 0; r < ROUNDS; ++r)); do
    for ((j = 0; j < n; ++j)); do
      entry=${BIN_LIST[$(( (j + r) % n ))]}
      tag=${entry%%=*}
      bin=${entry#*=}
      if ! run_one "$tag" "$bin" "$r" "$shape"; then
        fail_count=$((fail_count + 1))
      fi
    done
  done
done
echo "DONE fail_count=$fail_count"
[[ "$fail_count" -eq 0 ]]
