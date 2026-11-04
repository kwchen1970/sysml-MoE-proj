#!/bin/bash
# Shared launcher for one FlashMoE paper configuration.
# Expects to be sourced from an sbatch script after env.sh.
#
# Paper §4 shape that stays fixed inside a sweep:
#   H=2048, k=2, capacity factor 1.0 (the binary default EC=ceil(S/E)*k)
# Shipped kernel, not the paper's FP32 vanilla path:
#   BF16, gated MLP, eager timing (graph launches = 0)
# Timed iterations are capped at 4, with 1 warmup.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BIN="${REPO_ROOT}/csrc/cmake-build-release/testFlashMoE"
RESULT_DIR="${REPO_ROOT}/project/results"
RAW_DIR="${RESULT_DIR}/raw"
mkdir -p "$RAW_DIR"

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

paper_header() {
  local out=$1
  if [[ ! -s "$out" ]]; then
    echo "tag,ngpu,nodes,util_sm_mean,util_sm_max,rank,dtype,S,H,I,E,k,latency_ms,error_pct,EC,gpu,sm_arch,topology,mlp,bM,bN0,bK0,bN1,bK1,threads,blocks_per_sm,sms,blocks,rtol,atol,graph_launches,warmup,runs,workspace_mib" > "$out"
  fi
}

_dmon_summary() {
  local log=$1
  awk '
    $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ { sum += $2; n++; if ($2 > max) max = $2 }
    END {
      if (n == 0) print "nan,nan";
      else printf "%.2f,%.2f\n", sum / n, max
    }
  ' "$log"
}

# run_case TAG NGPU S H I E K [graph_launches] [warmup] [runs]
# Defaults match the paper protocol with the iteration cap: eager, 1 warmup, 4 timed.
run_case() {
  local tag=$1 ngpu=$2 S=$3 H=$4 I=$5 E=$6 k=$7
  local graph=${8:-0}
  local warmup=${9:-1}
  local nruns=${10:-4}
  local nodes gpn
  if (( ngpu <= 4 )); then
    nodes=1
    gpn=$ngpu
  else
    nodes=$(( ngpu / 4 ))
    gpn=4
  fi
  if (( SLURM_JOB_NUM_NODES < nodes )); then
    echo "allocation has ${SLURM_JOB_NUM_NODES} nodes, case ${tag} needs ${nodes}" >&2
    return 1
  fi
  if (( nodes > 1 )); then
    export NVSHMEM_REMOTE_TRANSPORT=libfabric
    export NVSHMEM_LIBFABRIC_PROVIDER=cxi
    export FI_CXI_DISABLE_HOST_REGISTER=1
  else
    export NVSHMEM_REMOTE_TRANSPORT=none
  fi

  local stamp out log dmon util_mean util_max line
  stamp=$(date +%Y%m%d_%H%M%S)
  out="${RESULT_DIR}/bench_${tag}.csv"
  log="${RAW_DIR}/${tag}_n${ngpu}_S${S}_E${E}_I${I}_${stamp}.log"
  dmon="${RAW_DIR}/${tag}_n${ngpu}_S${S}_E${E}_I${I}_${stamp}.dmon"
  paper_header "$out"

  # Every rank must see every local GPU. The upstream line
  # --gpus-per-task=1 --gpu-bind=closest hides NVLink peers, and this
  # NVSHMEM aborts while building its peer map. testFlashMoE itself is
  # unchanged: it still calls nvshmem_init before cudaSetDevice.
  # Multi-node also needs `module load libfabric/1.20.1` ahead of the
  # default CPE libfabric 2.3.1, which crashes inside nvshmem_init.
  local cvd
  cvd=$(seq -s, 0 $((gpn - 1)))
  echo "=== ${tag} ngpu=${ngpu} S=${S} H=${H} I=${I} E=${E} k=${k} transport=${NVSHMEM_REMOTE_TRANSPORT} cvd=${cvd} ==="

  srun --overlap --ntasks-per-node=1 -N "$nodes" -n "$nodes" \
    --gpus-per-node="$gpn" --gpu-bind=none -c 1 \
    env CUDA_VISIBLE_DEVICES="$cvd" nvidia-smi dmon -s u -d 1 -c 900 > "$dmon" 2>/dev/null &
  local dmon_pid=$!

  set +e
  timeout --signal=TERM 300 srun --overlap --mpi=cray_shasta \
    -N "$nodes" -n "$ngpu" --ntasks-per-node="$gpn" \
    --gpus-per-node="$gpn" --gpu-bind=none -c 8 \
    env CUDA_VISIBLE_DEVICES="$cvd" \
    "$BIN" "$S" "$H" "$I" "$E" "$k" "$graph" "$warmup" "$nruns" \
    > "$log" 2>&1
  local rc=$?
  set -e
  echo "srun rc=${rc} (log ${log})"

  sleep 1
  kill "$dmon_pid" 2>/dev/null || true
  wait "$dmon_pid" 2>/dev/null || true
  IFS=, read -r util_mean util_max < <(_dmon_summary "$dmon")

  local wrote=0
  while IFS= read -r line; do
    [[ "$line" =~ ^[0-9]+, ]] || continue
    # binary prints ", " between fields
    line=${line//, /,}
    echo "${tag},${ngpu},${nodes},${util_mean},${util_max},${line}" >> "$out"
    wrote=1
  done < "$log"

  if (( wrote == 0 )); then
    echo "${tag},${ngpu},${nodes},${util_mean},${util_max},FAIL,${S},${H},${I},${E},${k},rc=${rc},log=${log}" >> "${RESULT_DIR}/failures.csv"
    echo "case failed rc=${rc} log=${log}" >&2
    return 0
  fi
  echo "wrote ${out} (rc=${rc})"
}
