#!/bin/bash
# Thin Nsight Compute wrapper for a single FlashMoE shape.
# Writes under AB_ROOT/ncu/ by default so lanes do not clobber each other.
#
# Usage:
#   ./ncu_wrap.sh /path/to/testFlashMoE.hyb17 4096 2048 1408 16 6
#   LABEL=hyb17-anchor SET=speedoflight ./ncu_wrap.sh ...
set -euo pipefail

AB_ROOT=${AB_ROOT:-/pscratch/sd/r/rmf253/moe-exp/ab/tool}
NCU_BIN=${NCU_BIN:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Compute/ncu}
SET=${SET:-full}
LABEL=${LABEL:-run}
WARMUP=${WARMUP:-1}
RUNS=${RUNS:-1}
KERNEL_REGEX=${KERNEL_REGEX:-regex:flashmoe|FlashMoE|moe}

if [[ $# -lt 6 ]]; then
  echo "usage: $0 BIN S H I E k" >&2
  exit 1
fi

BIN=$1; S=$2; H=$3; I=$4; E=$5; k=$6
shift 6 || true

if [[ ! -x "$BIN" ]]; then
  echo "missing binary: $BIN" >&2
  exit 2
fi
if [[ ! -x "$NCU_BIN" ]]; then
  # Fall back to PATH / older HPC SDK layouts.
  if command -v ncu >/dev/null 2>&1; then
    NCU_BIN=$(command -v ncu)
  else
    echo "ncu not found; set NCU_BIN" >&2
    exit 3
  fi
fi

mkdir -p "${AB_ROOT}/ncu"
stamp=$(date +%Y%m%d-%H%M%S)
out_base="${AB_ROOT}/ncu/${LABEL}-S${S}-I${I}-E${E}-k${k}-${stamp}"

export NVSHMEM_REMOTE_TRANSPORT=none
export NVSHMEM_DISABLE_GDRCOPY=1
export NVSHMEM_DISABLE_NCCL=1

case "$SET" in
  speedoflight|sol)
    section_args=(--set speedoflightdefault)
    ;;
  full)
    section_args=(--set full)
    ;;
  mem)
    section_args=(--section MemoryWorkloadAnalysis --section LaunchStats --section Occupancy)
    ;;
  roofline)
    section_args=(--section SpeedOfLight --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis)
    ;;
  *)
    echo "unknown SET=$SET (speedoflight|full|mem|roofline)" >&2
    exit 4
    ;;
esac

echo "META ncu=$NCU_BIN set=$SET out=$out_base bin=$BIN shape=$S $H $I $E $k"
"$NCU_BIN" \
  --target-processes all \
  --kernel-name-base function \
  --kernel-name "$KERNEL_REGEX" \
  --launch-skip 0 \
  --launch-count 1 \
  --export "$out_base" \
  --force-overwrite \
  "${section_args[@]}" \
  "$BIN" "$S" "$H" "$I" "$E" "$k" 0 "$WARMUP" "$RUNS"

echo "WROTE ${out_base}.ncu-rep"
ls -la "${out_base}".* 2>/dev/null || true
