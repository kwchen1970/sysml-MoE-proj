#!/bin/bash
# Submit a lane-safe interleaved A/B job under /pscratch/.../ab/tool/.
#
# Usage:
#   BINS="HYB17=/pscratch/.../ab/testFlashMoE.hyb17 CAND=/pscratch/.../ab/tool/bins/testFlashMoE.cand" \
#     ./submit_ab.sh --cand CAND --name tool-cand1
#
# Optional env: ROUNDS WARMUP RUNS ERROR_MAX SHAPES_OVERRIDE QUEUE TIME ACCOUNT
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AB_ROOT=${AB_ROOT:-/pscratch/sd/r/rmf253/moe-exp/ab/tool}
SBATCH_SCRIPT=${SBATCH_SCRIPT:-"${SCRIPT_DIR}/ab_bench.sbatch"}
NAME=tool-ab
CAND_TAG=
BASE_TAG=HYB17
QUEUE=${QUEUE:-debug}
TIME=${TIME:-00:30:00}
ACCOUNT=${ACCOUNT:-m4999_g}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME=$2; shift 2 ;;
    --cand) CAND_TAG=$2; shift 2 ;;
    --base) BASE_TAG=$2; shift 2 ;;
    --script) SBATCH_SCRIPT=$2; shift 2 ;;
    --root) AB_ROOT=$2; shift 2 ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

: "${BINS:?set BINS=\"TAG=path ...\" before calling submit_ab.sh}"

# Default to more rounds than the old ROUNDS=3 so ~1% keeps clear the noise floor.
export ROUNDS=${ROUNDS:-5}
export WARMUP=${WARMUP:-2}
export RUNS=${RUNS:-16}

mkdir -p "${AB_ROOT}/jobs" "${AB_ROOT}/bins" "${AB_ROOT}/parsed" "${AB_ROOT}/ncu" "${AB_ROOT}/noise"

OUT="${AB_ROOT}/jobs/${NAME}-%j.out"
export BINS ROUNDS WARMUP RUNS ERROR_MAX SHAPES_OVERRIDE CAND_TAG BASE_TAG AB_ROOT SMOKE TIMEOUT_S

echo "Submitting name=$NAME cand=${CAND_TAG:-none} base=$BASE_TAG"
echo "  BINS=$BINS"
echo "  out=$OUT"

sbatch \
  -A "$ACCOUNT" \
  -q "$QUEUE" \
  -t "$TIME" \
  -J "$NAME" \
  -o "$OUT" \
  --export=ALL \
  "$SBATCH_SCRIPT"
