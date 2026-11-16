#!/bin/bash
# Submit a lane-safe interleaved A/B job under ab/<lane>/.
#
# Usage (tool lane):
#   source ./lane_env.sh tool
#   BINS="BK64=$AB_BINS/testFlashMoE.bk64 HYB2=$AB_BINS/testFlashMoE.hyb2 \
#         HYB17=$AB_BINS/testFlashMoE.hyb17 CAND=$AB_BINS/testFlashMoE.cand" \
#     ./submit_ab.sh --cand CAND --name tool-cand1
#
# Other lanes: source ./lane_env.sh main|ncu|epi|sched then same pattern with --root $AB_ROOT.
# Optional env: ROUNDS WARMUP RUNS ERROR_MAX SHAPES_OVERRIDE QUEUE TIME ACCOUNT
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
set -a
# Defaults: ROUNDS=8, MIN_GEO_MARGIN_PCT=1.0, BASE_TAG=HYB17
source "${SCRIPT_DIR}/defaults.env"
set +a

AB_ROOT=${AB_ROOT:-/pscratch/sd/r/rmf253/moe-exp/ab/tool}
SBATCH_SCRIPT=${SBATCH_SCRIPT:-"${SCRIPT_DIR}/ab_bench.sbatch"}
NAME=${LANE:+${LANE}-}ab
NAME=${NAME:-tool-ab}
CAND_TAG=
BASE_TAG=${BASE_TAG:-HYB17}
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
    --lane)
      # shellcheck disable=SC1091
      source "${SCRIPT_DIR}/lane_env.sh" "$2"
      NAME="${LANE}-ab"
      shift 2
      ;;
    -h|--help)
      sed -n '1,25p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

: "${BINS:?set BINS=\"TAG=path ...\" before calling submit_ab.sh}"

export ROUNDS=${ROUNDS:-8}
export WARMUP=${WARMUP:-2}
export RUNS=${RUNS:-16}

mkdir -p "${AB_ROOT}/jobs" "${AB_ROOT}/bins" "${AB_ROOT}/parsed" "${AB_ROOT}/ncu" "${AB_ROOT}/noise"

OUT="${AB_ROOT}/jobs/${NAME}-%j.out"
export BINS ROUNDS WARMUP RUNS ERROR_MAX SHAPES_OVERRIDE CAND_TAG BASE_TAG AB_ROOT SMOKE TIMEOUT_S

echo "Submitting name=$NAME lane=${LANE:-tool} cand=${CAND_TAG:-none} base=$BASE_TAG rounds=$ROUNDS"
echo "  AB_ROOT=$AB_ROOT"
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
