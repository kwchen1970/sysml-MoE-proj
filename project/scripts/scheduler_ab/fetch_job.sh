#!/bin/bash
# Fetch a tool-lane job .out from Perlmutter scratch and run the keep-gate pipeline.
#
# Usage:
#   ./fetch_job.sh 61378923 --cand HYB17B --base HYB17A --identity
#   ./fetch_job.sh tool-noise-61378923.out --cand CAND --base HYB17 --media-dir ...
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/keep_bar.env" 2>/dev/null || true
AB_ROOT=${AB_ROOT:-/pscratch/sd/r/rmf253/moe-exp/ab/tool}
AB_SCRATCH=${AB_SCRATCH:-/pscratch/sd/r/rmf253/moe-exp/ab}
LOCAL_DIR=${LOCAL_DIR:-"${SCRIPT_DIR}/../../../results/ab_tool"}
PIPELINE=${PIPELINE:-"${SCRIPT_DIR}/../../analysis/ab_tool/run_pipeline.sh"}
HOST=${NERSC_HOST:-perlmutter.nersc.gov}
# Default gate baseline stays the confirmed keep bar (HYB17).
BASE_DEFAULT=${KEEP_BASE:-HYB17}

if [[ $# -lt 1 ]]; then
  echo "usage: $0 JOBID|remote-or-local.out [pipeline args...]" >&2
  echo "  searches ab/{tool,main,sched,epi,ncu}/ for JOBID outs" >&2
  exit 1
fi

SPEC=$1
shift

mkdir -p "$LOCAL_DIR"
if [[ -f "$SPEC" ]]; then
  LOCAL_OUT=$SPEC
elif [[ "$SPEC" == *.out ]]; then
  scp -q "${HOST}:${AB_ROOT}/jobs/${SPEC}" "$LOCAL_DIR/" 2>/dev/null \
    || scp -q "${HOST}:${AB_ROOT}/noise/${SPEC}" "$LOCAL_DIR/" 2>/dev/null \
    || scp -q "${HOST}:${AB_ROOT}/ncu/${SPEC}" "$LOCAL_DIR/"
  LOCAL_OUT="${LOCAL_DIR}/$(basename "$SPEC")"
else
  # bare job id — search tool + sibling lane trees
  remote=$(ssh -o BatchMode=yes "$HOST" \
    "ls ${AB_ROOT}/jobs/*-${SPEC}.out ${AB_ROOT}/noise/*-${SPEC}.out ${AB_ROOT}/ncu/*-${SPEC}.out \
         ${AB_SCRATCH}/main/jobs/*${SPEC}*.out \
         ${AB_SCRATCH}/sched/*${SPEC}*.out ${AB_SCRATCH}/epi/*${SPEC}*.out \
         ${AB_SCRATCH}/ncu/out/*${SPEC}*.out 2>/dev/null | head -1")
  [[ -n "$remote" ]] || { echo "no out for job $SPEC under $AB_SCRATCH/{tool,main,sched,epi,ncu}"; exit 2; }
  scp -q "${HOST}:${remote}" "$LOCAL_DIR/"
  LOCAL_OUT="${LOCAL_DIR}/$(basename "$remote")"
fi

echo "LOCAL_OUT=$LOCAL_OUT KEEP_BASE=${BASE_DEFAULT} (pass --base/--reject-near-noise as needed)"
exec "$PIPELINE" "$LOCAL_OUT" --out-dir "${LOCAL_DIR}/gate-$(basename "$LOCAL_OUT" .out)" "$@"
