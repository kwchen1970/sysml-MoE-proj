#!/bin/bash
# Source this to bind a lane to unique ab/<lane>/ paths.
#
#   source /pscratch/.../ab/tool/scripts/scheduler_ab/lane_env.sh <lane>
#   # or: LANE=main source .../lane_env.sh
#
# Exports:
#   LANE AB_ROOT AB_JOBS AB_BINS AB_NCU AB_NOISE AB_PARSED AB_SCRIPTS
#   SCRATCH_AB  (shared reference binaries: hyb17/hyb2/bk64/…)
set -euo pipefail

SCRATCH_AB=${SCRATCH_AB:-/pscratch/sd/r/rmf253/moe-exp/ab}
LANE=${1:-${LANE:?set LANE or pass lane name (tool|main|ncu|epi|sched|…)}}

# Normalize: lowercase, no spaces.
LANE=$(printf '%s' "$LANE" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9._-')
if [[ -z "$LANE" ]]; then
  echo "lane_env: empty lane after normalize" >&2
  return 2 2>/dev/null || exit 2
fi

export LANE
export SCRATCH_AB
export AB_ROOT="${SCRATCH_AB}/${LANE}"
export AB_JOBS="${AB_ROOT}/jobs"
export AB_BINS="${AB_ROOT}/bins"
export AB_NCU="${AB_ROOT}/ncu"
export AB_NOISE="${AB_ROOT}/noise"
export AB_PARSED="${AB_ROOT}/parsed"
export AB_SCRIPTS="${SCRATCH_AB}/tool/scripts/scheduler_ab"

mkdir -p "$AB_JOBS" "$AB_BINS" "$AB_NCU" "$AB_NOISE" "$AB_PARSED"

# Convenience refs to shared keep-stack binaries (do not copy; symlink into lane bins).
for tag in hyb17 hyb2 hyb14 bk64; do
  src="${SCRATCH_AB}/testFlashMoE.${tag}"
  dst="${AB_BINS}/testFlashMoE.${tag}"
  if [[ -x "$src" && ! -e "$dst" ]]; then
    ln -sf "$src" "$dst"
  fi
done

echo "lane_env: LANE=$LANE AB_ROOT=$AB_ROOT"
echo "  shared baselines: $AB_BINS/testFlashMoE.{hyb17,hyb2,hyb14,bk64}"
echo "  submit: BINS=... $AB_SCRIPTS/submit_ab.sh --root \$AB_ROOT --name ${LANE}-cand1 --cand CAND --base HYB17"
