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

# Convenience refs to keep-stack binaries (symlink into lane bins).
# Never overwrite a real binary — only create missing links or refresh symlinks.
for tag in hyb22 hyb17 hyb2 hyb14 bk64; do
  src="${SCRATCH_AB}/testFlashMoE.${tag}"
  if [[ ! -x "$src" ]]; then
    src="${SCRATCH_AB}/main/bins/testFlashMoE.${tag}"
  fi
  dst="${AB_BINS}/testFlashMoE.${tag}"
  if [[ ! -x "$src" ]]; then
    continue
  fi
  if [[ -L "$dst" ]]; then
    ln -sfn "$src" "$dst"
  elif [[ ! -e "$dst" ]]; then
    ln -sf "$src" "$dst"
  fi
done

# shellcheck disable=SC1091
source "${AB_SCRIPTS}/keep_bar.env" 2>/dev/null || true
echo "lane_env: LANE=$LANE AB_ROOT=$AB_ROOT keep_base=${KEEP_BASE:-HYB22} (${KEEP_BASE_STATUS:-unknown})"
echo "  baselines: $AB_BINS/testFlashMoE.{hyb22,hyb17,hyb2,hyb14,bk64}"
echo "  submit: BINS=... $AB_SCRIPTS/submit_ab.sh --root \$AB_ROOT --name ${LANE}-cand1 --cand CAND --base ${KEEP_BASE:-HYB22}"
