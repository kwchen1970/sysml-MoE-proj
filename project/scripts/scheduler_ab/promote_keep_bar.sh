#!/bin/bash
# Promote KEEP_BASE after confirm job succeeds.
#
#   ./promote_keep_bar.sh              # check KEEP_BASE_CONFIRM_JOB from keep_bar.env
#   ./promote_keep_bar.sh 61380698     # explicit job id
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/keep_bar.env"
HOST=${NERSC_HOST:-perlmutter.nersc.gov}
CAND=${KEEP_CANDIDATE:-HYB22}
PREV=${KEEP_BASE:-HYB17}
JOB=${1:-${KEEP_CANDIDATE_RECONFIRM_JOB:-${KEEP_CANDIDATE_CONFIRM_JOB:?}}}
MARGIN=${PROMOTE_MIN_GEO_MARGIN_PCT:-1.0}

state=$(ssh -o BatchMode=yes "$HOST" \
  "sacct -j $JOB --format=State -n -P 2>/dev/null | head -1 | tr -d ' '" || true)
echo "confirm job $JOB state=${state:-unknown} cand=$CAND vs $PREV margin=$MARGIN"

case "${state}" in
  COMPLETED|COMPLETED+)
    ;;
  *)
    echo "not promoting: want COMPLETED, got ${state:-unknown}"
    echo "KEEP_BASE stays $KEEP_BASE (candidate $CAND status=${KEEP_CANDIDATE_STATUS:-unknown})"
    exit 2
    ;;
esac

# Fetch confirm out and require solid KEEP (not MARGINAL) vs current tip.
OUT_DIR=$(mktemp -d /tmp/promote-keep.XXXXXX)
remote=$(ssh -o BatchMode=yes "$HOST" \
  "ls /pscratch/sd/r/rmf253/moe-exp/ab/main/jobs/*${JOB}*.out \
      /pscratch/sd/r/rmf253/moe-exp/ab/main/*${JOB}*.out 2>/dev/null | head -1")
if [[ -z "$remote" ]]; then
  echo "no out file for job $JOB under ab/main/"
  exit 3
fi
scp -q "${HOST}:${remote}" "$OUT_DIR/confirm.out"
PYTHON=${AB_TOOL_PYTHON:-python3}
GATE="${SCRIPT_DIR}/../../analysis/ab_tool/parse_keep_gate.py"
"$PYTHON" "$GATE" "$OUT_DIR/confirm.out" \
  --cand "$CAND" --base "$PREV" --agg median --min-geo-margin-pct "$MARGIN" \
  --out "$OUT_DIR/gate" | tee "$OUT_DIR/gate.log"
verdict=$(awk '/^VERDICT /{print $2; exit}' "$OUT_DIR/gate/keep_gate.txt")
if [[ "$verdict" != "KEEP" ]]; then
  echo "confirm verdict=$verdict for $CAND vs $PREV — not promoting (need KEEP, not MARGINAL/NO_KEEP)"
  cat "$OUT_DIR/gate/keep_gate.txt"
  exit 4
fi

# Promote candidate → KEEP_BASE.
tmp=$(mktemp)
cat > "$tmp" <<EOF
# Promoted by promote_keep_bar.sh from job $JOB on $(date -Is)
KEEP_BASE=$CAND
KEEP_BASE_STATUS=confirmed
KEEP_BASE_CONFIRMED=1
KEEP_BASE_BRANCH=${KEEP_CANDIDATE_BRANCH:-unknown}
KEEP_BASE_SHA=${KEEP_CANDIDATE_SHA:-unknown}
KEEP_BASE_PREV=$PREV
KEEP_CANDIDATE=
KEEP_CANDIDATE_STATUS=
KEEP_CANDIDATE_CONFIRMED=
BASE_TAG=$CAND
AGG=median
EOF
mv "$tmp" "${SCRIPT_DIR}/keep_bar.env"
echo "PROMOTED KEEP_BASE=$CAND status=confirmed (job $JOB)"
cat "${SCRIPT_DIR}/keep_bar.env"
