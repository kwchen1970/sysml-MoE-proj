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
JOB=${1:-${KEEP_BASE_CONFIRM_JOB:?}}

state=$(ssh -o BatchMode=yes "$HOST" \
  "sacct -j $JOB --format=State -n -P 2>/dev/null | head -1 | tr -d ' '" || true)
echo "confirm job $JOB state=${state:-unknown}"

case "${state}" in
  COMPLETED|COMPLETED+)
    ;;
  *)
    echo "not promoting: want COMPLETED, got ${state:-unknown}"
    echo "KEEP_BASE stays $KEEP_BASE (status=$KEEP_BASE_STATUS)"
    exit 2
    ;;
esac

# Fetch confirm out and require KEEP vs prior tip (HYB17) under median agg.
OUT_DIR=$(mktemp -d /tmp/promote-keep.XXXXXX)
remote=$(ssh -o BatchMode=yes "$HOST" \
  "ls /pscratch/sd/r/rmf253/moe-exp/ab/main/*${JOB}*.out 2>/dev/null | head -1")
if [[ -z "$remote" ]]; then
  echo "no out file for job $JOB under ab/main/"
  exit 3
fi
scp -q "${HOST}:${remote}" "$OUT_DIR/confirm.out"
PYTHON=${AB_TOOL_PYTHON:-python3}
GATE="${SCRIPT_DIR}/../../analysis/ab_tool/parse_keep_gate.py"
"$PYTHON" "$GATE" "$OUT_DIR/confirm.out" \
  --cand "$KEEP_BASE" --base "${KEEP_BASE_PREV:-HYB17}" --agg median \
  --out "$OUT_DIR/gate" | tee "$OUT_DIR/gate.log"
if ! grep -q '^VERDICT KEEP' "$OUT_DIR/gate/keep_gate.txt"; then
  echo "confirm did not KEEP $KEEP_BASE vs $KEEP_BASE_PREV — not promoting"
  cat "$OUT_DIR/gate/keep_gate.txt"
  exit 4
fi

# Flip confirmed bit in keep_bar.env
tmp=$(mktemp)
sed \
  -e 's/^KEEP_BASE_STATUS=.*/KEEP_BASE_STATUS=confirmed/' \
  -e 's/^KEEP_BASE_CONFIRMED=.*/KEEP_BASE_CONFIRMED=1/' \
  "${SCRIPT_DIR}/keep_bar.env" > "$tmp"
mv "$tmp" "${SCRIPT_DIR}/keep_bar.env"
echo "PROMOTED KEEP_BASE=$KEEP_BASE status=confirmed (job $JOB)"
cat "${SCRIPT_DIR}/keep_bar.env"
