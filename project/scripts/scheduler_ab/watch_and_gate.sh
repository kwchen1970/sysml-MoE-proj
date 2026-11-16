#!/bin/bash
# Poll a Slurm job until it leaves the queue, then run fetch_job → keep gate.
#
#   ./watch_and_gate.sh 61379335 --cand HYB17B --base HYB17A --identity --phase identity
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOST=${NERSC_HOST:-perlmutter.nersc.gov}
POLL_S=${POLL_S:-60}

if [[ $# -lt 1 ]]; then
  echo "usage: $0 JOBID [pipeline args...]" >&2
  exit 1
fi
JOBID=$1
shift

echo "watching job $JOBID on $HOST (poll ${POLL_S}s)"
while true; do
  state=$(ssh -o BatchMode=yes "$HOST" \
    "sacct -j $JOBID --format=State -n -P 2>/dev/null | head -1 | tr -d ' '" || true)
  queue=$(ssh -o BatchMode=yes "$HOST" \
    "squeue -j $JOBID -h -o '%T' 2>/dev/null | head -1" || true)
  echo "$(date -Is) queue=${queue:-none} sacct=${state:-unknown}"
  case "${state}" in
    COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*)
      break
      ;;
  esac
  if [[ -z "$queue" && -n "$state" && "$state" != "PENDING" && "$state" != "RUNNING" && "$state" != "CONFIGURING" ]]; then
    break
  fi
  sleep "$POLL_S"
done

exec "${SCRIPT_DIR}/fetch_job.sh" "$JOBID" "$@"
