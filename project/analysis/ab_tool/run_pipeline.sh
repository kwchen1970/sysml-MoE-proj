#!/bin/bash
# One-shot: parse A/B out → keep gate → plots.
#
# Usage:
#   ./run_pipeline.sh job.out --cand HYB17 --base HYB2 \
#       --out-dir /tmp/hyb17_gate \
#       --media-dir "$STORE/media/hyb17_vs_hyb2"
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${AB_TOOL_PYTHON:-python3}
JOB_OUT=
CAND=
BASE=HYB22
OUT_DIR=
MEDIA_DIR=
ALSO_VS=(HYB17 HYB2 BK64)
IDENTITY=0
REQUIRE_ANCHOR=0
TITLE=
PHASE=
AGG=median
EXTRA_PARSE=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cand) CAND=$2; shift 2 ;;
    --base) BASE=$2; shift 2 ;;
    --out-dir) OUT_DIR=$2; shift 2 ;;
    --media-dir) MEDIA_DIR=$2; shift 2 ;;
    --also-vs) ALSO_VS=($2); shift 2 ;;
    --identity) IDENTITY=1; shift ;;
    --require-anchor-win) REQUIRE_ANCHOR=1; shift ;;
    --title) TITLE=$2; shift 2 ;;
    --phase) PHASE=$2; shift 2 ;;
    --agg) AGG=$2; shift 2 ;;
    --min-geo-margin-pct) EXTRA_PARSE+=(--min-geo-margin-pct "$2"); shift 2 ;;
    --auto-margin) EXTRA_PARSE+=(--auto-margin); shift ;;
    --require-loo-margin) EXTRA_PARSE+=(--require-loo-margin); shift ;;
    --error-max) EXTRA_PARSE+=(--error-max "$2"); shift 2 ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *)
      if [[ -z "$JOB_OUT" ]]; then
        JOB_OUT=$1
        shift
      else
        echo "unexpected arg: $1" >&2
        exit 1
      fi
      ;;
  esac
done

: "${JOB_OUT:?job out path required}"
: "${CAND:?--cand required}"
OUT_DIR=${OUT_DIR:-$(mktemp -d /tmp/ab-gate.XXXXXX)}
TITLE=${TITLE:-"${CAND} vs ${BASE}"}

PARSE_ARGS=(--cand "$CAND" --base "$BASE" --agg "$AGG" --out "$OUT_DIR" --also-vs "${ALSO_VS[@]}")
[[ "$IDENTITY" == 1 ]] && PARSE_ARGS+=(--identity)
[[ "$REQUIRE_ANCHOR" == 1 ]] && PARSE_ARGS+=(--require-anchor-win)
[[ -n "$PHASE" ]] && PARSE_ARGS+=(--phase "$PHASE")
if ((${#EXTRA_PARSE[@]})); then
  PARSE_ARGS+=("${EXTRA_PARSE[@]}")
fi

"$PYTHON" "${SCRIPT_DIR}/parse_keep_gate.py" "$JOB_OUT" "${PARSE_ARGS[@]}"

TAGS=("$BASE" "$CAND")
for t in "${ALSO_VS[@]}"; do
  [[ "$t" == "$BASE" || "$t" == "$CAND" ]] && continue
  # Only pass tags that appear in compare.csv header.
  header=$(head -1 "${OUT_DIR}/compare.csv")
  [[ ",${header}," == *",${t},"* ]] && TAGS+=("$t")
done

PLOT_DIR=${MEDIA_DIR:-"${OUT_DIR}/plots"}
"$PYTHON" "${SCRIPT_DIR}/plot_ab.py" "${OUT_DIR}/compare.csv" \
  --cand "$CAND" --base "$BASE" --tags "${TAGS[@]}" \
  --out-dir "$PLOT_DIR" --title "$TITLE"

# Mirror compare.csv next to plots for review.
cp -f "${OUT_DIR}/compare.csv" "${PLOT_DIR}/compare.csv"
cp -f "${OUT_DIR}/keep_gate.txt" "${PLOT_DIR}/keep_gate.txt"
cp -f "${OUT_DIR}/keep_gate.json" "${PLOT_DIR}/keep_gate.json"

echo "PIPELINE_OK out=$OUT_DIR plots=$PLOT_DIR"
cat "${OUT_DIR}/keep_gate.txt"
