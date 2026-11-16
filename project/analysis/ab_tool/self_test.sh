#!/bin/bash
# Smoke-test keep gate against checked-in A/B outputs.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
PYTHON=${AB_TOOL_PYTHON:-python3}
OUT=${1:-/tmp/ab-tool-selftest}
HYB17_OUT=${HYB17_OUT:-"${ROOT}/project/results/shared_a_ab/hyb17-61377792.out"}
HYB22_OUT=${HYB22_OUT:-"${ROOT}/project/results/ab_tool/hyb22-61378825.out"}
if [[ ! -f "$HYB17_OUT" ]]; then
  HYB17_OUT=/Users/ryanfujii/CS5470/sysml-MoE-proj/project/results/shared_a_ab/hyb17-61377792.out
fi
mkdir -p "$OUT"

# --- legacy HYB17 fixture (mean/median both OK) ---
"$PYTHON" "${SCRIPT_DIR}/parse_keep_gate.py" "$HYB17_OUT" \
  --cand HYB17 --base HYB2 --agg median --out "${OUT}/vs_hyb2" | tee "${OUT}/vs_hyb2.log"
grep -q '^VERDICT KEEP' "${OUT}/vs_hyb2/keep_gate.txt"

"$PYTHON" "${SCRIPT_DIR}/parse_keep_gate.py" "$HYB17_OUT" \
  --cand HYB14 --base HYB17 --agg median --out "${OUT}/hyb14_vs_hyb17" | tee "${OUT}/hyb14.log"
grep -q '^VERDICT NO_KEEP' "${OUT}/hyb14_vs_hyb17/keep_gate.txt"

"$PYTHON" "${SCRIPT_DIR}/parse_keep_gate.py" "$HYB17_OUT" \
  --cand HYB17 --base HYB2 --agg median --min-geo-margin-pct 5 --out "${OUT}/strict" | tee "${OUT}/strict.log"
# Faster than base but below a hard 5% margin → MARGINAL (not KEEP).
grep -q '^VERDICT MARGINAL' "${OUT}/strict/keep_gate.txt"

"$PYTHON" "${SCRIPT_DIR}/parse_keep_gate.py" "$HYB17_OUT" \
  --cand HYB17 --base BK64 --agg median --auto-margin --out "${OUT}/vs_bk64" | tee "${OUT}/vs_bk64.log"
grep -q '^VERDICT KEEP' "${OUT}/vs_bk64/keep_gate.txt"

# --- HYB22 multi-tag fixture still looks like KEEP (noise-fragile) ---
if [[ -f "$HYB22_OUT" ]]; then
  "$PYTHON" "${SCRIPT_DIR}/parse_keep_gate.py" "$HYB22_OUT" \
    --cand HYB22 --base HYB17 --agg median --out "${OUT}/hyb22_vs_hyb17" | tee "${OUT}/hyb22.log"
  grep -q '^VERDICT KEEP' "${OUT}/hyb22_vs_hyb17/keep_gate.txt"
  grep -E 'geo -3\.[0-9]+%' "${OUT}/hyb22_vs_hyb17/keep_gate.txt"
fi

# --- Clean 2-way confirm is MARGINAL (−0.037%) — do not promote ---
CONFIRM_OUT=${CONFIRM_OUT:-"${ROOT}/project/results/ab_tool/hyb22-confirm-61382160.out"}
if [[ -f "$CONFIRM_OUT" ]]; then
  "$PYTHON" "${SCRIPT_DIR}/parse_keep_gate.py" "$CONFIRM_OUT" \
    --cand HYB22 --base HYB17 --agg median --out "${OUT}/hyb22_confirm" | tee "${OUT}/hyb22_confirm.log"
  grep -q '^VERDICT MARGINAL' "${OUT}/hyb22_confirm/keep_gate.txt"
  grep -E 'geo -0\.0' "${OUT}/hyb22_confirm/keep_gate.txt"
fi

# --- 8-round reconfirm: raw KEEP but |geo|<CV → MARGINAL under --reject-near-noise ---
RECONFIRM_OUT=${RECONFIRM_OUT:-"${ROOT}/project/results/ab_tool/hyb22-reconfirm-61383530.out"}
if [[ -f "$RECONFIRM_OUT" ]]; then
  "$PYTHON" "${SCRIPT_DIR}/parse_keep_gate.py" "$RECONFIRM_OUT" \
    --cand HYB22 --base HYB17 --agg median --min-geo-margin-pct 1.0 \
    --out "${OUT}/hyb22_reconfirm_raw" | tee "${OUT}/hyb22_reconfirm_raw.log"
  grep -q '^VERDICT KEEP' "${OUT}/hyb22_reconfirm_raw/keep_gate.txt"
  "$PYTHON" "${SCRIPT_DIR}/parse_keep_gate.py" "$RECONFIRM_OUT" \
    --cand HYB22 --base HYB17 --agg median --min-geo-margin-pct 1.0 \
    --reject-near-noise --require-loo-margin --auto-margin \
    --out "${OUT}/hyb22_reconfirm_strict" | tee "${OUT}/hyb22_reconfirm_strict.log"
  grep -q '^VERDICT MARGINAL' "${OUT}/hyb22_reconfirm_strict/keep_gate.txt"
fi

echo "SELF_TEST_OK"
