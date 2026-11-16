#!/bin/bash
# Smoke-test keep gate against checked-in hyb17 A/B output.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
PYTHON=${AB_TOOL_PYTHON:-python3}
OUT=${1:-/tmp/ab-tool-selftest}
HYB17_OUT=${HYB17_OUT:-"${ROOT}/project/results/shared_a_ab/hyb17-61377792.out"}
# Fallback to sibling main-clone results if worktree has no copy.
if [[ ! -f "$HYB17_OUT" ]]; then
  HYB17_OUT=/Users/ryanfujii/CS5470/sysml-MoE-proj/project/results/shared_a_ab/hyb17-61377792.out
fi
mkdir -p "$OUT"

# Default margin 1.0%: HYB17 vs HYB2 (−1.27%) still KEEP.
"$PYTHON" "${SCRIPT_DIR}/parse_keep_gate.py" "$HYB17_OUT" \
  --cand HYB17 --base HYB2 --out "${OUT}/vs_hyb2" | tee "${OUT}/vs_hyb2.log"
grep -q '^VERDICT KEEP' "${OUT}/vs_hyb2/keep_gate.txt"

# HYB14 does not beat HYB17.
"$PYTHON" "${SCRIPT_DIR}/parse_keep_gate.py" "$HYB17_OUT" \
  --cand HYB14 --base HYB17 --out "${OUT}/hyb14_vs_hyb17" | tee "${OUT}/hyb14.log"
grep -q '^VERDICT NO_KEEP' "${OUT}/hyb14_vs_hyb17/keep_gate.txt"

# Explicit 5% margin rejects HYB17 vs HYB2.
"$PYTHON" "${SCRIPT_DIR}/parse_keep_gate.py" "$HYB17_OUT" \
  --cand HYB17 --base HYB2 --min-geo-margin-pct 5 --out "${OUT}/strict" | tee "${OUT}/strict.log"
grep -q '^VERDICT NO_KEEP' "${OUT}/strict/keep_gate.txt"

# --auto-margin raises to ~1.41% from odd/even bootstrap → rejects −1.27%.
"$PYTHON" "${SCRIPT_DIR}/parse_keep_gate.py" "$HYB17_OUT" \
  --cand HYB17 --base HYB2 --auto-margin --out "${OUT}/auto" | tee "${OUT}/auto.log"
grep -q '^VERDICT NO_KEEP' "${OUT}/auto/keep_gate.txt"

# HYB17 vs BK64 (−4.7%) clears default and auto margins.
"$PYTHON" "${SCRIPT_DIR}/parse_keep_gate.py" "$HYB17_OUT" \
  --cand HYB17 --base BK64 --auto-margin --out "${OUT}/vs_bk64" | tee "${OUT}/vs_bk64.log"
grep -q '^VERDICT KEEP' "${OUT}/vs_bk64/keep_gate.txt"

echo "SELF_TEST_OK"
