# FlashMoE A/B noise floor (tool lane)

Basis: interleaved job **61377792** (HYB17 keep A/B, ROUNDS=5, RUNS=16) until dedicated identity **61380130** finishes.

## Numbers

| Metric | Value |
|---|---|
| Mean per-shape CV (HYB17) | **2.96%** |
| Odd/even identity geo (HYB17) | **+0.939%** |
| Recommended hard margin | **1.0%** default / **1.41%** with `--auto-margin` |
| HYB17 vs HYB2 geo | −1.274% (median −2.08%, LOO [−1.81%, **−0.81%**]) |
| HYB17 vs BK64 geo | −4.696% (LOO [−5.08%, −4.31%]) |

## How other lanes should gate

```bash
source /pscratch/.../ab/tool/scripts/scheduler_ab/lane_env.sh <lane>
# submit with ROUNDS=8 (default via submit_ab.sh)
# parse with default margin 1.0%; add --auto-margin for ~1.4%
```

- Treat keeps with `|geo| ≲ 1–1.5%` as **provisional** unless LOO hi also clears the margin.
- HYB17 vs HYB2 is a **shape-fragile** keep under these rules (LOO hi fails 1%).
- HYB17 vs BK64 is a **robust** keep.

## Pending

Full identity study job **61380130** (`tool-noise`) — will replace the odd/even estimate when complete.
