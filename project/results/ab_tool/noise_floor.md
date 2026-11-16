# FlashMoE A/B noise floor (tool lane)

## Multi-tag fragility (HYB22 lesson)

| Job | Setup | Median geo HYB22/HYB17 |
|---|---|---|
| **61378825** | 7-tag interleaved (HYB17–23), ROUNDS=3 | **−3.669%** (looked like KEEP) |
| **61382160** | clean 2-way HYB17+HYB22, ROUNDS=5 | **−0.037%** (**MARGINAL**) |
| **61383530** | 8-round reconfirm | in flight |

The multi-tag win did **not** survive a dedicated confirm. Leave-one-out on 61382160 crosses zero (`[+1.6%, −1.4%]`).

### Rules
1. Prefer **2–3 tags** only.
2. Require dedicated confirm before promoting.
3. `|geo| ≲ 1%` or geo inside CV / odd-even identity → **MARGINAL**, not KEEP.
4. Effective bar remains **HYB17** until a clean confirm KEEP.

## Earlier noise basis (hyb17-61377792, ROUNDS=5)
| Metric | Value |
|---|---|
| Mean per-shape CV (HYB17) | ~3% |
| Odd/even identity geo | ~±1% |
| Default `--min-geo-margin-pct` | **1.0** |
