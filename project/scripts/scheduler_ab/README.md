# FlashMoE interleaved A/B harness (tool lane)

Lane-safe measurement loop for keep / no-keep decisions.

## Defaults (tightened 2026-11-15)

| Knob | Default | Why |
|---|---|---|
| `ROUNDS` | **8** | 5-round CV ≈ 3%; more rounds shrink geo noise |
| `RUNS` / `WARMUP` | 16 / 2 | same as prior keep jobs |
| `ERROR_MAX` | 0.25% | correctness band |
| `--min-geo-margin-pct` | **1.0** | clears ~±1% odd/even identity floor |
| `--base` | **`HYB22`** | provisional keep tip (confirm job 61380698) |
| `--agg` | **`median`** | matches main-lane keep CSVs; mean under-reports multi-tag A/Bs |
| `--auto-margin` | off | raises margin to bootstrap×1.5 (~1.4% on hyb17-61377792) |

`KEEP` requires correct errors, full shape coverage, and geo(cand/base) ≤ −margin% (per-shape **median** latency by default). See `keep_bar.env`.

## Paths

```
/pscratch/sd/r/rmf253/moe-exp/ab/<lane>/{jobs,bins,ncu,noise,parsed}
/pscratch/sd/r/rmf253/moe-exp/ab/tool/scripts/scheduler_ab/   # shared helpers
```

Shared keep-stack binaries stay in `ab/testFlashMoE.{hyb17,hyb2,hyb14,bk64}`; lanes symlink them into their own `bins/`.

## Other lanes — copy/paste

```bash
# On Perlmutter login/compute:
SCRIPTS=/pscratch/sd/r/rmf253/moe-exp/ab/tool/scripts/scheduler_ab
source "$SCRIPTS/lane_env.sh" main          # or: ncu | epi | sched | tool
# builds AB_ROOT=$SCRATCH_AB/$LANE and links baselines into $AB_BINS

# After you build a candidate:
cp /path/to/cmake-build-release/testFlashMoE "$AB_BINS/testFlashMoE.cand"

BINS="BK64=$AB_BINS/testFlashMoE.bk64 HYB2=$AB_BINS/testFlashMoE.hyb2 \
HYB17=$AB_BINS/testFlashMoE.hyb17 HYB22=$AB_BINS/testFlashMoE.hyb22 \
CAND=$AB_BINS/testFlashMoE.cand" \
  "$SCRIPTS/submit_ab.sh" --root "$AB_ROOT" --name "${LANE}-cand1" --cand CAND --base HYB22
```

Or one-shot: `"$SCRIPTS/submit_ab.sh" --lane main --cand CAND ...` (still needs `BINS`).

### Fetch + gate after the job

```bash
# From a laptop with the repo worktree / clone:
project/scripts/scheduler_ab/fetch_job.sh <jobid> \
  --cand CAND --base HYB22 --agg median \
  --media-dir /path/to/Project/media/${LANE}_cand_vs_hyb22
# Add --auto-margin for the stricter bootstrap margin.
# After confirm: ./promote_keep_bar.sh 61380698
```

## Tool-lane quick keep A/B

```bash
source ./lane_env.sh tool
export BINS="BK64=$AB_BINS/testFlashMoE.bk64 HYB2=$AB_BINS/testFlashMoE.hyb2 \
HYB17=$AB_BINS/testFlashMoE.hyb17 HYB22=$AB_BINS/testFlashMoE.hyb22 \
CAND=$AB_BINS/testFlashMoE.cand"
./submit_ab.sh --cand CAND --name tool-cand1 --base HYB22
```

Harness extras vs the old flat `ab_bench.sbatch`:

- unique `#SBATCH -o` under `ab/<lane>/jobs/`
- `META` lines (host, md5 of each binary, rounds/runs)
- smoke shape before the suite (`SMOKE=1`)
- per-row `ERROR_MAX` gate
- rotating interleaved order unchanged

## Auto keep gate + plots

```bash
project/analysis/ab_tool/run_pipeline.sh path/to/job.out \
  --cand CAND --base HYB17 \
  --out-dir /tmp/cand_gate \
  --media-dir /path/to/Project/media/cand_vs_hyb17
```

Gate also prints median Δ, leave-one-out geo range, mean CV, and odd/even identity bootstrap.

## Noise study

```bash
# Full (regular QOS): identity + margin suite — job 61378923
sbatch ./noise_study.sbatch

# Compact identity only (debug QOS, faster):
sbatch ./noise_identity_debug.sbatch
```

Parse:

```bash
project/analysis/ab_tool/run_pipeline.sh noise.out \
  --cand HYB17B --base HYB17A --identity --phase identity \
  --out-dir /tmp/noise --media-dir /path/to/media/noise_hyb17
```

## Nsight Compute

```bash
LABEL=hyb17-anchor SET=speedoflight \
  ./ncu_wrap.sh $AB_BINS/testFlashMoE.hyb17 4096 2048 1408 16 6

BINS="HYB17=$AB_BINS/testFlashMoE.hyb17" SET=speedoflight \
  sbatch -J ${LANE}-ncu -o $AB_NCU/%x-%j.out --export=ALL ./ncu_profile.sbatch
```
