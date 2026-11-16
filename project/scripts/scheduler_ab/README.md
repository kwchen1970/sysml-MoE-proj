# FlashMoE interleaved A/B harness (tool lane)

Lane-safe measurement loop for keep / no-keep decisions.

## Defaults (tightened 2026-11-15)

| Knob | Default | Why |
|---|---|---|
| `ROUNDS` | **8** | 5-round CV ≈ 3%; more rounds shrink geo noise |
| `RUNS` / `WARMUP` | 16 / 2 | same as prior keep jobs |
| `ERROR_MAX` | 0.25% | correctness band |
| `--min-geo-margin-pct` | **1.0** | clears ~±1% odd/even identity floor |
| `--base` | **`HYB17`** | effective keep tip (`KEEP_BASE_CONFIRMED=1`) |
| `--agg` | **`median`** | matches main-lane keep CSVs |
| HYB22 | **MARGINAL** | three-job noise-fragile; **do not promote** (see below) |
| promote | `--reject-near-noise --require-loo-margin --auto-margin` | refuses KEEP when \|geo\| < mean CV |

`KEEP` requires correct errors, full shape coverage, and geo(cand/base) ≤ −margin% (per-shape **median**). Faster-but-below-margin → **MARGINAL**. See `keep_bar.env`.

**HYB22 three-job story:** multi-tag 61378825 −3.7%; clean confirm 61382160 −0.037% MARGINAL; 8-round reconfirm 61383530 raw KEEP −4.337% but \|geo\| < CV 4.73% → promote refuses (MARGINAL). Bar stays **HYB17**.

**Multi-tag warning:** packing many variants in one interleaved job can invent large geo wins that vanish on a clean 2-way confirm. Prefer 2–3 tags.

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

BINS="HYB17=$AB_BINS/testFlashMoE.hyb17 CAND=$AB_BINS/testFlashMoE.cand" \
  "$SCRIPTS/submit_ab.sh" --root "$AB_ROOT" --name "${LANE}-cand1" --cand CAND --base HYB17
```

Or one-shot: `"$SCRIPTS/submit_ab.sh" --lane main --cand CAND ...` (still needs `BINS`).

### Fetch + gate after the job

```bash
project/scripts/scheduler_ab/fetch_job.sh <jobid> \
  --cand CAND --base HYB17 --agg median \
  --media-dir /path/to/Project/media/${LANE}_cand_vs_hyb17
# Promote only after a clean KEEP that also clears --reject-near-noise:
#   ./promote_keep_bar.sh <confirm_jobid>   # currently refuses HYB22
```


## Tool-lane quick keep A/B

```bash
source ./lane_env.sh tool
export BINS="HYB17=$AB_BINS/testFlashMoE.hyb17 CAND=$AB_BINS/testFlashMoE.cand"
./submit_ab.sh --cand CAND --name tool-cand1 --base HYB17
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
