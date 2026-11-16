# FlashMoE interleaved A/B harness (tool lane)

Lane-safe measurement loop for keep / no-keep decisions. All scratch I/O defaults to:

```
/pscratch/sd/r/rmf253/moe-exp/ab/tool/{jobs,bins,ncu,noise,parsed}
```

Other lanes should set `AB_ROOT=/pscratch/.../ab/<lane>` (or use their own copies) so binaries and `.out` files never clobber each other.

## Quick keep A/B

```bash
# On Perlmutter, after building/copying a candidate binary into ab/tool/bins/
export BINS="BK64=/pscratch/sd/r/rmf253/moe-exp/ab/testFlashMoE.bk64 \
HYB2=/pscratch/sd/r/rmf253/moe-exp/ab/testFlashMoE.hyb2 \
HYB17=/pscratch/sd/r/rmf253/moe-exp/ab/testFlashMoE.hyb17 \
CAND=/pscratch/sd/r/rmf253/moe-exp/ab/tool/bins/testFlashMoE.cand"

ROUNDS=5 CAND_TAG=CAND BASE_TAG=HYB17 \
  ./submit_ab.sh --name tool-cand1 --cand CAND --base HYB17
```

Harness features vs the old flat `ab_bench.sbatch`:

- unique `#SBATCH -o` under `ab/tool/jobs/`
- `META` lines (host, md5 of each binary, rounds/runs)
- optional smoke shape before the suite (`SMOKE=1`)
- per-row `ERROR_MAX` gate (default 0.25%)
- rotating interleaved order unchanged

## Auto keep gate + plots

After the job finishes (or on a checked-in `.out`):

```bash
project/analysis/ab_tool/run_pipeline.sh path/to/job.out \
  --cand CAND --base HYB17 \
  --out-dir /tmp/cand_gate \
  --media-dir /path/to/Project/media/cand_vs_hyb17 \
  --require-anchor-win   # optional stricter gate
```

`KEEP` requires:

1. no `FAIL` lines and cand `error_pct <= ERROR_MAX`
2. geometric mean of per-shape `(cand/base)` **&lt; 1** (cand faster)
3. full shape coverage for cand and base

Geo vs `HYB2` / `BK64` is always reported when those tags exist. Prefer beating **HYB17**.

### Measurement quality

On job 61377792 (ROUNDS=5), per-shape CV was ~3% and a simulated odd/even identity geo swung ~±1%. Keeps near **1%** are in the noise. Practical rules:

- use `ROUNDS=8`+ for candidate vs HYB17 decisions
- pass `--min-geo-margin-pct 1.0` (or higher after the noise study) when you need a hard margin above the identity floor
- the gate prints a WARNING when `|geo| < mean cand CV`

## Noise study

```bash
sbatch project/scripts/scheduler_ab/noise_study.sbatch
```

Runs HYB17-vs-HYB17 identity (high rounds, compact shapes) then a BK64/HYB2/HYB17 margin suite. Parse with:

```bash
# identity section only: extract manually or pass full out with --identity
project/analysis/ab_tool/run_pipeline.sh noise.out \
  --cand HYB17B --base HYB17A --identity \
  --out-dir /tmp/noise --media-dir /path/to/media/noise_hyb17
```

## Nsight Compute

```bash
# interactive / compute node
LABEL=hyb17-anchor SET=speedoflight \
  ./ncu_wrap.sh /pscratch/.../ab/testFlashMoE.hyb17 4096 2048 1408 16 6

# batch
BINS="HYB17=/pscratch/.../ab/testFlashMoE.hyb17" SET=speedoflight \
  sbatch -J tool-ncu-hyb17 -o /pscratch/.../ab/tool/ncu/%x-%j.out \
    --export=ALL project/scripts/scheduler_ab/ncu_profile.sbatch
```

Reports land in `ab/tool/ncu/`.
