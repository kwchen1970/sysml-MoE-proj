# MoE-layer campaign

BF16 gated MLP, Perlmutter A100-SXM4-80GB (L2 is 40 MB). Latency is the mean of 32 timed iterations unless noted. `blocks/SM` comes from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. Anchor: S=4096, H=2048, I=1408, E=16, k=6.

Arithmetic roof at the anchor is about 1.36 ms. Measured baseline is about 4.45 ms.

## Scheduler

Two-expert affinity (`exp/l2-expert-affinity`) then one-expert-when-it-fills (`exp/affinity-single-expert`).

Anchor, one GPU, three repeats of the two-expert policy against baseline:

| | repeat 1 | repeat 2 | repeat 3 | mean |
|---|---:|---:|---:|---:|
| baseline | 4.463 | 4.474 | 4.421 | 4.453 ms |
| two-expert | 4.101 | 4.090 | 4.060 | 4.084 ms |

That is −8.3%. The repeat spread is about 0.05 ms, and the gap is 0.37 ms.

The same policy on other shapes, one GPU:

| Shape | Baseline | Two-expert | Change |
|---|---:|---:|---:|
| S=1024 | 1.069 | 1.001 | −6.4% |
| S=16384 | 21.563 | 19.723 | −8.5% |
| I=1024 | 3.367 | 3.147 | −6.5% |
| I=4096 | 10.324 | 9.051 | −12.3% |
| E=64 | 4.765 | 4.465 | −6.3% |
| k=2 | 1.463 | 1.348 | −7.8% |

The saving grows with S, so it is paid per tile. It also grows as a fraction of runtime when I grows, which is when each expert's weights get larger than L2. One gated expert at I=1408 is about 17 MB. At I=4096 it is about 50 MB.

Nsight Compute on one anchor forward, after pausing DCGM:

| | Baseline | Two-expert |
|---|---:|---:|
| L2 throughput | 71.4% of peak | 51.1% of peak |
| Tensor pipe | 32.5% of peak | 36.3% of peak |
| DRAM throughput | ~717 GB/s | ~735 GB/s |
| Registers / thread | 255 | 255 |
| Dynamic shared / block | 65.5 KB | 65.5 KB |
| Occupancy limit, registers | 2 blocks | 2 blocks |
| Occupancy limit, shared | 2 blocks | 2 blocks |

L2 was the busiest pipe. Affinity lowered it and raised the tensor pipe. DRAM on this 80 GB A100 peaks near 2 TB/s, so the kernel is not sitting on DRAM peak. Occupancy cannot rise past 2 blocks: registers and shared memory both stop there.

Dropping the partner whenever the long pole can fill the free CTAs (`exp/affinity-single-expert`), one GPU, 32 iterations, paired with a fresh baseline and the two-expert binary:

| Shape | Baseline | Two-expert | Single | Single vs baseline |
|---|---:|---:|---:|---:|
| anchor | 4.488 | 4.088 | 3.924 | −12.6% |
| S=16384 | 21.561 | 19.801 | 19.346 | −10.3% |
| I=4096 | 10.371 | 9.054 | 8.865 | −14.5% |
| E=64 | 4.802 | 4.643 | 4.401 | −8.4% |
| 4 GPU, I=4096 | 10.484 | 9.297 | 9.125 | −13.0% |

Error stayed in the baseline band (about 0.04–0.12% at k=6). Occupancy stayed 2. The S=16384 and I=4096 single numbers are means of two runs; those pairs agreed to about 0.05 ms.

`exp/affinity-exclusive` never keeps a second expert in the hot set. The partner is only filler for idle CTAs. Paired against single-expert, 32 iterations:

| Shape | Single | Exclusive |
|---|---:|---:|
| anchor, repeat 1 | 3.931 | 3.918 |
| anchor, repeat 2 | 3.935 | 3.904 |
| S=16384 | 19.431 | 19.325 |
| I=4096 | 8.841 | 8.856 |
| E=64 | 4.400 | 4.423 |

The anchor means are 3.933 ms and 3.911 ms. That is 0.02 ms, inside the run-to-run spread of this comparison. The other three shapes move by about 0.5% in either direction. Refusing to prefer the partner when the long pole is short does not add another gain. The single-expert rule is where this knob stopped helping.

## Pipeline and tile width

| Branch | What changed | Anchor | blocks/SM |
|---|---|---:|---:|
| baseline | bN=128, 2 stages | 4.488 ms | 2 |
| `exp/deeper-k-pipeline` | bN=128, 3 stages | 6.075 ms | 1 |
| `exp/tile-n64` | bN=64, 2 stages | 7.670 ms | 2 |
| `exp/pipeline-fit-occupancy` | bN=64, 3 stages | 7.682 ms | 2 |

Three stages at bN=128 lost the second block and got slower. Narrowing N so three stages fit keeps 2 blocks, and the third stage matches the two-stage narrow tile. The extra stage is not the effect. The narrow tile is about 70% slower at the anchor, at S=16384, at I=4096, and at E=64. Registers still cap occupancy at 2, so the narrow tile did not buy another block.

## Dispatch

`exp/dispatch-cta-share` moved one-GPU latency by under 1% and was 2–5% slower on four GPUs.

## Resident warps (closed, mixed)

`exp/resident-warps` staged the gated activation through the output tile and raised `launch_bounds` to 3. After lifting `MAX_PROCESSORS` to 384, occupancy became 3 blocks/SM. Anchor moved 3.90 → 3.86 ms, but S=16384 and I=4096 got slower and I=4096 error rose to ~0.53%. Not kept as the default.

## Megakernel expert wave (in flight on this branch)

`exp/megakernel-expert-wave` packed up to four consecutive same-expert block-queue tiles into one doorbell. That code never fired under today’s affinity path: `runAffinity` calls `schedulerLoop(..., isMedley=true)` and the wave gate required `!isMedley`, and affinity spreads work across lanes so the sole-owner check almost never holds.

This branch (`cursor/scheduler-expert-wave-378d`) keeps the fused megakernel and:

1. Enables multi-lane waves whenever a scheduling pass holds only block-queue tiles (the common affinity GEMM1 case).
2. Budgets ready-queue slots by doorbells, not tiles, so waves stay correct across refill rounds.
3. Raises `kExpertWave` from 4 to 8 so affinity’s concentrated expert amortizes more doorbells per wake.

`PROCESSOR_STATE_SIZE` stays at 8 on Ampere so the scheduler warp does not grow its ready-queue sweep. Compare with the interleaved A/B harness against the single-expert affinity binary already on `exp/scheduler-throughput`.

## Expert wave under affinity (measured)

Job 61292465 on Perlmutter A100-SXM4-40GB, interleaved A/B, ROUNDS=3 RUNS=16.

Sole-owner waves with the processor always in the span loop were **slower** than single-expert affinity on most shapes (about +4% at the anchor, up to +12% at S=8192). Multi-lane packing hung and was abandoned.

Follow-up: keep the doorbell span encoding, but restore the original one-tile processor path when the high byte is zero so the common case pays no extra barriers. Re-measure.

## Expert wave under affinity — closed (no win)

Interleaved A/B on Perlmutter A100-SXM4-40GB (job 61292465): sole-owner waves were about +4% at the anchor and up to +12% on longer S when the processor always paid the span-loop barriers. Multi-lane packing hung (timeouts). Reverted. The useful residue is the survey: waves need a different binding than today’s per-lane global queues.

## Next: Ampere gated bK=64 / pS=1

Same AB smem footprint as bK=32/pS=2, fewer K trips. Pair against the single-expert affinity binary with the existing A/B harness.

## Ampere gated bK=64 / pS=1 (this PR)

Hardware: Perlmutter A100-SXM4-40GB. Harness: interleaved A/B, ROUNDS=3, WARMUP=2, RUNS=16.
Baseline: single-expert affinity (`exp/scheduler-throughput`, bK=32, pS=2).
Candidate: same scheduler + Ampere gated `bK=64`, `pS=1` (job 61293689).

| Shape | SCHED mean | BK64 mean | Δ |
|---|---:|---:|---:|
| **anchor S=4096 I=1408 E=16 k=6** | **4.467** | **3.813** | **−14.6%** |
| S=4096 I=2048 E=32 k=2 | 1.922 | 1.854 | −3.5% |
| S=4096 I=4096 E=32 k=2 | 3.244 | 3.088 | −4.8% |
| S=4096 I=8192 E=32 k=2 | 5.982 | 5.831 | −2.5% |
| S=8192 I=2048 E=32 k=2 | 3.571 | 3.606 | +1.0% |
| S=16384 I=2048 E=32 k=2 | 7.048 | 7.096 | +0.7% |

Anchor BK64 rounds were 3.812 / 3.813 / 3.815 ms (spread <0.01 ms). Occupancy stayed 2 blocks/SM. Error stayed in band (often lower). AB smem footprint matches bK=32/pS=2; the win is fewer K-loop trips on the fused tile.

## Next: gate-through-gmem + 3 blocks/SM on top of BK64

The gated path kept a second 32 KB C tile in smem as a gate cache. Staging that activation through the output tile in gmem drops Ampere gated AB smem to 32 KB at `bK=64 / pS=1`.

Tried spending the freed budget on larger tiles first:
- `bK=128 / pS=1` (job 61295788): illegal address on every shape.
- `bK=64 / pS=2` (job 61295979): same illegal address — not a usable spend.

Pivot to the resident-warps combo on top of BK64: gmem staging + `__launch_bounds__(threads, 3)` + `PROCESSOR_STATE_SIZE=12` (384 processors). Target occupancy 3 blocks/SM. Compare against the BK64 binary (occupancy 2) with the interleaved A/B harness. Prior resident-warps on bK=32 was mixed; re-check with bK=64.

Perlmutter job **61296242** submitted. Fetch blocked after NERSC sshproxy cert expired (valid to 2026-11-14 16:57 America/New_York); renew with `sshproxy -u rmf253 -o nersc -a -f` then pull `ab/occ3-61296242.out`.
