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
