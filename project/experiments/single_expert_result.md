# Single-expert scheduler

Good result, measured on Perlmutter. BF16 gated MLP. Latency is the mean of 32 timed iterations. Token rate is the batch size divided by that latency.

The scheduler keeps the expert with the most ready tiles on the GPU, and admits another expert only when that one cannot fill the free CTAs. The code is on `exp/affinity-single-expert`.

Anchor: S=4096, H=2048, I=1408, E=16, k=6, one A100.

| | Latency | Token rate |
|---|---:|---:|
| Baseline | 4.49 ms | 913k tokens/s |
| Single-expert | 3.92 ms | 1.04M tokens/s |

That is 12.6% lower latency and 14% higher token throughput. Relative error stayed in the baseline band.

The same binary on the other shapes:

| Shape | Baseline | Single-expert | Token rate |
|---|---:|---:|---|
| S=16384 | 21.56 ms | 19.35 ms | 760k → 847k tokens/s |
| I=4096 | 10.37 ms | 8.86 ms | 395k → 462k tokens/s |
| E=64 | 4.80 ms | 4.40 ms | 853k → 931k tokens/s |
| 4 GPUs, I=4096 | 10.48 ms | 9.13 ms | 391k → 449k tokens/s |

On one forward of the anchor, L2 throughput fell from 71% of peak to 51%, and the tensor pipe rose from 33% to 38%. Useful math at the anchor is 0.425 TFLOP, so 3.92 ms is 108 TFLOP/s, 35% of the A100 BF16 peak of 312.
