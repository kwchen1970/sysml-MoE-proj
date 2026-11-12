# MoE-layer campaign

Measurements are BF16 gated MLP on Perlmutter A100s. Latency is the mean of the timed iterations inside one process. `blocks/SM` is `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for the fused kernel.

Anchor shape unless noted: S=4096, H=2048, I=1408, E=16, k=6. Useful work at that point is about 0.425 TFLOP. A100 BF16 peak is 312 TFLOP/s, so the arithmetic roof is about 1.36 ms. The measured forward is about 4.4 ms.

## Closed

| Branch | Hypothesis | Result |
|---|---|---|
| `exp/l2-expert-affinity` | Issuing two hot experts, and anyone else only to fill idle CTAs, cuts weight reloads from a 40 MB L2. | One GPU: 4.412 to 4.083 ms (−7.5%) at the anchor, −5.4% at E=64, −2.8% at E=128, −5.9% at E=32 k=2 I=2048. Four GPUs: −3.1% at E=16, −0.4% at E=64. Occupancy stayed 2. Error stayed in the baseline band. No hang. |
| `exp/deeper-k-pipeline` | A third K-stage hides the copy the way Hopper's 3-stage pipeline does. | Occupancy fell to 1 block/SM (108 blocks). Anchor 4.412 to 6.075 ms (+38%). Every other point was 20–43% slower. |
| `exp/dispatch-cta-share` | Reserving 75% of CTAs for dispatch keeps processors idle. | One GPU moved by under 1%. Four GPUs were 2% and 5% slower. |

## In flight

| Branch | Hypothesis | What would support it |
|---|---|---|
| (no code) scale sweep of baseline vs affinity | The 7% is per-tile weight traffic, so it grows with I and holds its fraction as S grows. A fixed saving that shrinks as a fraction of a longer S is a startup effect. | Anchor repeated 3 times at 32 iterations. Also S=1024 and 16384, I=1024 and 4096, k=2, and the same anchor on 4 GPUs. |
| (no code) Nsight Compute on one forward | The affinity gain is fewer DRAM bytes, or it is something else and the next edit should follow the other limiter. | DRAM throughput and bytes, L2 hit rate, SM and tensor throughput, achieved occupancy. |
| `exp/pipeline-fit-occupancy` | The third stage lost only because occupancy fell. bN=64 keeps 3 stages near 52 KB, so 2 blocks/SM survive and the extra stage can hide the copy. | `blocks/SM` stays 2 and latency drops below the bN=128 2-stage baseline. If occupancy stays 2 and latency does not drop, the copy was not the limiter. |
| `exp/affinity-single-expert` | Two experts are ~34 MB and leave little L2 for activations. When the longer expert alone can fill every free CTA, drop the partner. | Anchor latency below the two-expert 4.083 ms, occupancy still 2, error still in band. A slower result means the partner's tiles were on the critical path. Runs after the scale sweep confirms the 7% is stable. |
