# Overnight optimization log

Anchor, unless a row says otherwise: S=4096, H=2048, I=1408, E=16, k=6, BF16 gated MLP, A100-40GB. Useful work is 0.425 TFLOP. Arithmetic roof on A100 BF16 (312 TFLOP/s) is 1.36 ms.

## Stack that counts

| Step | Latency | Tokens/s | Scope | Status |
|---|---:|---:|---|---|
| Fused kernel, baseline | 4.49 ms | 913k | full forward, 1 GPU, 32 iters | keep |
| Single-expert scheduler | 3.92 ms | 1.04M | full forward. Held on 4 GPU at I=4096: 10.48 to 9.13 ms | keep |
| bK=64, one pipeline stage, gate cache kept | 3.81 ms | 1.07M | full forward, anchor only, 8 iters | keep, small |
| cuBLASLt gate, up, down | 1.75 ms | — | expert math only | ceiling |
| Batched forward, first cut, 1 GPU | 3.26 ms | 1.25M | full forward, anchor, job 61157264. Same-node fused rerun 4.26 ms | keep the GEMM, the memory kernels were slow |
| Same forward, 4 GPU | 5.23 ms | 3.14M aggregate | job 61157434. GEMM 1.63 ms, one-way exchange 0.49 ms | first cut, slow elementwise kernels |
| Row-major forward, 1 GPU | 2.13 ms | 1.92M | jobs 61157994. 199 TFLOP/s, rel 0.46%. Holds on I=4096 (5.21 ms), S=16384 (8.48 ms), E=64 (2.20 ms) | the number to quote |
| Row-major forward, 4 GPU | 3.41 ms | 4.80M aggregate | job 61157995. I=4096 is 6.49 ms vs fused 9.13 ms | unpipelined |
| 4 GPU pipelined exchange | 2.98 ms | 5.50M aggregate | job 61158314, anchor only, rel 0.46% | overlap recovered 0.43 ms |

Two-expert affinity (~4.09 ms) is inside the scheduler step, not an extra gain.

## Dead ends — do not retry

| Attempt | What happened | Why it stays dead |
|---|---|---|
| 3 blocks/SM (`exp/resident-warps`) | Occupancy rose, anchor ~3.90 to 3.87 ms, larger shapes slower, I=4096 error 0.53% | The kernel is not waiting on free warps |
| Deeper K pipeline | 3 stages slower, 4 stages ~1%. Full fused kernel fell to 1 block/SM | Scoreboard stalls are not cured by another stage in this tile |
| bN=64 | ~7.67 ms vs 4.49 ms | Narrower N cuts the tensor tile and still hits the register cap |
| Gate+up fused into one N=2I GEMM | 2.40 ms vs 2.23 ms for two separate cuBLAS GEMMs | Wider N was the slower call |
| Gate result staged through the output tile at bK=64, pS=2 | 3.64 ms and 91.9% error | Numerically wrong. Do not quote the time |
| Always-clear the second hot expert | ~0.5% either way | Hot-set tightening is finished |
| Dispatch CTA share | <1% on 1 GPU, small regression on 4 GPU | Not the limiter |
| cuBLASDx SM80-specific MMA schedule | Not a legal mathdx option. Ampere stays on the generic tile | Do not chase an SM80 mathdx schedule |
| Gate and up on two streams | 3.26 ms either way on the first batched forward | One GEMM already fills the A100 |
| CUDA graph of the forward | 3.25 ms versus 3.26 ms eager | Launch overhead is not the gap |
| NCCL channels forced to 2, 4, or 8 | Pipelined anchor 7.25, 5.64, 3.73 ms versus 2.98 ms at the default | Job 61158524. Leave the channel count alone |

## Where it landed

The single-CTA GEMM was the ceiling. Replacing it with three row-major cuBLASLt GEMMs, plus a vectorized SiLU, is the forward to quote: 2.13 ms on 1 GPU (199 TFLOP/s, rel 0.46%). On 4 GPUs the same GEMM with the next exchange overlapped underneath is 2.98 ms, 5.50M tokens/s. About 0.85 ms of NVLink exchange remains in that step because the copy uses SMs. Multi-node RDMA has not been measured. The code is `csrc/tests/fast_moe.cu`, uncommitted.
