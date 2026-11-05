# Expert granularity experiment

For the current combined study, follow [RUN_1VS4.md](RUN_1VS4.md).
It includes ordered rebuild/submission commands and all comparison plots.

Question: At fixed tokens/GPU and top-k, how does expert count affect latency
and processor waiting? Optional CUDA instrumentation is included in the team repository; reference repositories remain unchanged.

## Baseline

Run commands from the sysml-MoE-proj repository root. Plotting requires Python 3 and Matplotlib. Every figure is exported as a
300-DPI PNG plus SVG and PDF. CSV files remain unchanged.

Your existing environment supplies Matplotlib:

```bash
conda activate /global/homes/k/kc994/.conda/envs/sysml
```

Alternatively, call that environment's Python executable directly. Use the
project CUDA module setup for compilation; the plotting environment is separate.

1. Set MATHDX_ROOT to an accessible MathDx installation. The new env.sh defaults
   to your home, rather than Jo's installation. Set REPO if submitting elsewhere.
2. Build on a GPU allocation (CMake queries the GPU), using your usual account:

```bash
salloc -A YOUR_ACCOUNT -C gpu -q interactive -N 1 --gpus-per-node=4 -t 00:30:00
source project/scripts/expert_granularity/env.sh
cmake -S csrc -B csrc/cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build csrc/cmake-build-release --target testFlashMoE --parallel 8
sbatch -A YOUR_ACCOUNT project/scripts/expert_granularity/baseline.sbatch --gpus 1 --repetitions 5
```

After validating the single-GPU results:

```bash
sbatch -A YOUR_ACCOUNT project/scripts/expert_granularity/baseline.sbatch --gpus 1 4 --repetitions 5
```

Default configurations: S=4096 per GPU, H=2048, I=1408, E=16/64/128,
k=6; graph_launches=8, warmup=128, runs=128. Five independent process
invocations per configuration, randomized configuration order. The order seed
is recorded; upstream benchmark tensor/routing seeds remain uncontrolled.
Explicit ngpu, run_id, invocation and repetition accompany every rank result.
Any launcher failure or missing/duplicate rank stops the job; full logs survive.
Results default to $SCRATCH/moe/expert_granularity/<timestamp-jobid>/.
--binary and --output can override paths; output directory must not exist.

```bash
python3 project/analysis/expert_granularity/summarize.py   /YOUR/SCRATCH/RUN/measurements.csv   --output project/results/expert_granularity/YOUR_RUN
```

To plot Jo's existing data without running a job:

```bash
python3 project/analysis/expert_granularity/summarize.py   --output project/results/expert_granularity/jo_reference
```

Outputs: summary.csv, latency_vs_experts.{png,svg,pdf}, throughput_vs_experts.{png,svg,pdf}.
New latency = maximum rank's average iteration time per invocation, summarized
across invocations. This is an approximation to distributed latency, not a
synchronized per-iteration global maximum. Error bars are sample standard
deviations. Jo's results retain her pooled-rank/repetition statistic and are
labeled separately; repeated sweep points remain pooled. Throughput is nominal
input tokens/second, not accepted expert assignments; investigate capacity
clipping before claiming equivalent useful work. Routing/gating is excluded
from timing. Nonzero error_pct means output elements failed the benchmark's
isclose tolerances; investigate before treating a result as validated.

## Device tracing (implemented; GPU validation pending)

The reference FlashMoE checkout lacks both named tracer files. Our optional
recorder uses Mirage-inspired designated-writer/per-block-buffer ideas.
See TRACE_DESIGN.md for provenance and interpretation.

Build a SEPARATE trace binary inside a GPU allocation:

```bash
source project/scripts/expert_granularity/env.sh
cmake -S csrc -B csrc/cmake-build-trace -DCMAKE_BUILD_TYPE=Release -DFLASHMOE_PROJECT_TRACE=ON
cmake --build csrc/cmake-build-trace --target testFlashMoE --parallel 8
sbatch -A YOUR_ACCOUNT project/scripts/expert_granularity/trace.sbatch --gpus 1 --repetitions 5
```

Begin with one GPU; later use --gpus 1 4. Jobs do not build dependencies.
Capture now records one ordinary launch after warmup and timed repeats.
The initial correctness launch and the timed repeats are unrecorded.
Old trace_1gpu outputs are initial-launch captures and must remain separately labeled.
Trace-build timed results still reflect the instrumentation resource footprint;
keep regular and trace builds/results separate. No active-recorder slowdown
measurement is currently implemented.

The job saves per-invocation/rank events and merges them to events.csv only
after checking block coverage and metadata against the benchmark rows.
Default capacity is 16384 intervals/block (about 108 MiB for 216 blocks).
Set FLASHMOE_TRACE_CAPACITY before submission if overflow occurs.
Dropped intervals make capture and analysis fail. Raw outputs stay on scratch.

```bash
python3 project/analysis/expert_granularity/waiting.py /SCRATCH/RUN/events.csv --output project/results/expert_granularity/YOUR_RUN
python3 project/analysis/expert_granularity/timeline.py /SCRATCH/RUN/events.csv --run-id RUN_ID_FROM_METADATA --invocation 0 --rank 0 --output project/results/expert_granularity/YOUR_RUN/timeline.svg
```

Outputs: waiting.csv, waiting_vs_experts.svg, work_bound.csv,
span_vs_work_bound.svg, task_durations.csv, and the selected timeline.
Invocation commands record expert counts; their order is randomized.
Use --duration-us with the same value for timeline panels to compare their
time scales. The OS lane is its lifetime, not scheduler utilization.
Task intervals include epilogues, transfers, signaling and a trace-only
completion barrier. Task types: 0=GEMM0, 1=GEMM1, 2=combine.

Before interpreting results, validate correctness, overflow, all-block coverage,
task counts and trace perturbation on GPUs. Waiting and the work-bound gap
are NOT measurements of scheduler cost by themselves.

All timeline, waiting and work-bound figures also export PNG, SVG and PDF
with the same filename stem. Existing plotting commands still work; an
--output timeline.svg request now also writes timeline.png and timeline.pdf.

## Single-node NVSHMEM GPU visibility

Use `launch_single_node.sh` for both diagnostics and baseline/trace runs.
Four ranks share the allocated GPU set (`--gpus-per-node=4 --gpu-bind=none`);
do not combine this with `--gpus-per-task=1`, manually narrow
CUDA_VISIBLE_DEVICES, or unset Slurm's allocation-provided visibility.
The 1-GPU path retains `--gpus-per-task=1 --gpu-bind=closest`.

The benchmark initializes MPI, obtains the shared-memory local rank, selects
that visible ordinal BEFORE explicit MPI-communicator NVSHMEM initialization,
and checks distinct physical UUIDs and CUDA P2P capability. Insufficient
visibility, duplicate UUIDs or inaccessible peers abort the whole MPI job.
Diagnostics print visibility, ordinal, UUID, PCI bus, peer capability and both
NVSHMEM host-init and deferred device-state synchronization stages.
The helper is intentionally limited to one node (current experiment scope).

NVSHMEM_REMOTE_TRANSPORT=none disables off-node transports; intra-node CUDA
P2P/NVLink remains required and enabled. GDRCopy/NCCL settings are unchanged;
MPI here exchanges host metadata, so MPICH_GPU_SUPPORT_ENABLED=0 is retained.
No remote-transport fallback is added to hide missing peer visibility.

From the repository root, submit manually (use your authorized account):

```bash
sbatch -A m4999_g -C 'gpu&hbm40g' gpu4_flashmoe_test.sbatch
```

The diagnostic incrementally rebuilds testFlashMoE inside its allocation
(maximum 240 seconds), then runs a 180-second-bounded 4-rank test.
The job has a 10-minute limit. A rebuild failure/timeout prevents the test;
it never silently runs a stale binary. REBUILD=0 is available only if you
have already rebuilt the changed source. Rebuild the trace binary separately
before trace experiments. Do not interpret diagnostic timing as a baseline.

CPU-only launch checks (srun is stubbed; no jobs submitted):
`python project/tests/expert_granularity/test_launch.py`.
Real GPU/P2P/NVSHMEM success remains unverified until the diagnostic runs.

References:
- https://docs.nvidia.com/nvshmem/api/latest/faq.html
- https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/performance.html
