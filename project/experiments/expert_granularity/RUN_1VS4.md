# Run the 1-versus-4-GPU granularity study (protocol v2)

Question: how do expert count and GPU count change processor waiting, task
structure and measured MoE latency? This extends the prior baseline-only study.

All commands below are run manually from the team repository root. The scripts
do not submit other jobs. Nothing should be run from the reference repositories.

## Build, then collect baseline, then collect warmed traces

Use your authorized account (examples use m4999_g). All three jobs request
A100 40GB nodes. Baseline compares 1 and 4 GPUs within the same allocation.
The trace job uses a separate allocation with the same hardware constraint.

```bash
cd ~/cs5470/final_proj/sysml-MoE-proj
STUDY="$SCRATCH/moe/expert_granularity/study-$(date -u +%Y%m%dT%H%M%S)"

BUILD_JOB=$(sbatch --parsable -A m4999_g -C 'gpu&hbm40g' project/scripts/expert_granularity/build.sbatch)

BASE_JOB=$(sbatch --parsable -A m4999_g -C 'gpu&hbm40g' --dependency=afterok:$BUILD_JOB project/scripts/expert_granularity/baseline.sbatch --gpus 1 4 --experts 16 64 128 --repetitions 5 --output "$STUDY/baseline")

TRACE_JOB=$(sbatch --parsable -A m4999_g -C 'gpu&hbm40g' --dependency=afterok:$BASE_JOB project/scripts/expert_granularity/trace.sbatch --gpus 1 4 --experts 16 64 128 --repetitions 5 --output "$STUDY/trace")

echo "Study: $STUDY"
echo "Build=$BUILD_JOB Baseline=$BASE_JOB Trace=$TRACE_JOB"
squeue -u "$USER"
```

afterok prevents measurement jobs from starting if their prerequisite fails.
If a prerequisite fails, inspect its log; dependent jobs may stay pending.
Do not resubmit with an existing output directory; choose a fresh study path.

Build requests 20 minutes. Baseline and trace each have a one-hour limit.
Based on earlier runs, budget roughly 8-15 minutes per sweep after allocation,
plus rebuild time and queue delays; first runs of the revised trace are not
yet timed/validated on GPUs. Five repetitions × three expert counts × two GPU
counts = 30 independent invocations per sweep, 75 rank measurement rows.

## Generate all research figures after both sweeps finish

```bash
conda activate /global/homes/k/kc994/.conda/envs/sysml
FIGURES=project/results/expert_granularity/study_1vs4_v2
python3 project/analysis/expert_granularity/summarize.py "$STUDY/baseline/measurements.csv" --no-reference --output "$FIGURES/baseline"
python3 project/analysis/expert_granularity/waiting.py "$STUDY/trace/events.csv" --output "$FIGURES/trace"
python3 project/analysis/expert_granularity/compare_traces.py "$STUDY/trace/events.csv" --output "$FIGURES/trace"
```

If opening a new terminal, restore STUDY to the printed study directory.
Plots are PNG, SVG and PDF:
- baseline/latency_vs_experts and throughput_vs_experts: warmed uninstrumented performance.
- trace/waiting_vs_experts: waiting fractions with one curve per GPU count.
- trace/block_time_breakdown: paired task/wait/other bars for each expert count.
- trace/task_count_comparison: tasks PER GPU per captured launch, by task type.
- trace/task_duration_comparison: task-weighted means within each invocation.
- trace/per_rank_waiting: four rank-local curves, each with variability over repetitions.
- trace/span_vs_work_bound: descriptive work bound; gap is not scheduler overhead.

Task counts are normalized per GPU because total input tokens grow with GPU
count. Cross-rank task durations are weighted by task count, then independent
invocations are averaged. Ranks are not treated as independent repetitions.
Comparison script rejects missing ranks; run.py also verifies complete block
coverage before merging events. Do not mix old initial-launch traces with v2.
The older plot_block_breakdown.py and plot_task_durations.py target the old
one-GPU directory; use compare_traces.py for this study instead.

## What v2 capture measures

The initial correctness execution is unrecorded. Trace buffers are allocated
before warmup, but recording is disabled during all warmup/timed graph replays.
After measurement, recording is enabled, ranks synchronize, and one ordinary
(non-graph) MoE execution is captured and exported. Trace scope is saved in
metadata.json. Recording is outside the reported latency. Trace-build timed
measurements are not the uninstrumented baseline.

This is post-warmup instrumented execution, not an exact replay of a timed
CUDA graph. Recorder stores/barriers/storage and pre-capture synchronization
can perturb execution. Active-recorder slowdown is not yet measured. Task
activity is not tensor-core utilization; mailbox waiting does not distinguish
scheduler cost from unavailable dependencies/communication. Output mismatches
still require investigation before claiming correctness-validated speedups.
