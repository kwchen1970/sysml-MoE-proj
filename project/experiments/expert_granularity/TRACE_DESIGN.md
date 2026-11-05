# Recorder design and provenance

References were read only:
- FlashMoE revision 9cc0c32443d2a2da6825a68af5ef83060329483b:
  neither infra/trace.cuh nor tests/trace_summary.py exists.
- Mirage revision 4d09bbadc465b50e9bc014095305c76d804da421:
  include/mirage/persistent_kernel/profiler.h,
  python/mirage/mpk/profiler_persistent.py and mpk/static_schedule.py.

Mirage suggested designated-thread event recording, per-block storage, GPU
global timestamps and rejecting incomplete exports. Its static scheduler also
demonstrates 64-bit timestamps. This recorder/exporter is a fresh implementation;
no Mirage code or runtime dependency is vendored. Existing FlashMoE notices
are preserved.

FLASHMOE_PROJECT_TRACE is OFF by default. Hooks disappear from normal builds.
The optional testFlashMoE build uses an inline device Buffer symbol and requires
relocatable device code (already enabled for that target). No public argument
or context layouts changed.

Thread 0 owns each block's bounded event stream. Use integer 64-bit globaltimer
nanoseconds. SM ID is sampled at block entry; this assumes no SM migration.
GPU clocks are not aligned: actor spans and work bounds are computed within
each rank then max-reduced across ranks.

Block windows run from recorded kernel entry to exit. Processors start in
other (including dispatch), transition to wait before mailbox polling, then
other for handshake/task loading, then task for full task execution. A
trace-only completion barrier precedes the next transition. Task type and tile
ID are metadata only on task rows. The OS block records one other interval
covering its lifetime; scheduler/subscriber roles are not separately traced.
This records thread-0 observation at block boundaries, not per-warp utilization.

Overflow increments a dropped counter without writing out of bounds.
Exporter synchronizes after capture, disables recording and copies
bounded records, refuses overwrites and rejects overflow. Intervals partition
each processor window. Zero-length transitions produce no row.
Protocol v2 captures one ordinary launch AFTER warmup/timed repetitions.
The initial correctness launch is unrecorded. Buffers are allocated before
warmup; start() enables recording afterward, followed by an MPI barrier.
Historical trace_1gpu files use the old initial-launch protocol. Recorder stores,
extra completion barriers and register/shared resource changes can perturb it.
Timed trace-build loops have recording disabled, so they do not measure
active-recorder overhead. Do not infer overhead from cold versus warm launches.

CSV columns:
run_id,invocation,ngpu,E,rank,block_id,sm_id,role,state,start_ns,end_ns,
window_start_ns,window_end_ns,dropped_events,task_type,tile_id.
Roles: processor/os. Processor states: other/wait/task.
Expected ranks/blocks are checked against measurements.csv by run.py before
merging. Direct analysis of a subset only supports claims about that subset.

Waiting does not distinguish scheduler delay from missing dependencies or
communication. Task durations include epilogues/transfers/signaling, not only
MMA. The task-work bound is max(sum(task durations)/processor count,
longest task), per rank then max-reduced. Captured actor span is max exit minus
min entry over all recorded block windows per rank, then max-reduced.
The gap includes dispatch, dependencies, imbalance, startup/drain and recording.
It is not additive scheduler overhead. Measured task durations depend on the
traced schedule, so the bound is descriptive, not a prediction of optimization.

Isolated CUDA compilation can run without MathDx/GPU allocation. Full benchmark
build and execution require GPU allocation and accessible dependencies.
No tracing accuracy, active-recorder overhead or speedup is claimed until
those checks have run.
