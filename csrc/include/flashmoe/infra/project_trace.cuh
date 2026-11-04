// Team project recorder. Design inspiration (not copied implementation):
// Mirage persistent_kernel/profiler.h and mpk/static_schedule.py.
// See project/experiments/expert_granularity/TRACE_DESIGN.md for provenance.
#ifndef FLASHMOE_PROJECT_TRACE_CUH
#define FLASHMOE_PROJECT_TRACE_CUH
#ifndef FLASHMOE_PROJECT_TRACE
#define FLASHMOE_PROJECT_TRACE 0
#endif
#if FLASHMOE_PROJECT_TRACE
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
namespace flashmoe::project_trace {
using Stamp = unsigned long long;
enum State : unsigned { other = 0, wait = 1, task = 2 };
struct Event { Stamp begin, end; unsigned state, task_type, tile; };
struct Lane {
  Stamp begin, end, last;
  unsigned count, dropped, sm, state, task_type, tile;
};
struct Buffer { Event* events; Lane* lanes; unsigned capacity; };
// Used only by the C++ benchmark's optional trace build. No public API change.
inline __device__ Buffer active{};
__device__ __forceinline__ Stamp now() {
  Stamp value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
  return value;
}
__device__ __forceinline__ void begin() {
  if (threadIdx.x || !active.lanes) return;
  unsigned sm;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  const auto t = now();
  active.lanes[blockIdx.x] = Lane{t, 0, t, 0, 0, sm, other, 0, 0};
}
// One writer per block; no global atomic counter or polling-loop events.
__device__ __forceinline__ void change(unsigned state, unsigned type = 0, unsigned tile = 0) {
  if (threadIdx.x || !active.lanes) return;
  auto& lane = active.lanes[blockIdx.x];
  const auto t = now();
  if (t > lane.last) {
    if (lane.count < active.capacity) {
      active.events[static_cast<size_t>(blockIdx.x) * active.capacity + lane.count++] =
        Event{lane.last, t, lane.state, lane.task_type, lane.tile};
    } else { ++lane.dropped; }
  }
  lane.last = t; lane.state = state; lane.task_type = type; lane.tile = tile;
}
__device__ __forceinline__ void finish() {
  if (threadIdx.x || !active.lanes) return;
  change(other);
  auto& lane = active.lanes[blockIdx.x];
  lane.end = lane.last;
}
}
#define FLASHMOE_TRACE_BEGIN() flashmoe::project_trace::begin()
#define FLASHMOE_TRACE_STATE(...) flashmoe::project_trace::change(__VA_ARGS__)
#define FLASHMOE_TRACE_FINISH() flashmoe::project_trace::finish()
#else
#define FLASHMOE_TRACE_BEGIN() ((void)0)
#define FLASHMOE_TRACE_STATE(...) ((void)0)
#define FLASHMOE_TRACE_FINISH() ((void)0)
#endif
#endif
