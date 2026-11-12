/******************************************************************************
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

//
// Created by Osayamen on 11/17/24.
//

#ifndef SCHEDULER_CUH
#define SCHEDULER_CUH

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/cmath>
#include <cute/numeric/integral_constant.hpp>
#include <cutlass/array.h>

#include "infra/affinity.cuh"
#include "infra/constants.cuh"
#include "infra/bitset.cuh"
#include "infra/checks.cuh"
#include "infra/dq.cuh"
#include "infra/task.cuh"
#include "infra/tq.cuh"

namespace flashmoe::scheduler
{
  __device__ __forceinline__
  auto circularIdx(const uint& cursor, const cuda::fast_mod_div<uint>& len) {
    //https://nvidia.github.io/cccl/libcudacxx/extended_api/math/fast_mod_div.html#libcudacxx-extended-api-math-fast-mod-div
#if defined(__CUDA_ARCH__)
    __builtin_assume(cursor != cuda::std::numeric_limits<uint>::max());
#endif
    return cursor % len;
  }
  using WarpScan = cub::WarpScan<uint>;

  template <
    DQType dqt = DQType::stride,
    int nQ = 0
  >
  __device__ __forceinline__
  void schedule(const cuda::fast_mod_div<uint>& processors_v, const uint& cSetB,
                const uint& canSchedule, const uint& qIdx, uint& lRQIdx,
                const uint& gRQIdx, uint* __restrict__ const& rQ,
                TQSignal* __restrict__ const& pDB) {
    cutlass::Array<uint, WORK_SET_SIZE> wSet{};
    static_assert(sizeof(TQSignal) == sizeof(uint64_t) && alignof(TQSignal) == alignof(uint64_t));
    auto sig = TQSignal{0U, 0U};
    for (uint k = 0; k < cSetB; ++k) {
      #pragma unroll
      for (uint l = 0; l < wSet.size(); ++l) {
        wSet[l] = rQ[circularIdx(gRQIdx + lRQIdx++, processors_v)];
      }
      #pragma unroll
      for (uint l = 0; l < wSet.size(); ++l) {
        // signal processor
        auto* __restrict__ pdbAddr = reinterpret_cast<uint64_t*>(pDB + wSet[l]);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> pdb{*pdbAddr};
        sig.encodeSig(DQ::next<dqt, nQ>(qIdx, k * wSet.size() + l));
        pdb.store(cuda::std::bit_cast<uint64_t>(sig), cuda::memory_order_release);
      }
    }
    // Residual scheduling
    const uint residue = canSchedule - cSetB * wSet.size();
    #pragma unroll
    for (uint l = 0; l < wSet.size(); ++l) {
      if (l < residue) {
        wSet[l] = rQ[circularIdx(gRQIdx + lRQIdx++, processors_v)];
      }
    }
    #pragma unroll
    for (uint l = 0; l < wSet.size(); ++l) {
      if (l < residue) {
        auto* __restrict__ pdbAddr = reinterpret_cast<uint64_t*>(pDB + wSet[l]);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> pdb{*pdbAddr};
        sig.encodeSig(DQ::next<dqt, nQ>(qIdx, cSetB * wSet.size() + l));
        pdb.store(cuda::std::bit_cast<uint64_t>(sig), cuda::memory_order_release);
      }
    }
  }

  // One processor runs `span` consecutive tasks from a block queue.
  // The ready-queue cursor advances once per processor, not once per tile.
  template <int nQ = 0>
  __device__ __forceinline__
  void scheduleWave(const cuda::fast_mod_div<uint>& processors_v, const uint canSchedule,
                    const uint span, const uint qIdx, uint& door, const uint gRQIdx,
                    uint* __restrict__ const& rQ, TQSignal* __restrict__ const& pDB) {
    uint issued = 0;
    while (issued < canSchedule) {
      const uint n = umin(span, canSchedule - issued);
      const uint proc = rQ[circularIdx(gRQIdx + door, processors_v)];
      ++door;
      const uint index = DQ::next<DQType::block, nQ>(qIdx, issued);
      TQSignal sig{0U, 0U};
      sig.signal = (n << kSignalIndexBits) | ((index + 1u) & kSignalIndexMask);
      auto* __restrict__ pdbAddr = reinterpret_cast<uint64_t*>(pDB + proc);
      cuda::atomic_ref<uint64_t, cuda::thread_scope_device> pdb{*pdbAddr};
      pdb.store(cuda::std::bit_cast<uint64_t>(sig), cuda::memory_order_release);
      issued += n;
    }
  }

  template <
    int processor_state_size,
    int subscriberCount,
    int sL,
    int schedulerCount,
    typename TQState
  >
    requires (schedulerCount == SCHEDULER_COUNT && isRegisterV<TQState>)
  __device__ __forceinline__
  void schedulerLoop(TQState& tqState,
                     const uint& processors,
                     const cuda::fast_mod_div<uint>& processors_v,
                     const unsigned int& tilesN1,
                     const unsigned int& tQOffset,
                     const unsigned int& gTbO,
                     uint& lTt, uint& processorTally,
                     uint& gRQIdx, uint& scheduled,
                     unsigned int* __restrict__ const& sQ,
                     uint* __restrict__ const& rQ,
                     TQSignal* __restrict__ const& pDB,
                     const bool& isMedley = false) {
    __shared__ WarpScan::TempStorage wSt;
    uint queueSlot;
    uint taskTally;
    // things are about to get warped :)
    // Aggregate tally across the warp
    __syncwarp();
    WarpScan(wSt).InclusiveSum(lTt, queueSlot, taskTally);
    queueSlot -= lTt;
    auto prefixTaskSum = 0U;
    uint issuedDoors = 0U;
    while (taskTally) {
      // Find processors if we are not currently aware of any
      while (!processorTally) {
        cutlass::Array<uint, processor_state_size> sQState{};
        // sweep sQ to identify ready processes
        uint lPt = 0U; // local processor tally
        #pragma unroll
        for (int j = 0; j < sQState.size(); ++j) {
          const auto idx = j * schedulerCount + threadIdx.x;
          if (idx < processors) {
            const auto readiness = atomicExch(sQ + (j * schedulerCount + threadIdx.x),
                                              observed) == ready;
            lPt += readiness;
            sQState[j] = readiness;
          }
        }
        uint startIdx;
        // Aggregate tally across the warp
        __syncwarp();
        WarpScan(wSt).InclusiveSum(lPt, startIdx, processorTally);
        startIdx -= lPt;
        // write to rQ
        const auto qSIdx = gRQIdx + prefixTaskSum;
        #pragma unroll
        for (uint j = 0; j < sQState.size(); ++j) {
          const auto idx = j * schedulerCount + threadIdx.x;
          if (idx < processors && sQState[j]) {
            // write ready process pid to rQ
            rQ[circularIdx((qSIdx + startIdx++), processors_v)] = idx;
          }
        }
        if (processorTally) {
          // Below ensures writes to rQ in shared memory are visible warp-wide before consumption
          __syncwarp();
        }
      }
      // One megakernel stays resident. When a single lane owns every ready tile
      // and there are enough free processors to give each of them several tiles,
      // wake each processor once for a run of one expert's tiles. The processor
      // walks that run before it returns to the doorbell, so the expert weight
      // stays hot and the OS block does not pay a round trip per 128x128 tile.
      uint headTasks = 0;
      if constexpr (sL > 0) {
        #pragma unroll
        for (uint j = 0; j < sL; ++j) {
          headTasks += tqState[j].tasks;
        }
      }
      // Wave only the block queues. A lane that still holds subscriber-queue
      // tasks keeps the original one-doorbell-per-tile path.
      const bool soleOwner = !isMedley && prefixTaskSum == 0 && taskTally >= kExpertWave &&
        __any_sync(0xffffffff, queueSlot == 0 && lTt >= taskTally && headTasks == 0);
      const uint wavePops = (taskTally + kExpertWave - 1) / kExpertWave;
      const bool wave = soleOwner && wavePops <= processorTally;
      // schedule tasks
      const auto tasks = wave ? taskTally : cute::min(processorTally, taskTally);
      prefixTaskSum += tasks;
      scheduled += tasks;
      taskTally -= tasks;
      uint waveDoor = 0;
      uint plainDoors = 0;
      // these will get scheduled now
      if (lTt > 0 && queueSlot < prefixTaskSum) {
        auto tasksToSchedule = umin(lTt, prefixTaskSum - queueSlot);
        lTt -= tasksToSchedule;
        if (isMedley) {
          if constexpr (sL > 0) {
            #pragma unroll
            for (uint j = 0; j < sL; ++j) {
              if (tqState[j].tasks > 0 && tasksToSchedule) {
                const auto canSchedule = cute::min(tasksToSchedule, tqState[j].tasks);
                const auto qIdx = DQ::next<DQType::stride, subscriberCount>(
                  j * schedulerCount + threadIdx.x,tqState[j].tQTail);
                tasksToSchedule -= canSchedule;
                tqState[j].tasks -= canSchedule;
                // have to increment tails as we will revisit this queue later on
                tqState[j].tQTail += canSchedule;
                const auto cSetB = canSchedule / WORK_SET_SIZE;
                schedule<DQType::stride, subscriberCount>(processors_v, cSetB, canSchedule, qIdx,
                                                          queueSlot, gRQIdx, rQ, pDB);
                plainDoors += canSchedule;
              }
            }
          }
        }
        #pragma unroll
        for (uint j = sL; j < tqState.size(); ++j) {
          if (tqState[j].tasks && tasksToSchedule) {
            const auto canSchedule = cute::min(tasksToSchedule, tqState[j].tasks);
            const auto qHead = (schedulerCount * (gTbO + (j - sL)) + threadIdx.x) * tilesN1 + tqState[j].tQTail;
            const auto qIdx = tQOffset + qHead;
            const bool oneQueue = canSchedule == tasksToSchedule;
            tasksToSchedule -= canSchedule;
            tqState[j].tasks -= canSchedule;
            // checkpoint state in case of partial scheduling
            tqState[j].tQTail += canSchedule;
            const auto cSetB = canSchedule / WORK_SET_SIZE;
            if (wave && queueSlot == 0 && oneQueue) {
              scheduleWave(processors_v, canSchedule, kExpertWave, qIdx, waveDoor, gRQIdx, rQ, pDB);
            }
            else {
              schedule<DQType::block>(processors_v, cSetB, canSchedule,
                                      qIdx, queueSlot, gRQIdx, rQ, pDB);
              plainDoors += canSchedule;
            }
          }
        }
      }
      const uint doors = __reduce_add_sync(0xffffffff, waveDoor + plainDoors);
      processorTally -= doors;
      issuedDoors += doors;
    }
    // clear checkpoints
    #pragma unroll
    for (uint j = sL; j < tqState.size(); ++j) {
      tqState[j].tQTail = 0;
    }
    // A wave consumes one ready-queue slot per processor. prefixTaskSum
    // still counts tiles, so the cursor follows issued doorbells.
    gRQIdx = circularIdx((gRQIdx + issuedDoors), processors_v);
  }

  template <int processor_state_size, unsigned int schedulerCount>
  /// Schedule Processor interrupts
  __device__ __forceinline__
  void sPI(unsigned int* __restrict__ const& rQ,
           uint* __restrict__ const& sQ,
           TQSignal* __restrict__ const& pDB,
           uint& gRQIdx,
           uint* __restrict__ const& scratch, // pre-filled with 1
           const uint& processors,
           const cuda::fast_mod_div<uint>& processors_v,
           const uint& processorTally) {
    __shared__ WarpScan::TempStorage wSt;
    cutlass::Array<uint, processor_state_size> sQState{};
    /// read through the ready queue first
    constexpr auto sig = TQSignal{0U, 1U}; // set interrupt to 1
    const auto tS = processorTally / schedulerCount + (threadIdx.x < processorTally % schedulerCount);
    const auto gRO = gRQIdx + (threadIdx.x * (processorTally / schedulerCount) +
      cute::min(threadIdx.x, processorTally % schedulerCount));
    // index can only wrap around once
    gRQIdx = circularIdx(gRO, processors_v);
    #pragma unroll
    for (uint i = 0; i < sQState.size(); ++i) {
      if (i < tS) {
        // shared -> registers
        sQState[i] = rQ[circularIdx(gRQIdx + i, processors_v)];
      }
    }
    #pragma unroll
    for (uint i = 0; i < sQState.size(); ++i) {
      if (i < tS) {
        // notify interrupts
        const auto pid = sQState[i];
        auto* __restrict__ db_p = reinterpret_cast<uint64_t*>(pDB + pid);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> pdb{*db_p};
        pdb.store(cuda::std::bit_cast<uint64_t>(sig), cuda::memory_order_release);
        scratch[pid] = 0U;
      }
    }
    __syncwarp();
    // Consolidate findings and populate the ready queue
    uint uI = 0U;
    // shared -> registers
    #pragma unroll
    for (uint i = 0; i < sQState.size(); ++i) {
      const auto idx = i * schedulerCount + threadIdx.x;
      if (idx < processors) {
        sQState[i] = scratch[idx];
      }
    }

    #pragma unroll
    for (uint i = 0; i < sQState.size(); ++i) {
      if ((i * schedulerCount + threadIdx.x) < processors) {
        uI += sQState[i];
      }
    }

    uint startIdx;
    uint pending;
    WarpScan(wSt).InclusiveSum(uI, startIdx, pending);
    startIdx -= uI;
    // enqueue all pending processes we discovered into the rQ
    #pragma unroll
    for (uint i = 0; i < sQState.size(); ++i) {
      const auto idx = i * schedulerCount + threadIdx.x;
      if (idx < processors && sQState[i]) {
        rQ[startIdx++] = idx;
      }
    }
    __syncwarp();
    auto remaining = pending / schedulerCount + (threadIdx.x < pending % schedulerCount);
    cuda::std::array<uint, sQState.size()> pids{};
    // read from rQ to registers
    #pragma unroll
    for (uint i = 0; i < sQState.size(); ++i) {
      const auto idx = i * schedulerCount + threadIdx.x;
      if (idx < pending) {
        sQState[i] = 1;
        pids[i] = rQ[idx];
      }
      else {
        sQState[i] = 0;
      }
    }

    while (remaining) {
      #pragma unroll
      for (uint j = 0; j < sQState.size(); ++j) {
        if (sQState[j]) {
          const auto pid = pids[j];
          const auto isReady = atomicExch(sQ + pid, observed) == ready;
          sQState[j] = !isReady;
          if (isReady) {
            // interrupt processor
            remaining -= 1;
            auto* __restrict__ db_p = reinterpret_cast<uint64_t*>(pDB + pid);
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> pdb{*db_p};
            pdb.store(cuda::std::bit_cast<uint64_t>(sig), cuda::memory_order_release);
          }
        }
      }
    }
  }

  // Issue at most one wave of free CTAs, preferring two experts whose weights
  // fit together in L2. Global (GEMM1) queues are consumed only when this
  // wave actually takes them, so a deferred queue stays visible.
  template <
    unsigned int subscribers,
    int processor_state_size
  >
  __device__ __forceinline__
  void runAffinity(cutlass::Array<TQState, (subscribers / SCHEDULER_COUNT) + QUEUE_STATE_SIZE>& tqState,
                   const uint& processors,
                   const cuda::fast_mod_div<uint>& processors_v,
                   const unsigned int& tilesN1,
                   const unsigned int& sO,
                   const unsigned int& gtQCL,
                   BitSet* __restrict__ const& bitSet,
                   unsigned int* __restrict__ const& tQHeads,
                   unsigned int* __restrict__ const& gtQHeads,
                   unsigned int* __restrict__ const& taskBound,
                   uint* __restrict__ const& rQ,
                   unsigned int* __restrict__ const& sQ,
                   TQSignal* __restrict__ const& pDB,
                   Task* __restrict__ const& tQ,
                   const uint& nLx,
                   const uint& ecTilesM,
                   const uint& dT,
                   uint& gRQIdx,
                   uint& processorTally,
                   uint& scheduled,
                   uint& tTB) {
    constexpr auto schedulerCount = SCHEDULER_COUNT;
    constexpr int sL = static_cast<int>(subscribers / schedulerCount);
    constexpr auto dQL = QUEUE_STATE_SIZE;
    constexpr auto bSw = sizeof(uint) * 8U;
    constexpr int kLocal = sL + dQL;
    constexpr int subCount = static_cast<int>(subscribers);
    constexpr int schedCount = schedulerCount;
    static_assert(kLocal <= 24);
    __shared__ unsigned affHist[256];
    __shared__ unsigned affChoice[2];
    __shared__ unsigned affLaneHot[32];
    __shared__ unsigned affLaneCold[32];
    __shared__ unsigned affHotBudget[32];
    __shared__ unsigned affColdBudget[32];

    unsigned sticky0 = kNoExpert;
    unsigned sticky1 = kNoExpert;
    uint window = 0;
    const uint nWindows = dT + 1;
    while (scheduled < tTB) {
      LocalQueue local[kLocal];
      unsigned take[kLocal];
      #pragma unroll
      for (int i = 0; i < kLocal; ++i) {
        local[i] = LocalQueue{kNoExpert, 0u, false};
        take[i] = 0u;
        tqState[i].tasks = 0u;
      }
      #pragma unroll
      for (uint i = 0; i < sL; ++i) {
        auto* __restrict__ tqHp = tQHeads + (i * schedulerCount + threadIdx.x);
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> tqh{*tqHp};
        const unsigned pending = tqh.load(cuda::memory_order_acquire) - tqState[i].tQTail;
        if (pending) {
          const uint tid = i * schedulerCount + threadIdx.x;
          const uint index = DQ::next<DQType::stride, subCount>(tid, tqState[i].tQTail);
          local[i] = LocalQueue{static_cast<unsigned>(tQ[index].localExpertIdx()), pending, false};
        }
      }

      const uint gw = window % nWindows;
      const uint sBIdx = threadIdx.x + (gw * dQL / bSw) * schedulerCount;
      auto sBS = bitSet[sBIdx];
      #pragma unroll
      for (uint pJ = 0; pJ < dQL; ++pJ) {
        const uint qIdx = schedulerCount * (dQL * gw + pJ) + threadIdx.x;
        const uint bIdx = (gw * dQL + pJ) % bSw;
        unsigned tasks = 0u;
        unsigned expert = kNoExpert;
        if (qIdx < gtQCL && !sBS.get(bIdx)) {
          cuda::atomic_ref<unsigned int, cuda::thread_scope_device> gtqH{*(gtQHeads + qIdx)};
          tasks = gtqH.load(cuda::memory_order_acquire);
          if (tasks) {
            const unsigned group = qIdx / ecTilesM;
            expert = group % nLx;
          }
        }
        local[sL + pJ] = LocalQueue{expert, tasks, true};
        tqState[sL + pJ].tQTail = 0u;
      }

      if (threadIdx.x == 0) {
        for (unsigned e = 0; e < nLx; ++e) {
          affHist[e] = 0u;
        }
      }
      __syncwarp();
      #pragma unroll
      for (int i = 0; i < kLocal; ++i) {
        if (local[i].tasks && local[i].expert < nLx) {
          atomicAdd(affHist + local[i].expert, local[i].tasks);
        }
      }
      __syncwarp();
      if (threadIdx.x == 0) {
        unsigned hot0 = kNoExpert;
        unsigned hot1 = kNoExpert;
        chooseHotExperts(affHist, nLx, sticky0, sticky1, hot0, hot1);
        const unsigned slotsNow = processorTally == 0 ? processors : processorTally;
        narrowToLongPole(affHist, hot0, hot1, slotsNow);
        affChoice[0] = hot0;
        affChoice[1] = hot1;
      }
      __syncwarp();
      const unsigned hot0 = affChoice[0];
      const unsigned hot1 = affChoice[1];
      unsigned hotLocal = 0u;
      unsigned coldLocal = 0u;
      #pragma unroll
      for (int i = 0; i < kLocal; ++i) {
        if (!local[i].tasks) {
          continue;
        }
        if (isHotExpert(local[i].expert, hot0, hot1)) {
          hotLocal += local[i].tasks;
        } else {
          coldLocal += local[i].tasks;
        }
      }
      affLaneHot[threadIdx.x] = hotLocal;
      affLaneCold[threadIdx.x] = coldLocal;
      __syncwarp();
      const unsigned slots = processorTally == 0 ? processors : processorTally;
      if (threadIdx.x == 0) {
        unsigned hotTotal = 0u;
        for (unsigned e = 0; e < nLx; ++e) {
          if (isHotExpert(e, hot0, hot1)) {
            hotTotal += affHist[e];
          }
        }
        const bool admitCold = hotTotal < slots;
        clampToSlots(affLaneHot, schedulerCount, admitCold ? 0xffffffffu : slots, affHotBudget);
        const unsigned coldSlots = admitCold && slots > hotTotal ? slots - hotTotal : 0u;
        clampToSlots(affLaneCold, schedulerCount, coldSlots, affColdBudget);
      }
      __syncwarp();
      assignQueues(local, static_cast<unsigned>(kLocal), hot0, hot1,
                   affHotBudget[threadIdx.x], affColdBudget[threadIdx.x], take);

      bool dirty = false;
      #pragma unroll
      for (uint pJ = 0; pJ < dQL; ++pJ) {
        const uint slot = sL + pJ;
        if (take[slot] == 0 || local[slot].tasks == 0) {
          continue;
        }
        const uint qIdx = schedulerCount * (dQL * gw + pJ) + threadIdx.x;
        const uint bIdx = (gw * dQL + pJ) % bSw;
        sBS.set(bIdx);
        dirty = true;
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> gtqH{*(gtQHeads + qIdx)};
        gtqH.store(tQHeadGroundState, cuda::memory_order_relaxed);
        tqState[slot].tasks = local[slot].tasks;
        tqState[slot].tQTail = 0u;
      }
      if (dirty) {
        bitSet[sBIdx] = sBS;
      }
      #pragma unroll
      for (uint i = 0; i < sL; ++i) {
        tqState[i].tasks = take[i];
      }
      uint lTt = 0u;
      #pragma unroll
      for (int i = 0; i < kLocal; ++i) {
        lTt += tqState[i].tasks;
      }
      schedulerLoop<processor_state_size, subCount, sL, schedCount>(
        tqState, processors, processors_v, tilesN1, sO, dQL * gw, lTt,
        processorTally, gRQIdx, scheduled, sQ, rQ, pDB, true);

      sticky0 = hot0;
      sticky1 = hot1;
      window = gw + 1;
      if (!threadIdx.x) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> tb{*taskBound};
        tTB = tb.load(cuda::memory_order_relaxed);
      }
      __syncwarp();
      tTB = __shfl_sync(0xffffffff, tTB, 0);
    }
  }

  template <
    unsigned int subscribers,
    int processor_state_size = PROCESSOR_STATE_SIZE
  >
  __device__ __forceinline__
  void start(uint* __restrict__ const& interruptScratch,
             BitSet* __restrict__ const& bitSet,
             const uint& processors,
             const cuda::fast_mod_div<uint>& processors_v,
             const unsigned int& tilesN1,
             const unsigned int& sO,
             const unsigned int& gtQCL,
             unsigned int* __restrict__ const& sInterrupts,
             unsigned int* __restrict__ const& tQHeads, // shared
             unsigned int* __restrict__ const& gtQHeads, // global
             unsigned int* __restrict__ const& taskBound, // shared
             unsigned int* __restrict__ const& rQ, // shared
             unsigned int* __restrict__ const& sQ, // global
             TQSignal* __restrict__ const& pDB,
             Task* __restrict__ const tQ = nullptr,
             const uint nLx = 0,
             const uint ecTilesM = 0) {
    static_assert(processor_state_size <= PROCESSOR_STATE_SIZE);
    static_assert(sizeof(TQSignal) == sizeof(uint64_t) && alignof(TQSignal) == alignof(uint64_t));
    // Assumptions are below, will assert them host-side
    // PROCESSOR_STATE_SIZE * schedulerCount >= processors
    uint scheduled = 0U;
    constexpr auto schedulerCount = SCHEDULER_COUNT; // number of scheduler threads
    static_assert(subscribers % schedulerCount == 0);
    constexpr auto sL = subscribers / schedulerCount;
    // initialize register buffers
    constexpr auto dQL = QUEUE_STATE_SIZE;
    constexpr auto bSw = sizeof(uint) * 8U;
    static_assert(dQL <= bSw);
    cutlass::Array<TQState, dQL + sL> tqState{};
    tqState.fill({0U, 0U});
    const uint dT = gtQCL / (schedulerCount * dQL);

    uint gRQIdx = 0U;
    uint processorTally = processors; // initially, all processors are available, ensure that rQ has all pids
    uint tTB = 0;
    if (!threadIdx.x) {
      cuda::atomic_ref<unsigned int, cuda::thread_scope_block> tb{*taskBound};
      tTB = tb.load(cuda::memory_order_relaxed);
    }
    tTB = __shfl_sync(0xffffffff, tTB, 0);
    bool ranAffinity = false;
    if constexpr ((sL + dQL) <= 24) {
      if (tQ != nullptr && nLx > 0 && nLx <= 256 && ecTilesM > 0) {
        runAffinity<subscribers, processor_state_size>(
          tqState, processors, processors_v, tilesN1, sO, gtQCL, bitSet, tQHeads, gtQHeads,
          taskBound, rQ, sQ, pDB, tQ, nLx, ecTilesM, dT, gRQIdx, processorTally, scheduled, tTB);
        ranAffinity = true;
      }
    }
    if (!ranAffinity) while (scheduled < tTB) {
      // statically sweep tQ for tasks
      uint lTt = 0U; // local task tally
      #pragma unroll
      for (uint i = 0; i < sL; ++i) {
        auto* __restrict__ tqHp = tQHeads + (i * schedulerCount + threadIdx.x);
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> tqh{*tqHp};
        const auto tasks = tqh.load(cuda::memory_order_acquire) - tqState[i].tQTail;
        tqState[i].tasks = tasks;
        lTt += tasks;
      }

      // Abstract queues as a 3-D tensor (B, Q, T),
      // where B is the batch dimension or total queue / (Q * T);
      // Q is the number of queues a thread observes in one-pass;
      // and T is the number of threads in a warp
      if (dT > 0) {
        auto sBS = bitSet[threadIdx.x];
        // One-shot scheduling, so tails are irrelevant.
        // Special case, where i == 0
        #pragma unroll
        for (uint j = sL; j < tqState.size(); ++j) {
          const auto pJ = j - sL;
          if (const auto isVisited = sBS.get(pJ % bSw); !isVisited) {
            const auto qIdx = schedulerCount * pJ + threadIdx.x;
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> gtqH{*(gtQHeads + qIdx)};
            const auto tasks = gtqH.load(cuda::memory_order_acquire);
            if (tasks) {
              // one and done
              sBS.set(pJ % bSw);
              gtqH.store(tQHeadGroundState, cuda::memory_order_relaxed);
            }
            tqState[j].tasks = tasks;
            lTt += tasks;
          }
        }
        bitSet[threadIdx.x] = sBS;
        // schedule observed tasks
        schedulerLoop<processor_state_size, subscribers, sL, schedulerCount>(tqState, processors, processors_v, tilesN1, sO, 0, lTt,
        processorTally, gRQIdx, scheduled, sQ, rQ, pDB, true);

        for (uint i = 1; i < dT; ++i) {
          const uint sBIdx = threadIdx.x + (i * dQL / bSw) * schedulerCount;
          sBS = bitSet[sBIdx];
          // Needed to enforce register storage
          #pragma unroll
          for (uint j = sL; j < tqState.size(); ++j) {
            const auto pJ = j - sL;
            const uint bIdx = (i * dQL + pJ) % bSw;
            if (const auto isVisited = sBS.get(bIdx); !isVisited) {
              const auto qIdx = schedulerCount * (dQL * i + pJ) + threadIdx.x;
              cuda::atomic_ref<unsigned int, cuda::thread_scope_device> gtqH{*(gtQHeads + qIdx)};
              const auto tasks = gtqH.load(cuda::memory_order_acquire);
              if (tasks) {
                sBS.set(bIdx);
                gtqH.store(tQHeadGroundState, cuda::memory_order_relaxed);
              }
              tqState[j].tasks = tasks;
              lTt += tasks;
            }
          }
          bitSet[sBIdx] = sBS;
          // schedule observed tasks
          schedulerLoop<processor_state_size, subscribers, sL, schedulerCount>(tqState, processors, processors_v, tilesN1, sO, i * dQL,
          lTt, processorTally, gRQIdx, scheduled, sQ, rQ, pDB);
        }
      }
      if (threadIdx.x < gtQCL - dT * dQL * schedulerCount) {
        const uint sBIdx = threadIdx.x + (dT * dQL / bSw) * schedulerCount;
        auto sBS = bitSet[sBIdx];
        // residue
        #pragma unroll
        for (uint j = sL; j < tqState.size(); ++j) {
          const auto pJ = j - sL;
          if (const auto qIdx = schedulerCount * (dQL * dT + pJ) + threadIdx.x; qIdx < gtQCL) {
            const uint bIdx = (dT * dQL + pJ) % bSw;
            if (const auto isVisited = sBS.get(bIdx); !isVisited) {
              cuda::atomic_ref<unsigned int, cuda::thread_scope_device> gtqH{*(gtQHeads + qIdx)};
              const auto tasks = gtqH.load(cuda::memory_order_acquire);
              if (tasks) {
                sBS.set(bIdx);
                gtqH.store(tQHeadGroundState, cuda::memory_order_relaxed);
              }
              tqState[j].tasks = tasks;
              lTt += tasks;
            }
          }
        }
        bitSet[sBIdx] = sBS;
      }
      // schedule observed tasks
      schedulerLoop<processor_state_size, subscribers, sL, schedulerCount>(tqState, processors, processors_v, tilesN1, sO, dQL * dT,
      lTt, processorTally, gRQIdx, scheduled, sQ, rQ, pDB, dT == 0);

      if (!threadIdx.x) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> tb{*taskBound};
        tTB = tb.load(cuda::memory_order_relaxed);
      }
      __syncwarp();
      tTB = __shfl_sync(0xffffffff, tTB, 0);
    }
    __syncwarp();
    // interrupt subscribers
    #pragma unroll
    for (uint sid = threadIdx.x; sid < (subscribers /  WARP_SIZE); sid += SCHEDULER_COUNT) {
      cuda::atomic_ref<uint, cuda::thread_scope_block> inr{*(sInterrupts + sid)};
      inr.store(1, cuda::memory_order_relaxed);
    }
    // interrupt processors
    sPI<processor_state_size, schedulerCount>(rQ, sQ, pDB, gRQIdx, interruptScratch, processors, processors_v, processorTally);
  }
}
#endif //SCHEDULER_CUH
