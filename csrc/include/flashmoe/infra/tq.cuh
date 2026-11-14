//
// Created by osayamen on 1/7/26.
//

#ifndef FLASHMOE_TQ_CUH
#define FLASHMOE_TQ_CUH

namespace flashmoe {
  enum SchedulerConstants : uint {
    interruptSignal = 0,
    tQHeadGroundState = 0
  };

  // rq
  enum ReadySignal : uint {
    observed,
    ready,
    COUNT = 2
  };

  struct __align__(8) TQState {
    uint tQTail;
    uint tasks;
  };


  // Doorbell index lives in the low 24 bits (stored as index + 1).
  // A non-zero high byte is a wave length: one wake runs that many
  // consecutive tasks from the same expert queue. High byte 0 means one task,
  // which is the original doorbell.
  // Affinity already concentrates one expert; pack more tiles per wake so the
  // OS block and doorbell path amortize across that expert's 128x128 tiles.
  constexpr uint kSignalIndexBits = 24;
  constexpr uint kSignalIndexMask = (1u << kSignalIndexBits) - 1u;
  // 4 matched the first megakernel-wave experiment. Larger values over-serialized
  // a processor on the login A100 smoke (paper shape got slower at 8).
  constexpr uint kExpertWave = 4;

  struct __align__(8) TQSignal {
    uint signal; // one ahead
    uint interrupt;

    __device__ __forceinline__
    void encodeSig(const uint &sig) {
      signal = sig + 1;
    }

    __device__ __forceinline__
    auto decodeSig() const {
      return signal - 1;
    }
  };
}
#endif //FLASHMOE_TQ_CUH
