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
  constexpr uint kSignalIndexBits = 24;
  constexpr uint kSignalIndexMask = (1u << kSignalIndexBits) - 1u;
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
