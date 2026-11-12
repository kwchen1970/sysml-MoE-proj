#pragma once

// Hypothesis: expert weights are reloaded from HBM because the scheduler
// issues every ready expert at once. Two gated experts of the shapes we run
// fill a 40 MB L2. Prefer that pair, and admit anyone else only when they
// are needed to fill free CTAs or they hold more ready work than the pair.

#ifdef __CUDACC__
#define FLASHMOE_HD __host__ __device__ __forceinline__
#else
#define FLASHMOE_HD inline
#endif

namespace flashmoe::scheduler {
constexpr unsigned kNoExpert = 0xffffffffu;

struct LocalQueue {
  unsigned expert;
  unsigned tasks;
  bool globalQueue;
};

// hist[e] is ready tasks whose front expert is e. Ties keep the lower index.
// A sticky expert stays primary until another expert has strictly more work.
FLASHMOE_HD void chooseHotExperts(const unsigned* hist, unsigned nExperts,
                                  unsigned sticky0, unsigned sticky1,
                                  unsigned& hot0, unsigned& hot1) {
  unsigned best = kNoExpert;
  unsigned bestCount = 0;
  unsigned second = kNoExpert;
  unsigned secondCount = 0;
  for (unsigned e = 0; e < nExperts; ++e) {
    const unsigned c = hist[e];
    if (c > bestCount) {
      second = best;
      secondCount = bestCount;
      best = e;
      bestCount = c;
    } else if (c > secondCount) {
      second = e;
      secondCount = c;
    }
  }
  const bool stickyAlive = sticky0 != kNoExpert && sticky0 < nExperts && hist[sticky0] > 0;
  if (stickyAlive && hist[sticky0] >= bestCount) {
    hot0 = sticky0;
  } else {
    hot0 = best;
  }
  hot1 = kNoExpert;
  if (hot0 == kNoExpert) {
    return;
  }
  if (stickyAlive && sticky0 != hot0) {
    hot1 = sticky0;
  } else if (best != kNoExpert && best != hot0) {
    hot1 = best;
  } else if (second != kNoExpert && second != hot0) {
    hot1 = second;
  }
  (void)sticky1;
}

FLASHMOE_HD bool isHotExpert(unsigned expert, unsigned hot0, unsigned hot1) {
  return expert == hot0 || (hot1 != kNoExpert && expert == hot1);
}

// Two gated experts of this size already fill a 40 MB L2. When the longer
// expert alone can occupy every free CTA, drop the partner.
FLASHMOE_HD void narrowToLongPole(const unsigned* hist, unsigned hot0, unsigned& hot1, unsigned slots) {
  if (hot0 != kNoExpert && slots > 0 && hist[hot0] >= slots) {
    hot1 = kNoExpert;
  }
}

// Prefix-clamp lane totals so the warp issues at most `slots` tasks.
FLASHMOE_HD void clampToSlots(const unsigned* local, unsigned n, unsigned slots, unsigned* budget) {
  unsigned prefix = 0;
  for (unsigned i = 0; i < n; ++i) {
    const unsigned before = prefix;
    prefix += local[i];
    if (before >= slots) {
      budget[i] = 0;
    } else {
      const unsigned room = slots - before;
      budget[i] = local[i] < room ? local[i] : room;
    }
  }
}

// Spend hotBudget on hot queues and coldBudget on the rest.
// Global queues are taken whole so a one-shot head is never split.
// Subscriber queues may be taken partially. Global queues are preferred
// inside a budget so a finished GEMM1 is not stuck behind more GEMM0.
FLASHMOE_HD void assignQueues(const LocalQueue* queues, unsigned nq,
                              unsigned hot0, unsigned hot1,
                              unsigned hotBudget, unsigned coldBudget,
                              unsigned* take) {
  for (unsigned i = 0; i < nq; ++i) {
    take[i] = 0;
  }
  for (unsigned pass = 0; pass < 2; ++pass) {
    const bool wantGlobal = pass == 0;
    for (unsigned i = 0; i < nq; ++i) {
      const LocalQueue& q = queues[i];
      if (q.tasks == 0 || q.globalQueue != wantGlobal || !isHotExpert(q.expert, hot0, hot1)) {
        continue;
      }
      if (hotBudget == 0) {
        continue;
      }
      if (q.globalQueue) {
        take[i] = q.tasks;
        hotBudget = hotBudget > q.tasks ? hotBudget - q.tasks : 0;
      } else if (hotBudget >= q.tasks) {
        take[i] = q.tasks;
        hotBudget -= q.tasks;
      } else {
        take[i] = hotBudget;
        hotBudget = 0;
      }
    }
  }
  for (unsigned pass = 0; pass < 2; ++pass) {
    const bool wantGlobal = pass == 0;
    for (unsigned i = 0; i < nq; ++i) {
      const LocalQueue& q = queues[i];
      if (q.tasks == 0 || q.globalQueue != wantGlobal || take[i] != 0 ||
          isHotExpert(q.expert, hot0, hot1)) {
        continue;
      }
      if (coldBudget == 0) {
        continue;
      }
      if (q.globalQueue) {
        take[i] = q.tasks;
        coldBudget = coldBudget > q.tasks ? coldBudget - q.tasks : 0;
      } else if (coldBudget >= q.tasks) {
        take[i] = q.tasks;
        coldBudget -= q.tasks;
      } else {
        take[i] = coldBudget;
        coldBudget = 0;
      }
    }
  }
}
} // namespace flashmoe::scheduler
