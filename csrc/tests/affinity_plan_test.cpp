#include <cstdio>

#include "../include/flashmoe/infra/affinity.cuh"

using flashmoe::scheduler::LocalQueue;
using flashmoe::scheduler::assignQueues;
using flashmoe::scheduler::chooseHotExperts;
using flashmoe::scheduler::clampToSlots;
using flashmoe::scheduler::isHotExpert;
using flashmoe::scheduler::kNoExpert;

static int failures = 0;

static void expect(bool cond, const char* msg) {
  if (!cond) {
    std::printf("FAIL %s\n", msg);
    ++failures;
  }
}

static void testClamp() {
  const unsigned local[4] = {8, 8, 8, 3};
  unsigned budget[4] = {};
  clampToSlots(local, 4, 10, budget);
  expect(budget[0] == 8 && budget[1] == 2 && budget[2] == 0 && budget[3] == 0, "clamp splits the slot cap");
  clampToSlots(local, 4, 100, budget);
  expect(budget[0] == 8 && budget[1] == 8 && budget[2] == 8 && budget[3] == 3, "clamp keeps everything under a wide cap");
}

static void testStickyAndLongPole() {
  unsigned hist[8] = {};
  hist[0] = 10;
  hist[1] = 10;
  unsigned hot0 = kNoExpert;
  unsigned hot1 = kNoExpert;
  chooseHotExperts(hist, 8, kNoExpert, kNoExpert, hot0, hot1);
  expect(hot0 == 0 && hot1 == 1, "tie keeps the lower expert primary");

  chooseHotExperts(hist, 8, 0, 1, hot0, hot1);
  expect(hot0 == 0, "equal work does not bounce off the sticky expert");

  hist[3] = 11;
  chooseHotExperts(hist, 8, 0, 1, hot0, hot1);
  expect(hot0 == 3, "strictly more work becomes the long pole");
  expect(hot1 == 0, "previous sticky expert stays second so its weights are not dropped");
}

static void testAdmitAndCap() {
  // 32 lanes. Lane t owns one subscriber queue of expert t%4 with 20 tasks.
  unsigned hist[4] = {160, 160, 160, 160};
  unsigned hot0, hot1;
  chooseHotExperts(hist, 4, kNoExpert, kNoExpert, hot0, hot1);
  expect(hot0 == 0 && hot1 == 1, "two fullest experts form the hot set");
  const unsigned hotTotal = hist[hot0] + hist[hot1];
  const unsigned slots = 32;
  expect(hotTotal >= slots, "hot work fills the machine");

  unsigned hotLocal[32] = {};
  unsigned coldLocal[32] = {};
  LocalQueue queues[32];
  for (unsigned t = 0; t < 32; ++t) {
    queues[t] = LocalQueue{t % 4, 20, false};
    if (isHotExpert(queues[t].expert, hot0, hot1)) {
      hotLocal[t] = 20;
    } else {
      coldLocal[t] = 20;
    }
  }
  unsigned hotBudget[32];
  unsigned coldBudget[32];
  clampToSlots(hotLocal, 32, slots, hotBudget);
  const unsigned coldSlots = 0;
  clampToSlots(coldLocal, 32, coldSlots, coldBudget);

  unsigned issuedHot = 0;
  unsigned issuedCold = 0;
  for (unsigned t = 0; t < 32; ++t) {
    unsigned take = 0;
    assignQueues(&queues[t], 1, hot0, hot1, hotBudget[t], coldBudget[t], &take);
    if (isHotExpert(queues[t].expert, hot0, hot1)) {
      issuedHot += take;
    } else {
      issuedCold += take;
    }
  }
  expect(issuedCold == 0, "cold experts wait while the hot pair fills every CTA");
  expect(issuedHot == slots, "the wave size is the number of free CTAs");
}

static void testAdmitColdWhenShort() {
  // Top two experts have 4 tasks each. A third expert is colder and is only
  // admitted because 8 tasks cannot occupy 32 CTAs.
  unsigned hist[4] = {4, 4, 0, 4};
  unsigned hot0, hot1;
  chooseHotExperts(hist, 4, kNoExpert, kNoExpert, hot0, hot1);
  expect(hot0 == 0 && hot1 == 1, "the two fullest experts are hot");
  const unsigned slots = 32;
  const unsigned hotTotal = hist[hot0] + hist[hot1];
  expect(hotTotal < slots, "hot work cannot fill the CTAs");

  LocalQueue queues[2] = {
    LocalQueue{0, 4, false},
    LocalQueue{3, 40, false},
  };
  unsigned hotBudget[1];
  unsigned coldBudget[1];
  const unsigned hotLocal[1] = {4};
  const unsigned coldLocal[1] = {40};
  clampToSlots(hotLocal, 1, 0xffffffffu, hotBudget);
  const unsigned coldSlots = slots - hotTotal;
  clampToSlots(coldLocal, 1, coldSlots, coldBudget);
  unsigned take[2] = {};
  assignQueues(queues, 2, hot0, hot1, hotBudget[0], coldBudget[0], take);
  expect(take[0] == 4, "all hot tasks are issued when they do not fill the GPU");
  expect(take[1] == coldSlots, "a cold expert is admitted only to fill idle CTAs");
}

static void testGlobalQueueIsWhole() {
  LocalQueue queues[1] = {LocalQueue{1, 16, true}};
  unsigned take = 0;
  assignQueues(queues, 1, 1, kNoExpert, 5, 0, &take);
  expect(take == 16, "a GEMM1 queue is taken whole so its head is not split");
  take = 0;
  assignQueues(queues, 1, 1, kNoExpert, 0, 0, &take);
  expect(take == 0, "a zero budget leaves the one-shot queue untouched");
}

int main() {
  testClamp();
  testStickyAndLongPole();
  testAdmitAndCap();
  testAdmitColdWhenShort();
  testGlobalQueueIsWhole();
  if (failures) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("affinity plan ok\n");
  return 0;
}
