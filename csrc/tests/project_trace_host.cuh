// Host exporter for the optional team project recorder.
#pragma once
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include "../include/flashmoe/infra/project_trace.cuh"
#if FLASHMOE_PROJECT_TRACE
namespace flashmoe::project_trace {
inline void checked(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
// Strict token strings keep exported CSV fields safe without escaping.
inline std::string token(const char* name, const char* fallback) {
  const auto* value = std::getenv(name);
  std::string s = value ? value : fallback;
  if (s.empty() || s.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.:") != std::string::npos)
    throw std::runtime_error(std::string("Invalid trace identifier: ") + name);
  return s;
}
class Capture {
  Buffer buffer{};
  unsigned blocks;
  int rank, world, experts;
  std::string prefix;
public:
  Capture(unsigned n, int rank_, int world_, int experts_) : blocks(n), rank(rank_), world(world_), experts(experts_) {
    const char* path = std::getenv("FLASHMOE_TRACE_PREFIX");
    if (!path) return;
    prefix = path;
    if (prefix.empty()) throw std::runtime_error("FLASHMOE_TRACE_PREFIX must not be empty");
    const auto parent = std::filesystem::path(prefix).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    buffer.capacity = 16384;
    if (const char* cap = std::getenv("FLASHMOE_TRACE_CAPACITY")) {
      char* end = nullptr;
      const auto value = std::strtoul(cap, &end, 10);
      if (!*cap || *end || value < 16 || value > 1048576)
        throw std::runtime_error("FLASHMOE_TRACE_CAPACITY must be 16..1048576 intervals/block");
      buffer.capacity = static_cast<unsigned>(value);
    }
    // Complete setup before enabling recorder; no captured graph is involved.
    try {
      checked(cudaMalloc(reinterpret_cast<void**>(&buffer.events), sizeof(Event) * blocks * buffer.capacity));
      checked(cudaMalloc(reinterpret_cast<void**>(&buffer.lanes), sizeof(Lane) * blocks));
      checked(cudaMemset(buffer.lanes, 0, sizeof(Lane) * blocks));
      // Allocated before warmup; recording remains disabled until start().
    } catch (...) {
      cudaFree(buffer.lanes); cudaFree(buffer.events);
      throw;
    }
  }
  Capture(const Capture&) = delete;
  Capture& operator=(const Capture&) = delete;
  ~Capture() {
    if (buffer.lanes) {
      const Buffer off{};
      cudaMemcpyToSymbol(active, &off, sizeof(off));
      cudaFree(buffer.lanes); cudaFree(buffer.events);
    }
  }
  bool requested() const { return buffer.lanes != nullptr; }
  void start(cudaStream_t stream) {
    if (!buffer.lanes) return;
    checked(cudaStreamSynchronize(stream));
    checked(cudaMemset(buffer.lanes, 0, sizeof(Lane) * blocks));
    checked(cudaMemcpyToSymbol(active, &buffer, sizeof(buffer)));
  }
  void export_csv(cudaStream_t stream) {
    if (!buffer.lanes) return;
    checked(cudaStreamSynchronize(stream));
    const Buffer off{};
    checked(cudaMemcpyToSymbol(active, &off, sizeof(off)));
    std::vector<Lane> lanes(blocks);
    checked(cudaMemcpy(lanes.data(), buffer.lanes, sizeof(Lane) * blocks, cudaMemcpyDeviceToHost));
    const auto run = token("FLASHMOE_TRACE_RUN_ID", "capture");
    const auto invocation = token("FLASHMOE_TRACE_INVOCATION", "0");
    const auto filename = prefix + ".rank" + std::to_string(rank) + ".csv";
    if (std::filesystem::exists(filename)) throw std::runtime_error("Refusing to overwrite trace " + filename);
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot open trace " + filename);
    out << "run_id,invocation,ngpu,E,rank,block_id,sm_id,role,state,start_ns,end_ns,window_start_ns,window_end_ns,dropped_events,task_type,tile_id\n";
    const char* states[] = {"other", "wait", "task"};
    unsigned overflow = 0;
    for (unsigned b = 0; b < blocks; ++b) {
      const auto& lane = lanes[b];
      if (!lane.end || !lane.count) throw std::runtime_error("Missing block trace");
      overflow += lane.dropped;
      std::vector<Event> events(lane.count);
      checked(cudaMemcpy(events.data(), buffer.events + static_cast<size_t>(b) * buffer.capacity,
                         events.size() * sizeof(Event), cudaMemcpyDeviceToHost));
      for (const auto& e : events) {
        out << run << ',' << invocation << ',' << world << ',' << experts << ',' << rank << ',' << b << ',' << lane.sm << ','
            << (b + 1 == blocks ? "os" : "processor") << ',' << states[e.state] << ','
            << e.begin << ',' << e.end << ',' << lane.begin << ',' << lane.end << ',' << lane.dropped << ','
            << e.task_type << ',' << e.tile << '\n';
      }
    }
    out.flush();
    if (!out) throw std::runtime_error("Error writing trace " + filename);
    std::cerr << "Device trace: " << filename << "; dropped intervals=" << overflow << '\n';
    if (overflow) throw std::runtime_error("Trace buffer overflow; increase FLASHMOE_TRACE_CAPACITY");
  }
};
}
#endif
