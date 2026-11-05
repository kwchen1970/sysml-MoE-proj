// Benchmark-only single-node GPU selection and MPI/NVSHMEM bootstrap.
#pragma once
#include <cuda_runtime.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace flashmoe_bench {
struct Runtime { unsigned device; bool owns_mpi; };
inline const char* env(const char* key) {
  const char* value = std::getenv(key);
  return value ? value : "<unset>";
}
[[noreturn]] inline void fail(int rank, const std::string& message) {
  std::fprintf(stderr, "[FlashMoE rank=%d] preflight FAILED: %s\n", rank, message.c_str());
  std::fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 1);
  std::abort();
}
inline void cuda_check(cudaError_t result, int rank, const char* what) {
  if (result != cudaSuccess) fail(rank, std::string(what) + ": " + cudaGetErrorString(result));
}
inline std::string uuid_string(const cudaUUID_t& uuid) {
  const char* hex = "0123456789abcdef";
  std::string result;
  for (unsigned char byte : uuid.bytes) {
    result += hex[byte >> 4]; result += hex[byte & 15];
  }
  return result;
}
inline Runtime initialize() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) MPI_Init(nullptr, nullptr);
  int rank = 0, world = 0, local_rank = 0, local_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm local;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
  MPI_Comm_rank(local, &local_rank);
  MPI_Comm_size(local, &local_size);
  if (world != local_size) fail(rank, "This experiment launcher supports one node only.");
  int count = 0;
  cuda_check(cudaGetDeviceCount(&count), rank, "cudaGetDeviceCount");
  std::fprintf(stderr,
    "[FlashMoE rank=%d local_rank=%d/%d] CUDA_VISIBLE_DEVICES=%s visible_count=%d "
    "SLURM_LOCALID=%s remote_transport=%s\n",
    rank, local_rank, local_size, env("CUDA_VISIBLE_DEVICES"), count,
    env("SLURM_LOCALID"), env("NVSHMEM_REMOTE_TRANSPORT"));
  std::fflush(stderr);
  if (count < local_size)
    fail(rank, "Not enough mutually visible GPUs. Use the repository single-node launcher; "
               "multi-GPU steps need --gpus-per-node=4 --gpu-bind=none, not --gpus-per-task=1.");
  // All allocated devices remain visible; one rank owns each selected device.
  const int selected = local_rank;
  cuda_check(cudaSetDevice(selected), rank, "cudaSetDevice");
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, selected), rank, "cudaGetDeviceProperties");
  char bus[32]{};
  cuda_check(cudaDeviceGetPCIBusId(bus, sizeof(bus), selected), rank, "cudaDeviceGetPCIBusId");
  std::fprintf(stderr, "[FlashMoE rank=%d] selected_ordinal=%d GPU=%s UUID=%s PCI=%s\n",
    rank, selected, prop.name, uuid_string(prop.uuid).c_str(), bus);
  std::fflush(stderr);
  std::vector<cudaUUID_t> peers(local_size);
  MPI_Allgather(&prop.uuid, sizeof(cudaUUID_t), MPI_BYTE,
                peers.data(), sizeof(cudaUUID_t), MPI_BYTE, local);
  for (int peer = 0; peer < local_size; ++peer) {
    if (peer == local_rank) continue;
    if (!std::memcmp(&prop.uuid, &peers[peer], sizeof(cudaUUID_t)))
      fail(rank, "Two local ranks selected the same physical GPU UUID.");
    int ordinal = -1;
    for (int device = 0; device < count; ++device) {
      cudaDeviceProp candidate{};
      cuda_check(cudaGetDeviceProperties(&candidate, device), rank, "peer properties");
      if (!std::memcmp(&candidate.uuid, &peers[peer], sizeof(cudaUUID_t))) ordinal = device;
    }
    if (ordinal < 0) fail(rank, "A peer GPU UUID is absent from this rank's visible devices.");
    int accessible = 0;
    cuda_check(cudaDeviceCanAccessPeer(&accessible, selected, ordinal), rank, "cudaDeviceCanAccessPeer");
    std::fprintf(stderr, "[FlashMoE rank=%d] peer_local_rank=%d peer_ordinal=%d P2P=%d\n",
                 rank, peer, ordinal, accessible);
    if (!accessible) fail(rank, "CUDA P2P is unavailable to a selected peer; no transport fallback enabled.");
  }
  MPI_Comm_free(&local);
  MPI_Barrier(MPI_COMM_WORLD);
  std::fprintf(stderr, "[FlashMoE rank=%d] distinct GPU/P2P preflight OK; NVSHMEM init BEGIN\n", rank);
  std::fflush(stderr);
  MPI_Comm comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  attr.mpi_comm = &comm;
  const int status = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  if (status) fail(rank, "nvshmemx_init_attr failed: " + std::to_string(status));
  std::fprintf(stderr, "[FlashMoE rank=%d] NVSHMEM init returned OK (PE=%d/%d); "
                       "device-state synchronization BEGIN\n", rank, nvshmem_my_pe(), nvshmem_n_pes());
  std::fflush(stderr);
  return {static_cast<unsigned>(selected), !initialized};
}
}
