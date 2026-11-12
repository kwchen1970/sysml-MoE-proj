// Proof of concept for the K-loop, not the full MoE forward.
// Times the shipping CollectiveMainloop (bf16, 128x128) against cuBLAS on one GPU.
// A and B are filled with 1, so every C element equals K. That checks the store
// path without pulling in MatX.
//
// cuBLASDx rejects sm_modifier::arch_specific below SM90, so an SM80-specific
// MMA schedule is not a candidate here. The knobs this binary can actually
// change are pipeline stages and bK.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <cublasdx.hpp>

#include "flashmoe/tile.cuh"

using Element = __nv_bfloat16;

#define CHECK_CUDA(expr)                                                       \
  do {                                                                         \
    const cudaError_t status_ = (expr);                                        \
    if (status_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,                   \
              cudaGetErrorString(status_));                                    \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS(expr)                                                     \
  do {                                                                         \
    const cublasStatus_t status_ = (expr);                                     \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "cuBLAS %s:%d status %d\n", __FILE__, __LINE__,          \
              static_cast<int>(status_));                                      \
      return;                                                                  \
    }                                                                          \
  } while (0)

__global__ void fillOnes(Element* p, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    p[i] = __float2bfloat16(1.f);
  }
}

__global__ void maxAbsErr(const Element* c, int n, float expect, float* blockErr) {
  float local = 0.f;
  for (int i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x); i < n;
       i += static_cast<int>(blockDim.x * gridDim.x)) {
    local = fmaxf(local, fabsf(__bfloat162float(c[i]) - expect));
  }
  __shared__ float scratch[256];
  scratch[threadIdx.x] = local;
  __syncthreads();
  for (int offset = 128; offset > 0; offset >>= 1) {
    if (static_cast<int>(threadIdx.x) < offset) {
      scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    blockErr[blockIdx.x] = scratch[0];
  }
}

float reduceErr(const Element* c, int n, float expect) {
  constexpr int threads = 256;
  constexpr int blocks = 128;
  float* blockErr = nullptr;
  cudaMalloc(&blockErr, blocks * sizeof(float));
  maxAbsErr<<<blocks, threads>>>(c, n, expect, blockErr);
  float host[blocks];
  cudaMemcpy(host, blockErr, sizeof(host), cudaMemcpyDeviceToHost);
  cudaFree(blockErr);
  float err = 0.f;
  for (float v : host) {
    err = std::max(err, v);
  }
  return err;
}

template <typename Tile>
__launch_bounds__(Tile::BLAS::max_threads_per_block, 1)
__global__ void kloop(const Element* __restrict__ a, const Element* __restrict__ b,
                      Element* __restrict__ c, int M, int N, int K) {
  using BLAS = typename Tile::BLAS;
  constexpr int bM = cublasdx::size_of<BLAS>::m;
  constexpr int bN = cublasdx::size_of<BLAS>::n;
  const int nTiles = (M / bM) * (N / bN);
  extern __shared__ __align__(16) cuda::std::byte workspace[];
  for (int tileIdx = static_cast<int>(blockIdx.x); tileIdx < nTiles; tileIdx += static_cast<int>(gridDim.x)) {
    auto accumulator = BLAS::suggest_accumulator();
    const auto tileCoord = flashmoe::tile::idx2Coord(M / bM, N / bN, tileIdx);
    constexpr Tile mainloop{};
    mainloop(workspace, a, b, accumulator, M, N, K, tileCoord);
    auto gC = flashmoe::tile::getC<bM, bN, cublasdx::arrangement_of_v_c<BLAS>>(
        c, M, N, cute::select<0, 1>(tileCoord));
    auto sC = cublasdx::make_tensor(reinterpret_cast<Element*>(workspace), BLAS::suggest_layout_smem_c());
    const auto cFrag = accumulator.get_results();
    auto dFrag = cublasdx::make_fragment_like<Element>(cFrag);
    constexpr int accumSize = cublasdx::size(cFrag);
    constexpr flashmoe::Converter<Element, float> storeConv{};
    cute::for_each(cute::make_int_sequence<accumSize>{}, [&](auto i) {
      dFrag(i) = storeConv(cFrag(i));
    });
    __syncthreads();
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(dFrag, sC, accumulator);
    __syncthreads();
    cublasdx::copy<BLAS, cublasdx::alignment_of<BLAS>::c>(sC, gC);
  }
}

void benchCublas(cublasHandle_t handle, int M, int N, int K, const Element* a, const Element* b, Element* c) {
  const float alpha = 1.f;
  const float beta = 0.f;
  // Row-major C(MxN) = A(MxK) * B_col(KxN) is column-major C^T = B^T * A^T.
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, b, CUDA_R_16BF, K, a,
                            CUDA_R_16BF, K, &beta, c, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <int bM, int bN, int bK, int pS>
void benchTile(const char* name, int M, int N, int K, Element* a, Element* b, Element* c, int sms) {
  constexpr int arch = 800;
  constexpr int threads =
      flashmoe::tile::suggest_thread_count<bM, bN, bK, arch, Element, float>();
  using Tile = flashmoe::tile::CollectiveMainloop<bM, bN, bK, arch, Element, float, threads, pS>;
  auto* kernel = kloop<Tile>;
  constexpr int sharedBytes = std::max(Tile::SharedSizeAB::value, Tile::SharedSizeC::value);
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedBytes));
  int blocksPerSm = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kernel, threads, sharedBytes));
  cudaFuncAttributes attr{};
  CHECK_CUDA(cudaFuncGetAttributes(&attr, kernel));
  const int nTiles = (M / bM) * (N / bN);
  const int blocks = std::min(nTiles, blocksPerSm * sms);
  constexpr int warmup = 5;
  constexpr int runs = 20;
  for (int i = 0; i < warmup; ++i) {
    kernel<<<blocks, threads, sharedBytes>>>(a, b, c, M, N, K);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < runs; ++i) {
    kernel<<<blocks, threads, sharedBytes>>>(a, b, c, M, N, K);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  ms /= static_cast<float>(runs);
  const double flops = 2.0 * static_cast<double>(M) * N * K;
  const double tflops = flops / (static_cast<double>(ms) * 1.0e9);
  const float err = reduceErr(c, M * N, static_cast<float>(K));
  printf("tile,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%.2f,%.3f\n", name, M, N, K, bM, bN, bK, pS,
         threads, sharedBytes, blocksPerSm, blocks, attr.numRegs, static_cast<int>(attr.localSizeBytes), ms, tflops,
         err);
  fflush(stdout);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void benchLibrary(cublasHandle_t handle, int M, int N, int K, Element* a, Element* b, Element* c) {
  constexpr int warmup = 5;
  constexpr int runs = 20;
  for (int i = 0; i < warmup; ++i) {
    benchCublas(handle, M, N, K, a, b, c);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < runs; ++i) {
    benchCublas(handle, M, N, K, a, b, c);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  ms /= static_cast<float>(runs);
  const double flops = 2.0 * static_cast<double>(M) * N * K;
  const double tflops = flops / (static_cast<double>(ms) * 1.0e9);
  const float err = reduceErr(c, M * N, static_cast<float>(K));
  printf("cublas,cublas,%d,%d,%d,0,0,0,0,0,0,0,0,0,0,%.4f,%.2f,%.3f\n", M, N, K, ms, tflops, err);
  fflush(stdout);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void runShape(cublasHandle_t handle, int M, int N, int K, int sms) {
  Element* a = nullptr;
  Element* b = nullptr;
  Element* c = nullptr;
  cudaMalloc(&a, static_cast<size_t>(M) * K * sizeof(Element));
  cudaMalloc(&b, static_cast<size_t>(N) * K * sizeof(Element));
  cudaMalloc(&c, static_cast<size_t>(M) * N * sizeof(Element));
  fillOnes<<<(static_cast<int>(static_cast<size_t>(M) * K) + 255) / 256, 256>>>(a, static_cast<size_t>(M) * K);
  fillOnes<<<(static_cast<int>(static_cast<size_t>(N) * K) + 255) / 256, 256>>>(b, static_cast<size_t>(N) * K);
  cudaDeviceSynchronize();
  benchLibrary(handle, M, N, K, a, b, c);
  benchTile<128, 128, 32, 2>("bk32_ps2", M, N, K, a, b, c, sms);
  benchTile<128, 128, 32, 3>("bk32_ps3", M, N, K, a, b, c, sms);
  benchTile<128, 128, 32, 4>("bk32_ps4", M, N, K, a, b, c, sms);
  benchTile<128, 128, 64, 2>("bk64_ps2", M, N, K, a, b, c, sms);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

int main() {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  printf("device %s sms %d l2 %d\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize);
  printf("kind,name,M,N,K,bM,bN,bK,pS,threads,smem,blocks_per_sm,blocks,regs,local_bytes,ms,tflops,max_abs_err\n");
  fflush(stdout);
  cublasHandle_t handle{};
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublasCreate failed\n");
    return 1;
  }
  const int sms = prop.multiProcessorCount;
  // One expert of the anchor, under-fills 216 CTAs: M=1536, N=1408, K=2048.
  runShape(handle, 1536, 1408, 2048, sms);
  // Enough 128x128 tiles to cover 2 blocks on every SM. K matches GEMM0 and GEMM1.
  runShape(handle, 3072, 2048, 2048, sms);
  runShape(handle, 3072, 2048, 1408, sms);
  cublasDestroy(handle);
  return 0;
}
