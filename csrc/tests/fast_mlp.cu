// The fused kernel gives each CTA one 128x128 tile and computes it with a
// single-CTA cuBLASDx GEMM. The anchor work is not that shape. Every expert
// has the same M, N, and K, so each projection is one strided-batched GEMM.
//
// Anchor FLOPs: E=16, M=1536, H=2048, I=1408, gated MLP, 0.425 TFLOP.
// Matrices are column-major.

#include <cstdio>
#include <cstdlib>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

using Element = __nv_bfloat16;

#define CHECK_CUDA(expr)                                                       \
  do {                                                                         \
    cudaError_t status_ = (expr);                                              \
    if (status_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA %s\n", cudaGetErrorString(status_));               \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS(expr)                                                     \
  do {                                                                         \
    cublasStatus_t status_ = (expr);                                           \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "cuBLAS status %d\n", static_cast<int>(status_));        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void fillOnes(Element* p, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    p[i] = __float2bfloat16(1.f);
  }
}

__global__ void siluMul(const Element* __restrict__ gate, const Element* __restrict__ up,
                        Element* __restrict__ out, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = __bfloat162float(gate[i]);
    float y = __bfloat162float(up[i]);
    float s = x / (1.f + expf(-x));
    out[i] = __float2bfloat16(s * y);
  }
}

// both is E matrices of shape (M x 2I), column-major, gate in columns [0, I) and
// up in columns [I, 2I). out is E matrices of shape (M x I), column-major.
__global__ void siluMulHalves(const Element* __restrict__ both, Element* __restrict__ out, int M, int I, int E) {
  const size_t n = static_cast<size_t>(E) * static_cast<size_t>(M) * static_cast<size_t>(I);
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += stride) {
    const int per = M * I;
    const int e = static_cast<int>(idx / static_cast<size_t>(per));
    const int rem = static_cast<int>(idx % static_cast<size_t>(per));
    const int m = rem % M;
    const int col = rem / M;
    const Element* base = both + (static_cast<size_t>(e) * M * (2 * I));
    const float x = __bfloat162float(base[static_cast<size_t>(col) * M + m]);
    const float y = __bfloat162float(base[static_cast<size_t>(I + col) * M + m]);
    const float s = x / (1.f + expf(-x));
    out[idx] = __float2bfloat16(s * y);
  }
}

struct LtGemm {
  cublasLtHandle_t lt{};
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t aDesc{};
  cublasLtMatrixLayout_t bDesc{};
  cublasLtMatrixLayout_t cDesc{};
  cublasLtMatmulAlgo_t algo{};
  void* workspace = nullptr;
  size_t workspaceBytes = 128ull << 20;
  bool ok = false;
};

void ltCreate(LtGemm& g, int M, int N, int K, int batch) {
  CHECK_CUBLAS(cublasLtCreate(&g.lt));
  CHECK_CUBLAS(cublasLtMatmulDescCreate(&g.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t trans = CUBLAS_OP_N;
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(g.op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(g.op, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&g.aDesc, CUDA_R_16BF, M, K, M));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&g.bDesc, CUDA_R_16BF, K, N, K));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&g.cDesc, CUDA_R_16BF, M, N, M));
  const int64_t strideA = static_cast<int64_t>(M) * K;
  const int64_t strideB = static_cast<int64_t>(K) * N;
  const int64_t strideC = static_cast<int64_t>(M) * N;
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(g.aDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(g.bDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(g.cDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(g.aDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(g.bDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(g.cDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));
  cublasLtMatmulPreference_t pref{};
  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g.workspaceBytes,
                                                    sizeof(g.workspaceBytes)));
  cublasLtMatmulHeuristicResult_t heuristic{};
  int found = 0;
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(g.lt, g.op, g.aDesc, g.bDesc, g.cDesc, g.cDesc, pref, 1, &heuristic, &found));
  cublasLtMatmulPreferenceDestroy(pref);
  if (found < 1) {
    fprintf(stderr, "no cublasLt algo for %d %d %d batch %d\n", M, N, K, batch);
    exit(1);
  }
  g.algo = heuristic.algo;
  CHECK_CUDA(cudaMalloc(&g.workspace, g.workspaceBytes));
  g.ok = true;
}

void ltRun(LtGemm& g, const Element* a, const Element* b, Element* c) {
  const float alpha = 1.f;
  const float beta = 0.f;
  CHECK_CUBLAS(cublasLtMatmul(g.lt, g.op, &alpha, a, g.aDesc, b, g.bDesc, &beta, c, g.cDesc, c, g.cDesc, &g.algo,
                              g.workspace, g.workspaceBytes, 0));
}

void gemm(cublasHandle_t handle, int M, int N, int K, int batch, const Element* a, const Element* b, Element* c) {
  const float alpha = 1.f;
  const float beta = 0.f;
  CHECK_CUBLAS(cublasGemmStridedBatchedEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, a, CUDA_R_16BF, M, static_cast<long long>(M) * K, b,
      CUDA_R_16BF, K, static_cast<long long>(K) * N, &beta, c, CUDA_R_16BF, M, static_cast<long long>(M) * N, batch,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

float timeLoop(cudaEvent_t start, cudaEvent_t stop, int warmup, int runs, const auto& body) {
  for (int i = 0; i < warmup; ++i) {
    body();
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < runs; ++i) {
    body();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  return ms / static_cast<float>(runs);
}

int main() {
  constexpr int E = 16;
  constexpr int M = 1536;
  constexpr int H = 2048;
  constexpr int I = 1408;
  constexpr int warmup = 5;
  constexpr int runs = 20;
  const double flops = 6.0 * static_cast<double>(E) * M * I * H;

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  printf("device %s sms %d\n", prop.name, prop.multiProcessorCount);

  const size_t tokenN = static_cast<size_t>(E) * M * H;
  const size_t wN = static_cast<size_t>(E) * H * I;
  const size_t midN = static_cast<size_t>(E) * M * I;
  Element* tokens = nullptr;
  Element* wGate = nullptr;
  Element* wUp = nullptr;
  Element* wDown = nullptr;
  Element* wBoth = nullptr;
  Element* cGate = nullptr;
  Element* cUp = nullptr;
  Element* cBoth = nullptr;
  Element* activated = nullptr;
  Element* out = nullptr;
  CHECK_CUDA(cudaMalloc(&tokens, tokenN * sizeof(Element)));
  CHECK_CUDA(cudaMalloc(&wGate, wN * sizeof(Element)));
  CHECK_CUDA(cudaMalloc(&wUp, wN * sizeof(Element)));
  CHECK_CUDA(cudaMalloc(&wDown, wN * sizeof(Element)));
  CHECK_CUDA(cudaMalloc(&wBoth, 2 * wN * sizeof(Element)));
  CHECK_CUDA(cudaMalloc(&cGate, midN * sizeof(Element)));
  CHECK_CUDA(cudaMalloc(&cUp, midN * sizeof(Element)));
  CHECK_CUDA(cudaMalloc(&cBoth, 2 * midN * sizeof(Element)));
  CHECK_CUDA(cudaMalloc(&activated, midN * sizeof(Element)));
  CHECK_CUDA(cudaMalloc(&out, tokenN * sizeof(Element)));

  auto launchFill = [](Element* p, size_t n) {
    fillOnes<<<static_cast<int>((n + 255) / 256), 256>>>(p, n);
  };
  launchFill(tokens, tokenN);
  launchFill(wGate, wN);
  launchFill(wUp, wN);
  launchFill(wDown, wN);
  const size_t one = static_cast<size_t>(H) * I;
  for (int e = 0; e < E; ++e) {
    CHECK_CUDA(cudaMemcpy(wBoth + (2 * e) * one, wGate + e * one, one * sizeof(Element), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(wBoth + (2 * e + 1) * one, wUp + e * one, one * sizeof(Element), cudaMemcpyDeviceToDevice));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cublasHandle_t handle{};
  CHECK_CUBLAS(cublasCreate(&handle));
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  const int actBlocks = static_cast<int>((midN + 255) / 256);
  const float separateMs = timeLoop(start, stop, warmup, runs, [&] {
    gemm(handle, M, I, H, E, tokens, wGate, cGate);
    gemm(handle, M, I, H, E, tokens, wUp, cUp);
    siluMul<<<actBlocks, 256>>>(cGate, cUp, activated, midN);
    gemm(handle, M, H, I, E, activated, wDown, out);
  });
  const float fusedMs = timeLoop(start, stop, warmup, runs, [&] {
    gemm(handle, M, 2 * I, H, E, tokens, wBoth, cBoth);
    siluMulHalves<<<1024, 256>>>(cBoth, activated, M, I, E);
    gemm(handle, M, H, I, E, activated, wDown, out);
  });

  LtGemm gateLt{};
  LtGemm upLt{};
  LtGemm downLt{};
  ltCreate(gateLt, M, I, H, E);
  ltCreate(upLt, M, I, H, E);
  ltCreate(downLt, M, H, I, E);
  const float ltMs = timeLoop(start, stop, warmup, runs, [&] {
    ltRun(gateLt, tokens, wGate, cGate);
    ltRun(upLt, tokens, wUp, cUp);
    siluMul<<<actBlocks, 256>>>(cGate, cUp, activated, midN);
    ltRun(downLt, activated, wDown, out);
  });

  auto report = [&](const char* name, float ms) {
    printf("%s, %.4f ms, %.1f TFLOP/s\n", name, ms, flops / (static_cast<double>(ms) * 1.0e9));
  };
  report("separate_gate_up", separateMs);
  report("fused_gate_up", fusedMs);
  report("cublaslt_separate", ltMs);
  printf("anchor_flops %.6f TFLOP\n", flops / 1.0e12);
  printf("current_fused_kernel 3.81 ms, %.1f TFLOP/s\n", flops / (3.81 * 1.0e9));
  return 0;
}
