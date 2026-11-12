// Expert-parallel gated MLP.
//
// The fused kernel issues one 128x128 single-CTA GEMM at a time. This forward
// packs each expert into one matrix and runs three cuBLASLt GEMMs (gate, up,
// down) plus SiLU. One GPU skips the network. Several GPUs keep blocked
// expert ownership, exchange packed rows with NCCL all-to-all, and run the
// same GEMMs on the local experts. Communication sits outside the GEMM, at
// expert granularity, which is the overlap grain that still fills an A100.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#ifdef FAST_MOE_DISTRIBUTED
#include <mpi.h>
#include <nccl.h>
#endif

using Element = __nv_bfloat16;

#define CHECK_CUDA(expr)                                                       \
  do {                                                                         \
    cudaError_t status_ = (expr);                                              \
    if (status_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA %s (%s:%d)\n", cudaGetErrorString(status_),        \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS(expr)                                                     \
  do {                                                                         \
    cublasStatus_t status_ = (expr);                                           \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "cuBLAS %d (%s:%d)\n", static_cast<int>(status_),        \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int rankId = 0;
int worldSize = 1;

#ifdef FAST_MOE_DISTRIBUTED
#define CHECK_NCCL(expr)                                                       \
  do {                                                                         \
    ncclResult_t status_ = (expr);                                             \
    if (status_ != ncclSuccess) {                                              \
      fprintf(stderr, "NCCL %s (%s:%d)\n", ncclGetErrorString(status_),        \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

ncclComm_t ncclComm{};

void exchange(const void* send, void* recv, size_t count, size_t elem, ncclDataType_t type,
              cudaStream_t stream) {
  const char* src = static_cast<const char*>(send);
  char* dst = static_cast<char*>(recv);
  const size_t bytes = count * elem;
  CHECK_CUDA(cudaMemcpyAsync(dst + static_cast<size_t>(rankId) * bytes, src + static_cast<size_t>(rankId) * bytes,
                             bytes, cudaMemcpyDeviceToDevice, stream));
  CHECK_NCCL(ncclGroupStart());
  for (int peer = 0; peer < worldSize; ++peer) {
    if (peer == rankId) continue;
    CHECK_NCCL(ncclSend(src + static_cast<size_t>(peer) * bytes, count, type, peer, ncclComm, stream));
    CHECK_NCCL(ncclRecv(dst + static_cast<size_t>(peer) * bytes, count, type, peer, ncclComm, stream));
  }
  CHECK_NCCL(ncclGroupEnd());
}
#endif

constexpr float kToken = 0.02f;
constexpr float kWeight = 0.02f;
constexpr float kBias = 0.01f;

__global__ void fillConst(Element* p, size_t n, float v) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) p[i] = __float2bfloat16(v);
}

__global__ void fillPattern(Element* tokens, int S, int H, int rank) {
  int t = blockIdx.x;
  if (t >= S) return;
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    float v = (static_cast<float>(rank) + 1.f) * 0.001f +
              (static_cast<float>(t) + 1.f) * 0.0001f +
              (static_cast<float>(h) + 1.f) * 0.000013f;
    tokens[static_cast<size_t>(t) * H + h] = __float2bfloat16(v);
  }
}

__global__ void packRows(const Element* __restrict__ tokens,
                         const int* __restrict__ slotToken, Element* __restrict__ dst,
                         int H, int slots) {
  int slot = blockIdx.x;
  if (slot >= slots) return;
  const int t = slotToken[slot];
  const uint4* src = reinterpret_cast<const uint4*>(tokens + static_cast<size_t>(t) * H);
  uint4* out = reinterpret_cast<uint4*>(dst + static_cast<size_t>(slot) * H);
  const int nvec = H >> 3;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) out[i] = src[i];
}

__global__ void checkPacked(const Element* __restrict__ packed,
                            const Element* __restrict__ tokens,
                            const int* __restrict__ slotToken, int H, int slots,
                            int* __restrict__ bad) {
  int slot = blockIdx.x;
  if (slot >= slots) return;
  const int t = slotToken[slot];
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    const Element got = packed[static_cast<size_t>(slot) * H + h];
    const Element expect = tokens[static_cast<size_t>(t) * H + h];
    if (__bfloat162float(got) != __bfloat162float(expect)) atomicAdd(bad, 1);
  }
}

// Row-major [batch, rows, cols] -> column-major [batch, rows, cols].
__global__ void rm2cm(const Element* __restrict__ src, Element* __restrict__ dst, int rows,
                      int cols) {
  constexpr int T = 32;
  __shared__ Element tile[T][T + 1];
  const size_t off = static_cast<size_t>(blockIdx.z) * rows * cols;
  const int r = blockIdx.y * T + threadIdx.y;
  const int c = blockIdx.x * T + threadIdx.x;
  if (r < rows && c < cols) {
    tile[threadIdx.y][threadIdx.x] = src[off + static_cast<size_t>(r) * cols + c];
  }
  __syncthreads();
  const int r2 = blockIdx.y * T + threadIdx.x;
  const int c2 = blockIdx.x * T + threadIdx.y;
  if (r2 < rows && c2 < cols) {
    dst[off + static_cast<size_t>(c2) * rows + r2] = tile[threadIdx.x][threadIdx.y];
  }
}

// Column-major -> row-major.
__global__ void cm2rm(const Element* __restrict__ src, Element* __restrict__ dst, int rows,
                      int cols) {
  constexpr int T = 32;
  __shared__ Element tile[T][T + 1];
  const size_t off = static_cast<size_t>(blockIdx.z) * rows * cols;
  const int r = blockIdx.y * T + threadIdx.x;
  const int c = blockIdx.x * T + threadIdx.y;
  if (r < rows && c < cols) {
    tile[threadIdx.y][threadIdx.x] = src[off + static_cast<size_t>(c) * rows + r];
  }
  __syncthreads();
  const int r2 = blockIdx.y * T + threadIdx.y;
  const int c2 = blockIdx.x * T + threadIdx.x;
  if (r2 < rows && c2 < cols) {
    dst[off + static_cast<size_t>(r2) * cols + c2] = tile[threadIdx.x][threadIdx.y];
  }
}

__global__ void checkCm(const Element* __restrict__ cm, const Element* __restrict__ tokens,
                        const int* __restrict__ rowToken, int rows, int cols, int batch,
                        int* __restrict__ bad) {
  const size_t n = static_cast<size_t>(batch) * rows * cols;
  for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const int b = static_cast<int>(idx / (static_cast<size_t>(rows) * cols));
    const int rem = static_cast<int>(idx % (static_cast<size_t>(rows) * cols));
    const int c = rem / rows;
    const int r = rem % rows;
    const int t = rowToken[b * rows + r];
    const Element expect = tokens[static_cast<size_t>(t) * cols + c];
    const Element got = cm[idx];
    if (__bfloat162float(got) != __bfloat162float(expect)) atomicAdd(bad, 1);
  }
}

// One block copies one expert row. Rows are contiguous in H in both layouts.
__global__ void reorderRows(const Element* __restrict__ src, Element* __restrict__ dst, int nLocal, int peers,
                            int mPeer, int H, int inverse) {
  const int row = static_cast<int>(blockIdx.x);
  const int rows = nLocal * peers * mPeer;
  if (row >= rows) return;
  const int slot = row % mPeer;
  const int tmp = row / mPeer;
  const int peer = tmp % peers;
  const int le = tmp / peers;
  const size_t leRow = static_cast<size_t>(row);
  const size_t peerRow = (static_cast<size_t>(peer) * nLocal + le) * mPeer + slot;
  const size_t from = (inverse ? leRow : peerRow) * H;
  const size_t to = (inverse ? peerRow : leRow) * H;
  const uint4* in = reinterpret_cast<const uint4*>(src + from);
  uint4* out = reinterpret_cast<uint4*>(dst + to);
  const int nvec = H >> 3;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) out[i] = in[i];
}

// idx walks the expert-major layout [le][peer][slot][h]. Used for the small index check.
template <typename T>
__global__ void reorderExperts(const T* __restrict__ src, T* __restrict__ dst, int nLocal,
                               int peers, int mPeer, int H, int inverse) {
  const size_t n = static_cast<size_t>(nLocal) * peers * mPeer * H;
  for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
    int h = static_cast<int>(idx % static_cast<size_t>(H));
    size_t t = idx / static_cast<size_t>(H);
    const int slot = static_cast<int>(t % static_cast<size_t>(mPeer));
    t /= static_cast<size_t>(mPeer);
    const int peer = static_cast<int>(t % static_cast<size_t>(peers));
    const int le = static_cast<int>(t / static_cast<size_t>(peers));
    const size_t peerMajor =
        ((static_cast<size_t>(peer) * nLocal + le) * mPeer + slot) * H + h;
    if (inverse) {
      dst[peerMajor] = src[idx];
    } else {
      dst[idx] = src[peerMajor];
    }
  }
}

__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }

// Eight contiguous bf16s per thread. Column-major groups share one bias column.
// Row-major groups are eight consecutive columns of one row.
__global__ void siluVec(const Element* __restrict__ gate, const Element* __restrict__ up,
                        const Element* __restrict__ biasG, const Element* __restrict__ biasU,
                        Element* __restrict__ out, int M, int Idim, int batch, int expertBase,
                        int rowMajor) {
  const size_t nvec = (static_cast<size_t>(batch) * M * Idim) >> 3;
  const size_t v = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v >= nvec) return;
  const size_t elem = v << 3;
  const size_t per = static_cast<size_t>(M) * Idim;
  const int e = static_cast<int>(elem / per);
  const int rem = static_cast<int>(elem % per);
  const uint4 g4 = reinterpret_cast<const uint4*>(gate)[v];
  const uint4 u4 = reinterpret_cast<const uint4*>(up)[v];
  const Element* g = reinterpret_cast<const Element*>(&g4);
  const Element* u = reinterpret_cast<const Element*>(&u4);
  uint4 packed;
  Element* dst = reinterpret_cast<Element*>(&packed);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int col = rowMajor ? (rem % Idim) + i : rem / M;
    const float gv = __bfloat162float(g[i]) + __bfloat162float(biasG[(expertBase + e) * Idim + col]);
    const float uv = __bfloat162float(u[i]) + __bfloat162float(biasU[(expertBase + e) * Idim + col]);
    dst[i] = __float2bfloat16(silu(gv) * uv);
  }
  reinterpret_cast<uint4*>(out)[v] = packed;
}

__global__ void addBiasVec(Element* __restrict__ y, const Element* __restrict__ bias, int M, int H,
                           int batch, int expertBase) {
  const size_t nvec = (static_cast<size_t>(batch) * M * H) >> 3;
  const size_t v = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v >= nvec) return;
  const size_t elem = v << 3;
  const int h = static_cast<int>(elem % static_cast<size_t>(H));
  const int e = static_cast<int>(elem / (static_cast<size_t>(M) * H));
  uint4 cur = reinterpret_cast<uint4*>(y)[v];
  Element* p = reinterpret_cast<Element*>(&cur);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float val = __bfloat162float(p[i]) + __bfloat162float(bias[(expertBase + e) * H + h + i]);
    p[i] = __float2bfloat16(val);
  }
  reinterpret_cast<uint4*>(y)[v] = cur;
}

void launchSilu(const Element* gate, const Element* up, const Element* biasG, const Element* biasU,
                Element* out, int M, int Idim, int batch, int expertBase, int rowMajor, cudaStream_t stream) {
  const size_t nvec = (static_cast<size_t>(batch) * M * Idim) >> 3;
  const int blocks = static_cast<int>((nvec + 255) / 256);
  siluVec<<<blocks, 256, 0, stream>>>(gate, up, biasG, biasU, out, M, Idim, batch, expertBase, rowMajor);
}

void launchBias(Element* y, const Element* bias, int M, int H, int batch, int expertBase, cudaStream_t stream) {
  const size_t nvec = (static_cast<size_t>(batch) * M * H) >> 3;
  const int blocks = static_cast<int>((nvec + 255) / 256);
  addBiasVec<<<blocks, 256, 0, stream>>>(y, bias, M, H, batch, expertBase);
}

__global__ void combineRows(const Element* __restrict__ expertRm, const int* __restrict__ tokenExpert,
                            const int* __restrict__ tokenSlot, Element* __restrict__ out, int H, int k,
                            int mPeer, float prob) {
  const int t = blockIdx.x;
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < k; ++j) {
      const int e = tokenExpert[t * k + j];
      const int s = tokenSlot[t * k + j];
      const size_t index = (static_cast<size_t>(e) * mPeer + s) * H + h;
      acc += prob * __bfloat162float(expertRm[index]);
    }
    out[static_cast<size_t>(t) * H + h] = __float2bfloat16(acc);
  }
}

struct LtGemm {
  cublasLtHandle_t lt{};
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t aDesc{};
  cublasLtMatrixLayout_t bDesc{};
  cublasLtMatrixLayout_t cDesc{};
  cublasLtMatmulAlgo_t algos[16]{};
  int nAlgos = 0;
  int best = 0;
  bool rowMajor = false;
  void* workspace = nullptr;
  size_t workspaceBytes = 256ull << 20;
};

void ltCreate(LtGemm& g, int M, int N, int K, int batch, bool rowMajor = false) {
  g.rowMajor = rowMajor;
  CHECK_CUBLAS(cublasLtCreate(&g.lt));
  CHECK_CUBLAS(cublasLtMatmulDescCreate(&g.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t trans = CUBLAS_OP_N;
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(g.op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(g.op, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));
  // Row-major C[M,N] = A[M,K] * B[K,N] is the column-major product B^T * A^T.
  const int aRows = rowMajor ? N : M;
  const int aCols = rowMajor ? K : K;
  const int aLd = rowMajor ? N : M;
  const int bRows = rowMajor ? K : K;
  const int bCols = rowMajor ? M : N;
  const int bLd = rowMajor ? K : K;
  const int cRows = rowMajor ? N : M;
  const int cCols = rowMajor ? M : N;
  const int cLd = rowMajor ? N : M;
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&g.aDesc, CUDA_R_16BF, aRows, aCols, aLd));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&g.bDesc, CUDA_R_16BF, bRows, bCols, bLd));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&g.cDesc, CUDA_R_16BF, cRows, cCols, cLd));
  const int64_t strideA = static_cast<int64_t>(aRows) * aCols;
  const int64_t strideB = static_cast<int64_t>(bRows) * bCols;
  const int64_t strideC = static_cast<int64_t>(cRows) * cCols;
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
  cublasLtMatmulHeuristicResult_t found[16]{};
  int n = 0;
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(g.lt, g.op, g.aDesc, g.bDesc, g.cDesc, g.cDesc, pref, 16, found, &n));
  cublasLtMatmulPreferenceDestroy(pref);
  if (n < 1) {
    fprintf(stderr, "no cublasLt algo for M=%d N=%d K=%d batch=%d\n", M, N, K, batch);
    exit(1);
  }
  g.nAlgos = n;
  for (int i = 0; i < n; ++i) g.algos[i] = found[i].algo;
  CHECK_CUDA(cudaMalloc(&g.workspace, g.workspaceBytes));
}

void ltDestroy(LtGemm& g) {
  if (g.workspace) cudaFree(g.workspace);
  if (g.aDesc) cublasLtMatrixLayoutDestroy(g.aDesc);
  if (g.bDesc) cublasLtMatrixLayoutDestroy(g.bDesc);
  if (g.cDesc) cublasLtMatrixLayoutDestroy(g.cDesc);
  if (g.op) cublasLtMatmulDescDestroy(g.op);
  if (g.lt) cublasLtDestroy(g.lt);
  g = {};
}

cublasStatus_t ltRun(LtGemm& g, const Element* act, const Element* w, Element* c, cudaStream_t stream,
                     int algo) {
  const float alpha = 1.f;
  const float beta = 0.f;
  const Element* A = g.rowMajor ? w : act;
  const Element* B = g.rowMajor ? act : w;
  return cublasLtMatmul(g.lt, g.op, &alpha, A, g.aDesc, B, g.bDesc, &beta, c, g.cDesc, c, g.cDesc,
                         &g.algos[algo], g.workspace, g.workspaceBytes, stream);
}

void ltMust(LtGemm& g, const Element* act, const Element* w, Element* c, cudaStream_t stream) {
  CHECK_CUBLAS(ltRun(g, act, w, c, stream, g.best));
}

template <typename Fn>
float timeLoop(cudaEvent_t start, cudaEvent_t stop, cudaStream_t stream, int warmup, int runs, Fn&& body) {
  for (int i = 0; i < warmup; ++i) body();
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < runs; ++i) body();
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  return ms / static_cast<float>(runs);
}

float hostExpected(int H, int I, int k) {
  const float a = __bfloat162float(__float2bfloat16(kToken));
  const float b = __bfloat162float(__float2bfloat16(kWeight));
  const float bias = __bfloat162float(__float2bfloat16(kBias));
  const float prod = a * b;
  float dotH = 0.f;
  for (int i = 0; i < H; ++i) dotH += prod;
  const float g = __bfloat162float(__float2bfloat16(dotH + bias));
  const float u = g;
  const float s = g / (1.f + expf(-g));
  const float act = __bfloat162float(__float2bfloat16(s * u));
  const float prodI = act * b;
  float dotI = 0.f;
  for (int i = 0; i < I; ++i) dotI += prodI;
  const float downGemm = __bfloat162float(__float2bfloat16(dotI));
  const float down = __bfloat162float(__float2bfloat16(downGemm + bias));
  const float prob = 1.f / static_cast<float>(k);
  float acc = 0.f;
  for (int j = 0; j < k; ++j) acc += prob * down;
  return acc;
}

struct Shape {
  int S, H, I, E, k;
};

struct Report {
  float total = -1.f;
  float pack = -1.f;
  float comm = -1.f;
  float gemm = -1.f;
  float epilogue = -1.f;
  float combine = -1.f;
  float overlap = -1.f;
  float graph = -1.f;
  float err = -1.f;
  int algos = 0;
};

void launchFill(Element* p, size_t n, float v) {
  fillConst<<<static_cast<int>((n + 255) / 256), 256>>>(p, n, v);
}

void launchTranspose(bool toCol, const Element* src, Element* dst, int rows, int cols, int batch,
                     cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((cols + 31) / 32, (rows + 31) / 32, batch);
  if (toCol) {
    rm2cm<<<grid, block, 0, stream>>>(src, dst, rows, cols);
  } else {
    cm2rm<<<grid, block, 0, stream>>>(src, dst, rows, cols);
  }
}

Report runShape(const Shape& shape, int warmup, int runs, cudaStream_t stream, cudaStream_t side) {
  const int S = shape.S;
  const int H = shape.H;
  const int I = shape.I;
  const int E = shape.E;
  const int k = shape.k;
  if (E % worldSize != 0 || (static_cast<long long>(S) * k) % E != 0) {
    fprintf(stderr, "rank %d: E must divide world and S*k\n", rankId);
    exit(1);
  }
  if (H % 8 != 0 || I % 8 != 0) {
    fprintf(stderr, "H and I must be multiples of 8\n");
    exit(1);
  }
  const int nLocal = E / worldSize;
  const int mPeer = S * k / E;
  const int M = worldSize * mPeer;
  if (M % 8 != 0) {
    fprintf(stderr, "M must be a multiple of 8\n");
    exit(1);
  }
  const int expertBase = rankId * nLocal;
  const float prob = 1.f / static_cast<float>(k);
  const double flops = 6.0 * static_cast<double>(nLocal) * M * static_cast<double>(I) * H;

  std::vector<int> slotToken(static_cast<size_t>(E) * mPeer);
  std::vector<int> tokenExpert(static_cast<size_t>(S) * k);
  std::vector<int> tokenSlot(static_cast<size_t>(S) * k);
  std::vector<int> next(E, 0);
  for (int t = 0; t < S; ++t) {
    for (int j = 0; j < k; ++j) {
      const int e = (t * k + j) % E;
      const int slot = next[e]++;
      slotToken[static_cast<size_t>(e) * mPeer + slot] = t;
      tokenExpert[static_cast<size_t>(t) * k + j] = e;
      tokenSlot[static_cast<size_t>(t) * k + j] = slot;
    }
  }

  auto alloc = [](size_t n) {
    Element* p = nullptr;
    CHECK_CUDA(cudaMalloc(&p, n * sizeof(Element)));
    return p;
  };
  const size_t tokenN = static_cast<size_t>(S) * H;
  const size_t weightN = static_cast<size_t>(E) * H * I;
  const size_t sendN = static_cast<size_t>(E) * mPeer * H;
  const size_t localActN = static_cast<size_t>(nLocal) * M * H;
  const size_t localMidN = static_cast<size_t>(nLocal) * M * I;

  Element* tokens = alloc(tokenN);
  Element* wGate = alloc(weightN);
  Element* wUp = alloc(weightN);
  Element* wDown = alloc(weightN);
  Element* biasG = alloc(static_cast<size_t>(E) * I);
  Element* biasU = alloc(static_cast<size_t>(E) * I);
  Element* biasD = alloc(static_cast<size_t>(E) * H);
  Element* packed = alloc(sendN);
  Element* gatheredRm = alloc(localActN);
  Element* actCm = alloc(localActN);
  Element* cGate = alloc(localMidN);
  Element* cUp = alloc(localMidN);
  Element* activated = alloc(localMidN);
  Element* cDown = alloc(localActN);
  Element* downRm = alloc(localActN);
  Element* returned = worldSize > 1 ? alloc(sendN) : downRm;
  Element* out = alloc(tokenN);
  int* dSlot = nullptr;
  int* dExpert = nullptr;
  int* dTokSlot = nullptr;
  CHECK_CUDA(cudaMalloc(&dSlot, slotToken.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dExpert, tokenExpert.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dTokSlot, tokenSlot.size() * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(dSlot, slotToken.data(), slotToken.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dExpert, tokenExpert.data(), tokenExpert.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dTokSlot, tokenSlot.data(), tokenSlot.size() * sizeof(int), cudaMemcpyHostToDevice));

  launchFill(wGate, weightN, kWeight);
  launchFill(wUp, weightN, kWeight);
  launchFill(wDown, weightN, kWeight);
  launchFill(biasG, static_cast<size_t>(E) * I, kBias);
  launchFill(biasU, static_cast<size_t>(E) * I, kBias);
  launchFill(biasD, static_cast<size_t>(E) * H, kBias);
  CHECK_CUDA(cudaDeviceSynchronize());

  const Element* localGate = wGate + static_cast<size_t>(expertBase) * H * I;
  const Element* localUp = wUp + static_cast<size_t>(expertBase) * H * I;
  const Element* localDown = wDown + static_cast<size_t>(expertBase) * I * H;

  auto enqueuePack = [&](cudaStream_t st) {
    packRows<<<E * mPeer, 128, 0, st>>>(tokens, dSlot, packed, H, E * mPeer);
  };
  // Leaves expert activations in actCm, column-major, one matrix per local expert.
  // downRm is scratch for the peer/expert reorder and is rewritten by the epilogue.
  auto enqueueActivations = [&](cudaStream_t st) {
    if (worldSize == 1) {
      launchTranspose(true, packed, actCm, M, H, nLocal, st);
      return;
    }
#ifdef FAST_MOE_DISTRIBUTED
    const size_t count = static_cast<size_t>(nLocal) * mPeer * H;
    exchange(packed, gatheredRm, count, sizeof(Element), ncclBfloat16, st);
    reorderRows<<<nLocal * worldSize * mPeer, 128, 0, st>>>(gatheredRm, downRm, nLocal, worldSize, mPeer, H, 0);
    launchTranspose(true, downRm, actCm, M, H, nLocal, st);
#else
    (void)st;
#endif
  };

  LtGemm gateLt{};
  LtGemm upLt{};
  LtGemm downLt{};
  // Build a real column-major activation once so autotune hits the timed path.
  fillPattern<<<S, 128, 0, stream>>>(tokens, S, H, rankId);
  enqueuePack(stream);
  if (worldSize == 1) {
    int* bad = nullptr;
    CHECK_CUDA(cudaMalloc(&bad, sizeof(int)));
    CHECK_CUDA(cudaMemset(bad, 0, sizeof(int)));
    checkPacked<<<E * mPeer, 128, 0, stream>>>(packed, tokens, dSlot, H, E * mPeer, bad);
    launchTranspose(true, packed, actCm, M, H, nLocal, stream);
    checkCm<<<256, 256, 0, stream>>>(actCm, tokens, dSlot, M, H, nLocal, bad);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    int hbad = 0;
    CHECK_CUDA(cudaMemcpy(&hbad, bad, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(bad));
    if (hbad != 0) {
      fprintf(stderr, "pack/transpose mismatches %d\n", hbad);
      exit(1);
    }
  }
#ifdef FAST_MOE_DISTRIBUTED
  if (worldSize > 1) {
    int* ids = nullptr;
    int* idsRecv = nullptr;
    int* idsReordered = nullptr;
    const size_t idCount = static_cast<size_t>(E) * mPeer;
    CHECK_CUDA(cudaMalloc(&ids, idCount * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&idsRecv, idCount * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&idsReordered, static_cast<size_t>(nLocal) * M * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(ids, slotToken.data(), idCount * sizeof(int), cudaMemcpyHostToDevice));
    const size_t count = static_cast<size_t>(nLocal) * mPeer;
    exchange(ids, idsRecv, count, sizeof(int), ncclInt32, stream);
    reorderExperts<int><<<256, 256, 0, stream>>>(idsRecv, idsReordered, nLocal, worldSize, mPeer, 1, 0);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<int> got(static_cast<size_t>(nLocal) * M);
    CHECK_CUDA(cudaMemcpy(got.data(), idsReordered, got.size() * sizeof(int), cudaMemcpyDeviceToHost));
    int mismatches = 0;
    for (int src = 0; src < worldSize; ++src) {
      std::vector<int> cursor(E, 0);
      for (int t = 0; t < S; ++t) {
        for (int j = 0; j < k; ++j) {
          const int e = (t * k + j) % E;
          const int slot = cursor[e]++;
          if (e / nLocal != rankId) continue;
          const int le = e % nLocal;
          const int row = (le * worldSize + src) * mPeer + slot;
          if (got[row] != t) ++mismatches;
        }
      }
    }
    CHECK_CUDA(cudaFree(ids));
    CHECK_CUDA(cudaFree(idsRecv));
    CHECK_CUDA(cudaFree(idsReordered));
    if (mismatches != 0) {
      fprintf(stderr, "rank %d routing exchange mismatches %d\n", rankId, mismatches);
      exit(1);
    }
  }
#endif

  launchFill(tokens, tokenN, kToken);
  CHECK_CUDA(cudaDeviceSynchronize());
  enqueuePack(stream);
  enqueueActivations(stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  ltCreate(gateLt, M, I, H, nLocal);
  ltCreate(upLt, M, I, H, nLocal);
  ltCreate(downLt, M, H, I, nLocal);

  cudaEvent_t ev0{};
  cudaEvent_t ev1{};
  CHECK_CUDA(cudaEventCreate(&ev0));
  CHECK_CUDA(cudaEventCreate(&ev1));
  auto tuneReal = [&](LtGemm& g, const Element* act, const Element* w, Element* c) {
    float bestMs = 1.0e30f;
    int best = 0;
    int accepted = 0;
    for (int algo = 0; algo < g.nAlgos; ++algo) {
      if (ltRun(g, act, w, c, stream, algo) != CUBLAS_STATUS_SUCCESS) continue;
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        cudaGetLastError();
        continue;
      }
      for (int i = 0; i < 2; ++i) {
        if (ltRun(g, act, w, c, stream, algo) != CUBLAS_STATUS_SUCCESS) {
          accepted = -1;
          break;
        }
      }
      if (accepted < 0) {
        cudaGetLastError();
        accepted = 0;
        continue;
      }
      CHECK_CUDA(cudaEventRecord(ev0, stream));
      for (int i = 0; i < 3; ++i) CHECK_CUBLAS(ltRun(g, act, w, c, stream, algo));
      CHECK_CUDA(cudaEventRecord(ev1, stream));
      CHECK_CUDA(cudaEventSynchronize(ev1));
      float ms = 0.f;
      CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
      ms /= 3.f;
      ++accepted;
      if (ms < bestMs) {
        bestMs = ms;
        best = algo;
      }
    }
    if (accepted == 0) {
      fprintf(stderr, "every cublasLt algo failed\n");
      exit(1);
    }
    g.best = best;
    return bestMs;
  };
  const float gateMs = tuneReal(gateLt, actCm, localGate, cGate);
  const float upMs = tuneReal(upLt, actCm, localUp, cUp);
  launchSilu(cGate, cUp, biasG, biasU, activated, M, I, nLocal, expertBase, 0, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  const float downMs = tuneReal(downLt, activated, localDown, cDown);
  if (rankId == 0) {
    printf("tune gate %.4f up %.4f down %.4f ms  algos %d %d %d  best %d %d %d\n", gateMs, upMs, downMs,
           gateLt.nAlgos, upLt.nAlgos, downLt.nAlgos, gateLt.best, upLt.best, downLt.best);
    fflush(stdout);
  }

  auto enqueueEpilogue = [&](cudaStream_t st) {
    launchSilu(cGate, cUp, biasG, biasU, activated, M, I, nLocal, expertBase, 0, st);
    ltMust(downLt, activated, localDown, cDown, st);
    launchTranspose(false, cDown, downRm, M, H, nLocal, st);
    launchBias(downRm, biasD, M, H, nLocal, expertBase, st);
  };
  auto enqueueReturn = [&](cudaStream_t st) {
#ifdef FAST_MOE_DISTRIBUTED
    if (worldSize > 1) {
      reorderRows<<<nLocal * worldSize * mPeer, 128, 0, st>>>(downRm, packed, nLocal, worldSize, mPeer, H, 1);
      const size_t count = static_cast<size_t>(nLocal) * mPeer * H;
      exchange(packed, returned, count, sizeof(Element), ncclBfloat16, st);
    }
#else
    (void)st;
#endif
  };
  auto enqueueCombine = [&](cudaStream_t st) {
    const Element* src = worldSize > 1 ? returned : downRm;
    combineRows<<<S, 256, 0, st>>>(src, dExpert, dTokSlot, out, H, k, mPeer, prob);
  };
  auto forward = [&](cudaStream_t st) {
    enqueuePack(st);
    enqueueActivations(st);
    ltMust(gateLt, actCm, localGate, cGate, st);
    ltMust(upLt, actCm, localUp, cUp, st);
    enqueueEpilogue(st);
    enqueueReturn(st);
    enqueueCombine(st);
  };

  // Correctness on the constant fill.
  forward(stream);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamSynchronize(stream));
  const float expect = hostExpected(H, I, k);
  std::vector<Element> hostOut(tokenN);
  CHECK_CUDA(cudaMemcpy(hostOut.data(), out, tokenN * sizeof(Element), cudaMemcpyDeviceToHost));
  double maxAbs = 0.0;
  for (size_t i = 0; i < hostOut.size(); i += std::max<size_t>(1, hostOut.size() / 4096)) {
    const double got = __bfloat162float(hostOut[i]);
    maxAbs = std::max(maxAbs, std::abs(got - static_cast<double>(expect)));
  }
  // Denser check on the first row, where a routing bug shows up first.
  for (int h = 0; h < H; h += 16) {
    const double got = __bfloat162float(hostOut[h]);
    maxAbs = std::max(maxAbs, std::abs(got - static_cast<double>(expect)));
  }
  const double rel = maxAbs / std::max(1.0e-6, std::abs(static_cast<double>(expect)));
  if (rankId == 0) {
    printf("check expect %.6f got0 %.6f maxAbs %.5f rel %.5f\n", expect, __bfloat162float(hostOut[0]), maxAbs, rel);
    fflush(stdout);
  }
  if (rel > 0.03) {
    fprintf(stderr, "rank %d correctness rel %.4f\n", rankId, rel);
    exit(1);
  }

  Report report;
  report.err = static_cast<float>(rel);
  report.algos = gateLt.nAlgos;
  report.total = timeLoop(ev0, ev1, stream, warmup, runs, [&] { forward(stream); });
    report.pack = timeLoop(ev0, ev1, stream, 1, runs, [&] {
    enqueuePack(stream);
    if (worldSize == 1) launchTranspose(true, packed, actCm, M, H, nLocal, stream);
  });
  report.gemm = timeLoop(ev0, ev1, stream, 1, runs, [&] {
    ltMust(gateLt, actCm, localGate, cGate, stream);
    ltMust(upLt, actCm, localUp, cUp, stream);
    ltMust(downLt, activated, localDown, cDown, stream);
  });
  report.epilogue = timeLoop(ev0, ev1, stream, 1, runs, [&] { enqueueEpilogue(stream); });
  report.combine = timeLoop(ev0, ev1, stream, 1, runs, [&] { enqueueCombine(stream); });
#ifdef FAST_MOE_DISTRIBUTED
  if (worldSize > 1) {
    report.comm = timeLoop(ev0, ev1, stream, 1, runs, [&] {
      const size_t count = static_cast<size_t>(nLocal) * mPeer * H;
      exchange(packed, returned, count, sizeof(Element), ncclBfloat16, stream);
    });
  }
#endif

  (void)side;
  auto show = report.total;
#ifdef FAST_MOE_DISTRIBUTED
  MPI_Allreduce(&report.total, &show, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
  report.total = show;
#endif
  if (rankId == 0) {
    const double tflops = flops / (static_cast<double>(report.total) * 1.0e9);
    const double toks = static_cast<double>(S) * worldSize / (static_cast<double>(report.total) * 1.0e-3);
    printf("fast,%d,%d,%d,%d,%d,%d,%.4f,%.1f,%.0f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f\n",
           worldSize, S, H, I, E, k, report.total, tflops, toks, report.pack, report.comm, report.gemm,
           report.epilogue, report.combine, report.overlap, report.graph, report.err);
    printf("stages pack %.4f comm %.4f gemm %.4f epilogue %.4f combine %.4f | overlap %.4f graph %.4f\n",
           report.pack, report.comm, report.gemm, report.epilogue, report.combine, report.overlap, report.graph);
    fflush(stdout);
  }

  // Row-major GEMMs read packed expert rows directly, so the two layout
  // transposes drop out of the forward.
  LtGemm gateR{};
  LtGemm upR{};
  LtGemm downR{};
  ltCreate(gateR, M, I, H, nLocal, true);
  ltCreate(upR, M, I, H, nLocal, true);
  ltCreate(downR, M, H, I, nLocal, true);
  auto stageRow = [&](cudaStream_t st) {
    enqueuePack(st);
#ifdef FAST_MOE_DISTRIBUTED
    if (worldSize > 1) {
      const size_t count = static_cast<size_t>(nLocal) * mPeer * H;
      exchange(packed, gatheredRm, count, sizeof(Element), ncclBfloat16, st);
      reorderRows<<<nLocal * worldSize * mPeer, 128, 0, st>>>(gatheredRm, actCm, nLocal, worldSize, mPeer, H, 0);
    }
#else
    (void)st;
#endif
  };
  stageRow(stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  const Element* rowAct = worldSize == 1 ? packed : actCm;
  const float rowGate = tuneReal(gateR, rowAct, localGate, cGate);
  const float rowUp = tuneReal(upR, rowAct, localUp, cUp);
  launchSilu(cGate, cUp, biasG, biasU, activated, M, I, nLocal, expertBase, 1, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  const float rowDown = tuneReal(downR, activated, localDown, downRm);
  if (rankId == 0) {
    printf("row-tune gate %.4f up %.4f down %.4f best %d %d %d\n", rowGate, rowUp, rowDown, gateR.best, upR.best,
           downR.best);
    fflush(stdout);
  }
  auto forwardRow = [&](cudaStream_t st) {
    stageRow(st);
    const Element* act = worldSize == 1 ? packed : actCm;
    ltMust(gateR, act, localGate, cGate, st);
    ltMust(upR, act, localUp, cUp, st);
    launchSilu(cGate, cUp, biasG, biasU, activated, M, I, nLocal, expertBase, 1, st);
    ltMust(downR, activated, localDown, downRm, st);
    launchBias(downRm, biasD, M, H, nLocal, expertBase, st);
    enqueueReturn(st);
    enqueueCombine(st);
  };
  forwardRow(stream);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaMemcpy(hostOut.data(), out, tokenN * sizeof(Element), cudaMemcpyDeviceToHost));
  double rowAbs = 0.0;
  for (int h = 0; h < H; h += 16) {
    rowAbs = std::max(rowAbs, std::abs(static_cast<double>(__bfloat162float(hostOut[h])) - expect));
  }
  const double rowRel = rowAbs / std::max(1.0e-6, std::abs(static_cast<double>(expect)));
  float rowMs = -1.f;
  if (rowRel <= 0.03) {
    rowMs = timeLoop(ev0, ev1, stream, warmup, runs, [&] { forwardRow(stream); });
#ifdef FAST_MOE_DISTRIBUTED
    float rowMax = rowMs;
    MPI_Allreduce(&rowMs, &rowMax, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    rowMs = rowMax;
#endif
  }
  if (rankId == 0) {
    const double rowTflops = rowMs > 0.f ? flops / (static_cast<double>(rowMs) * 1.0e9) : 0.0;
    const double rowToks = rowMs > 0.f ? static_cast<double>(S) * worldSize / (static_cast<double>(rowMs) * 1.0e-3) : 0.0;
    printf("fast-row,%d,%d,%d,%d,%d,%d,%.4f,%.1f,%.0f,rel %.5f\n", worldSize, S, H, I, E, k, rowMs, rowTflops, rowToks,
           rowRel);
    fflush(stdout);
  }
#ifdef FAST_MOE_DISTRIBUTED
  if (worldSize > 1) {
    Element* pk[2] = {packed, nullptr};
    Element* gv[2] = {gatheredRm, nullptr};
    Element* ac[2] = {actCm, nullptr};
    Element* cg[2] = {cGate, nullptr};
    Element* cu[2] = {cUp, nullptr};
    Element* mid[2] = {activated, nullptr};
    Element* dn[2] = {downRm, nullptr};
    Element* rt[2] = {returned, nullptr};
    pk[1] = alloc(sendN);
    gv[1] = alloc(localActN);
    ac[1] = alloc(localActN);
    cg[1] = alloc(localMidN);
    cu[1] = alloc(localMidN);
    mid[1] = alloc(localMidN);
    dn[1] = alloc(localActN);
    rt[1] = alloc(sendN);
    cudaEvent_t ready[2]{};
    cudaEvent_t done[2]{};
    CHECK_CUDA(cudaEventCreate(&ready[0]));
    CHECK_CUDA(cudaEventCreate(&ready[1]));
    CHECK_CUDA(cudaEventCreate(&done[0]));
    CHECK_CUDA(cudaEventCreate(&done[1]));
    const size_t count = static_cast<size_t>(nLocal) * mPeer * H;
    const int rowBlocks = nLocal * worldSize * mPeer;
    auto dispatchSlot = [&](int s, cudaStream_t st) {
      packRows<<<E * mPeer, 128, 0, st>>>(tokens, dSlot, pk[s], H, E * mPeer);
      exchange(pk[s], gv[s], count, sizeof(Element), ncclBfloat16, st);
      reorderRows<<<rowBlocks, 128, 0, st>>>(gv[s], ac[s], nLocal, worldSize, mPeer, H, 0);
    };
    auto computeSlot = [&](int s, cudaStream_t st) {
      ltMust(gateR, ac[s], localGate, cg[s], st);
      ltMust(upR, ac[s], localUp, cu[s], st);
      launchSilu(cg[s], cu[s], biasG, biasU, mid[s], M, I, nLocal, expertBase, 1, st);
      ltMust(downR, mid[s], localDown, dn[s], st);
      launchBias(dn[s], biasD, M, H, nLocal, expertBase, st);
    };
    auto finishSlot = [&](int s, cudaStream_t st) {
      reorderRows<<<rowBlocks, 128, 0, st>>>(dn[s], pk[s], nLocal, worldSize, mPeer, H, 1);
      exchange(pk[s], rt[s], count, sizeof(Element), ncclBfloat16, st);
      combineRows<<<S, 256, 0, st>>>(rt[s], dExpert, dTokSlot, out, H, k, mPeer, prob);
    };
    int phase = 0;
    dispatchSlot(0, side);
    CHECK_CUDA(cudaEventRecord(ready[0], side));
    auto step = [&]() {
      const int s = phase;
      const int nxt = phase ^ 1;
      CHECK_CUDA(cudaStreamWaitEvent(stream, ready[s], 0));
      computeSlot(s, stream);
      CHECK_CUDA(cudaEventRecord(done[s], stream));
      dispatchSlot(nxt, side);
      CHECK_CUDA(cudaEventRecord(ready[nxt], side));
      CHECK_CUDA(cudaStreamWaitEvent(side, done[s], 0));
      finishSlot(s, side);
      phase ^= 1;
    };
    for (int i = 0; i < 2; ++i) step();
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamSynchronize(side));
    CHECK_CUDA(cudaMemcpy(hostOut.data(), out, tokenN * sizeof(Element), cudaMemcpyDeviceToHost));
    double pipeAbs = 0.0;
    for (int h = 0; h < H; h += 16) {
      pipeAbs = std::max(pipeAbs, std::abs(static_cast<double>(__bfloat162float(hostOut[h])) - expect));
    }
    const double pipeRel = pipeAbs / std::max(1.0e-6, std::abs(static_cast<double>(expect)));
    float pipeMs = -1.f;
    if (pipeRel <= 0.03) {
      for (int i = 0; i < warmup; ++i) step();
      CHECK_CUDA(cudaStreamSynchronize(stream));
      CHECK_CUDA(cudaStreamSynchronize(side));
      CHECK_CUDA(cudaEventRecord(ev0, stream));
      CHECK_CUDA(cudaStreamWaitEvent(side, ev0, 0));
      for (int i = 0; i < runs; ++i) step();
      CHECK_CUDA(cudaEventRecord(ev1, side));
      CHECK_CUDA(cudaEventSynchronize(ev1));
      CHECK_CUDA(cudaEventElapsedTime(&pipeMs, ev0, ev1));
      pipeMs /= static_cast<float>(runs);
      float pipeMax = pipeMs;
      MPI_Allreduce(&pipeMs, &pipeMax, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
      pipeMs = pipeMax;
    }
    if (rankId == 0) {
      const double pipeTflops = pipeMs > 0.f ? flops / (static_cast<double>(pipeMs) * 1.0e9) : 0.0;
      const double pipeToks =
          pipeMs > 0.f ? static_cast<double>(S) * worldSize / (static_cast<double>(pipeMs) * 1.0e-3) : 0.0;
      printf("fast-pipe,%d,%d,%d,%d,%d,%d,%.4f,%.1f,%.0f,rel %.5f\n", worldSize, S, H, I, E, k, pipeMs, pipeTflops,
             pipeToks, pipeRel);
      fflush(stdout);
    }
    CHECK_CUDA(cudaEventDestroy(ready[0]));
    CHECK_CUDA(cudaEventDestroy(ready[1]));
    CHECK_CUDA(cudaEventDestroy(done[0]));
    CHECK_CUDA(cudaEventDestroy(done[1]));
    for (int s = 1; s < 2; ++s) {
      cudaFree(pk[s]);
      cudaFree(gv[s]);
      cudaFree(ac[s]);
      cudaFree(cg[s]);
      cudaFree(cu[s]);
      cudaFree(mid[s]);
      cudaFree(dn[s]);
      cudaFree(rt[s]);
    }
  }
#endif
  ltDestroy(gateR);
  ltDestroy(upR);
  ltDestroy(downR);

  CHECK_CUDA(cudaEventDestroy(ev0));
  CHECK_CUDA(cudaEventDestroy(ev1));
  ltDestroy(gateLt);
  ltDestroy(upLt);
  ltDestroy(downLt);
  auto freed = [](Element*& p) {
    if (p) cudaFree(p);
    p = nullptr;
  };
  freed(tokens);
  freed(wGate);
  freed(wUp);
  freed(wDown);
  freed(biasG);
  freed(biasU);
  freed(biasD);
  freed(packed);
  freed(gatheredRm);
  freed(actCm);
  freed(cGate);
  freed(cUp);
  freed(activated);
  freed(cDown);
  freed(downRm);
  if (worldSize > 1) freed(returned);
  freed(out);
  cudaFree(dSlot);
  cudaFree(dExpert);
  cudaFree(dTokSlot);
  return report;
}

int main(int argc, char** argv) {
#ifdef FAST_MOE_DISTRIBUTED
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  int visible = 0;
  CHECK_CUDA(cudaGetDeviceCount(&visible));
  CHECK_CUDA(cudaSetDevice(rankId % visible));
  ncclUniqueId id{};
  if (rankId == 0) CHECK_NCCL(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  CHECK_NCCL(ncclCommInitRank(&ncclComm, worldSize, id, rankId));
#else
  (void)argc;
#endif
  int dev = 0;
  CHECK_CUDA(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  if (rankId == 0) {
    printf("device %s sms %d world %d\n", prop.name, prop.multiProcessorCount, worldSize);
    printf("csv,world,S,H,I,E,k,ms,tflops,tokens_per_s,pack_ms,comm_ms,gemm_ms,epilogue_ms,combine_ms,overlap_ms,graph_ms,rel_err\n");
    fflush(stdout);
  }

  int warmup = 2;
  int runs = 8;
  std::vector<Shape> shapes;
  if (argc > 5) {
    shapes.push_back(Shape{std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]),
                           std::atoi(argv[5])});
    if (argc > 6) warmup = std::atoi(argv[6]);
    if (argc > 7) runs = std::atoi(argv[7]);
  } else {
    shapes.push_back(Shape{4096, 2048, 1408, 16, 6});
    shapes.push_back(Shape{4096, 2048, 4096, 16, 6});
    shapes.push_back(Shape{16384, 2048, 1408, 16, 6});
    shapes.push_back(Shape{4096, 2048, 1408, 64, 6});
    shapes.push_back(Shape{4096, 2048, 2048, 32, 2});
  }

  cudaStream_t stream{};
  cudaStream_t side{};
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUDA(cudaStreamCreate(&side));
  for (const Shape& shape : shapes) {
    if (rankId == 0) {
      printf("=== S %d H %d I %d E %d k %d ===\n", shape.S, shape.H, shape.I, shape.E, shape.k);
      fflush(stdout);
    }
    runShape(shape, warmup, runs, stream, side);
  }
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaStreamDestroy(side));
#ifdef FAST_MOE_DISTRIBUTED
  ncclCommDestroy(ncclComm);
  MPI_Finalize();
#endif
  return 0;
}
