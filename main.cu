#include <cassert>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <mma.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <algorithm>
#include <cooperative_groups.h>
#include <cuda/barrier>

#include "BCSRMatrix.cuh"
#include "CSRMatrix.cuh"
#include "HCSRMatrix.h"
#include "Matrix.cuh"
#include "miscutil.h"

unsigned int N = 0;
constexpr unsigned int N_THREADS = 32;
extern const int BLOCK_SIZE = 16;
string MATRIX_A_PATH = "../medA.mat";
string MATRIX_B_PATH = "../medB.mat";
string MATRIX_PATTERN = "random";
// sparsity extracted from filename (A_<size>_s<sp>.mat), default empty
string MATRIX_SPARSITY = "";

using namespace std;
using namespace nvcuda;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define BLOCKSIZE 32
#define CEIL_DIV(_a, _b) ((_a) / (_b) + ((_a) % (_b) > 0 ? 1 : 0))
#define CHECK_CUDA_ERRORS(_where) \
    error = cudaGetLastError(); \
    if (error != cudaSuccess) \
        cout << _where << " CUDA " \
        "error: " << cudaGetErrorString(error) << '\n'; \
    assert(error == cudaSuccess);
#define BYTES_SIZE(T) (N * N * sizeof(T))
#define MALLOC_MATRIX(T) static_cast<T *>(malloc(BYTES_SIZE(T)));
#define ALLOC_GPU_MEM \
    cudaDeviceReset(); \
    CHECK_CUDA_ERRORS("cudaDeviceReset") \
    bcsrA->copyToDevice(&gpuBCSRHdr, &gpuBCSRIdx, &gpuBCSRData); \
    csrA->copyToDevice(&gpuCSRHdr, &gpuCSRIdx, &gpuCSRData); \
    cudaMalloc(reinterpret_cast<void **>(&gpuA_half), BYTES_SIZE(half)); \
    cudaMalloc(reinterpret_cast<void **>(&gpuB_half), BYTES_SIZE(half)); \
    cudaMalloc(reinterpret_cast<void **>(&gpuC), BYTES_SIZE(float)); \
    cudaMalloc(reinterpret_cast<void **>(&gpuCPart), BYTES_SIZE(float)); \
    cudaMemcpy(gpuA_half, matrixA->data, BYTES_SIZE(half), \
               cudaMemcpyHostToDevice); \
    cudaMemcpy(gpuB_half, matrixB->data, BYTES_SIZE(half), \
               cudaMemcpyHostToDevice); \
    CHECK_CUDA_ERRORS("cudaMemcpy")
#define PREPARE_FUNC(_name) \
    /* silent prepare: initialize buffers and events (no console output) */ \
    memset(memC, 0, BYTES_SIZE(float)); \
    cudaMemset(gpuC, 0, BYTES_SIZE(float)); \
    cudaMemset(gpuCPart, 0, BYTES_SIZE(float)); \
    cudaEventCreate(&t1); \
    cudaEventCreate(&t2); \
    cudaEventRecord(t1, 0);
#define END_FUNC(_name, ...) \
    cudaDeviceSynchronize(); \
    cudaEventRecord(t2, 0); \
    cudaEventSynchronize(t2); \
    cudaEventElapsedTime(&ms, t1, t2); \
    __VA_ARGS__ \
    cudaMemcpy(memC, gpuC, BYTES_SIZE(float), cudaMemcpyDeviceToHost); \
    /* Clear device result buffers so next run doesn't inherit previous values */ \
    if (gpuC) cudaMemset(gpuC, 0, BYTES_SIZE(float)); \
    if (gpuCPart) cudaMemset(gpuCPart, 0, BYTES_SIZE(float)); \
    cudaEventDestroy(t1); \
    cudaEventDestroy(t2); \
    /* CSV: name,time(ms),diff(maxdiff),maxrelativeerr(avgrelerr),size,pattern,sparsity,matrixApath */ \
    printf("%s,%f,%lf,%lf,%u,%s,%s,%s\n", _name, ms, maxdiff(memC, correctMatrix, N), avgrelerr(memC, correctMatrix, N), N, MATRIX_PATTERN.c_str(), MATRIX_SPARSITY.c_str(), MATRIX_A_PATH.c_str());

// Run the provided call 20 times, collect timings, print median and metrics.
#define RUN_20_AND_REPORT(_name, ...) do { \
    float __r20_times[20]; \
    for (int __r20_iter = 0; __r20_iter < 20; ++__r20_iter) { \
        /* prepare buffers and timing events per iteration */ \
        memset(memC, 0, BYTES_SIZE(float)); \
        cudaMemset(gpuC, 0, BYTES_SIZE(float)); \
        cudaMemset(gpuCPart, 0, BYTES_SIZE(float)); \
        cudaEventCreate(&t1); cudaEventCreate(&t2); \
        cudaEventRecord(t1, 0); \
        __VA_ARGS__; \
        cudaEventRecord(t2, 0); \
        cudaEventSynchronize(t2); \
        cudaEventElapsedTime(&ms, t1, t2); \
        __r20_times[__r20_iter] = ms; \
        cudaEventDestroy(t1); cudaEventDestroy(t2); \
    } \
    /* copy final result from device (from last iteration) */ \
    cudaMemcpy(memC, gpuC, BYTES_SIZE(float), cudaMemcpyDeviceToHost); \
    if (gpuC) cudaMemset(gpuC, 0, BYTES_SIZE(float)); \
    if (gpuCPart) cudaMemset(gpuCPart, 0, BYTES_SIZE(float)); \
    std::sort(__r20_times, __r20_times + 20); \
    float __r20_median = __r20_times[10]; \
    /* CSV: name,time(ms),diff(maxdiff),maxrelativeerr(avgrelerr),size,pattern,sparsity,matrixApath */ \
    printf("%s,%f,%lf,%lf,%u,%s,%s,%s\n", _name, __r20_median, maxdiff(memC, correctMatrix, N), avgrelerr(memC, correctMatrix, N), N, MATRIX_PATTERN.c_str(), MATRIX_SPARSITY.c_str(), MATRIX_A_PATH.c_str()); \
} while(0)

/**
 * Dense matrix multiplication in CPU
 */
float *matrixMulCPU(const half *A, const half *B, float *C) {
    memset(C, 0, sizeof(float) * N * N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i * N + j] += __half2float(A[i * N + k]) * __half2float(
                    B[k * N + j]);
            }
        }
    }
    return C;
}

// MATRIX MULTIPLICATION ALGORITHMS

/**
 * Dense matrix multiplication in GPU
 * // O(n) per thread
 */
__global__ void denseMatrixMul(const half *d_A, const half *d_B, float *d_C,
                               const unsigned int n) {
    const unsigned int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (rowIdx < n && colIdx < n) {
        float tmp = 0.0f;
        for (int k = 0; k < n; k++) {
            // Accumulate results for a single element
            // There's no need here to use reduction  or atomic add, because this
            // thread is the only one accessing this location
            tmp += __half2float(d_A[rowIdx * n + k]) *
                    __half2float(d_B[k * n + colIdx]);
        }
        d_C[rowIdx * n + colIdx] = tmp;
    }
}

/**
 * Dense matrix multiplication in GPU with memory coalescence
 * // O(n) per thread
 */
__global__ void denseMatrixMulCo(const half *d_A, const half *d_B, float *d_C,
                                 const unsigned int n) {
    const unsigned int rowIdx = blockIdx.y *
        CEIL_DIV(n, gridDim.y) + threadIdx.x / n;
    const unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x % n;

    if (rowIdx < n && colIdx < n) {
        float tmp = 0.0f;
        for (int k = 0; k < n; k++) {
            tmp += __half2float(d_A[rowIdx * n + k]) * __half2float(
                d_B[k * n + colIdx]);
        }
        d_C[rowIdx * n + colIdx] = tmp;
    }
}

/**
 * Multiply two dense matrices using tensors wmma
 */

__global__ void denseMatrixMulTensor(const half *d_A, const half *d_B,
                                     float *d_C, const unsigned int n) {
    // Calculate which 16x16 tile this thread block handles
    const unsigned int warp_row = blockIdx.y * 16;
    const unsigned int warp_col = blockIdx.x * 16;

    if (warp_row >= n || warp_col >= n) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Accumulate over K dimension in 16x16 chunks
    for (int k = 0; k < n; k += 16) {
        wmma::load_matrix_sync(a_frag, d_A + warp_row * n + k, n);
        wmma::load_matrix_sync(b_frag, d_B + k * n + warp_col, n);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(d_C + warp_row * n + warp_col, c_frag, n,
                            wmma::mem_row_major);
}

/**
 * Multiply a CSR matrix x a dense matrix
 * C must be initialized and filled with 0s
 *
 * O(R) R = non zeroes in this row
 */
__global__ void sparseMatrixMult1Co(const int *hdr, const int *idx,
                                    const half *data, const half *B, float *C,
                                    const unsigned int n) {
    const unsigned int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (rowIdx < n && colIdx < n) {
        float tmp = 0.0f;
        for (int k = hdr[rowIdx]; k < hdr[rowIdx + 1]; k++) {
            tmp += __half2float(data[k]) * __half2float(
                B[idx[k] * n + colIdx]);
        }
        C[rowIdx * n + colIdx] = tmp;
    }
}

/**
 * Multiply a CSR matrix x a dense matrix
 * C must be initialized and filled with 0s
 *
 * O(R) R = non zeroes in this row
 */
__global__ void sparseMatrixMult1(const int *hdr, const int *idx,
                                  const half *data, const half *B, float *C,
                                  const unsigned int n) {
    const unsigned int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (rowIdx < n && colIdx < n) {
        for (int k = hdr[rowIdx]; k < hdr[rowIdx + 1]; k++) {
            C[rowIdx * n + colIdx] += __half2float(data[k]) * __half2float(
                B[idx[k] * n + colIdx]);
        }
    }
}

/**
 * Multiply a CSR matrix x a dense matrix
 * C must be initialized and filled with 0s
 */
__global__ void sparseMatrixMult2(const int *hdr, const int *idx,
                                  const half *data, const half *B, float *C,
                                  const unsigned int n) {
    const unsigned int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < n) {
        int i = 0;
        for (int row = 0; row < n; row++) {
            for (; i < hdr[row + 1]; i++) {
                atomicAdd(&C[row * n + k],
                          __half2float(data[i]) * __half2float(
                              B[idx[i] * n + k]));
            }
        }
    }
}

/**
 * Multiply a CSR matrix x a dense matrix
 * C must be initialized and filled with 0s
 */
__global__ void sparseMatrixMult3(const int *hdr, const int *idx,
                                  const half *data, const half *B, float *C,
                                  const unsigned int n) {
    const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < hdr[n]) {
        int row = 0;
        while (row < n && i >= hdr[row + 1]) row++;

        for (int k = 0; k < n; k++) {
            atomicAdd(&C[row * n + k],
                      __half2float(data[i]) * __half2float(B[idx[i] * n + k]));
        }
    }
}

/**
 * Multiply a BCSR matrix and a dense matrix using tensors
 */
__global__ void sparseMatrixMulTensor1(const int *hdr, const int *idx,
                                      const half *data, const half *B,
                                      float *C, const unsigned int n) {
    const unsigned int warpRow = blockIdx.y * 16;
    const unsigned int warpCol = blockIdx.x * 16;

    if (warpRow >= n || warpCol >= n) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
    for (int k = hdr[warpRow / 16]; k < hdr[warpRow / 16 + 1]; k++) {
        wmma::load_matrix_sync(a_frag, data + k * 16 * 16, 16);
        wmma::load_matrix_sync(b_frag, B + idx[k] * 16 * n + warpCol, n);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warpRow * n + warpCol, c_frag, n,
                            wmma::mem_row_major);
}

/**
 * Multiply a BCSR matrix and a dense matrix using tensors
 */
__global__ void sparseMatrixMulTensor(const int *hdr, const int *idx,
                                      const half *data, const half *B,
                                      float *C, const unsigned int n) {
    const unsigned int warpRow = blockIdx.y * 16;
    const unsigned int warpCol = blockIdx.x * 16;

    if (warpRow >= n || warpCol >= n) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = hdr[warpRow / 16]; k < hdr[warpRow / 16 + 1]; k++) {
        wmma::load_matrix_sync(a_frag, data + k * 16 * 16, 16);
        wmma::load_matrix_sync(b_frag, B + idx[k] * 16 * n + warpCol, n);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warpRow * n + warpCol, c_frag, n,
                            wmma::mem_row_major);
}

__global__ void addMatrices(float *C, const float *CPart, const unsigned int n) {
    const unsigned int rowIdx = blockIdx.y *
        CEIL_DIV(n, gridDim.y) + threadIdx.x / n;
    const unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x % n;

    if (rowIdx < n && colIdx < n) {
        C[rowIdx * n + colIdx] += CPart[rowIdx * n + colIdx];
    }
}

int main(const int argc, const char **argv) {
    if (argc >= 2) MATRIX_A_PATH = argv[1];
    if (argc >= 3) MATRIX_B_PATH = argv[2];
    if (argc >= 4) MATRIX_PATTERN = argv[3];

    const Matrix *matrixA = new Matrix(MATRIX_A_PATH);
    const Matrix *matrixB = new Matrix(MATRIX_B_PATH);
    assert(matrixA->rows && matrixA->cols && matrixB->rows && matrixB->cols);
    assert(matrixA->cols == matrixB->rows);
    N = matrixA->cols;

    // compute true sparsity = nonZeros / (N*N) from matrixA
    if (matrixA->rows > 0 && matrixA->cols > 0) {
        double total = static_cast<double>(matrixA->rows) * static_cast<double>(matrixA->cols);
        double sp = 0.0;
        if (total > 0.0) sp = static_cast<double>(matrixA->nonZeros) / total;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4) << sp;
        MATRIX_SPARSITY = oss.str();
    }
    // allow optional override: argv[4] can set printed sparsity (e.g. 0.35)
    if (argc >= 5) {
        MATRIX_SPARSITY = string(argv[4]);
    }

    cublasHandle_t cublasHandle;
    constexpr float alpha = 1.0;
    constexpr float beta = 0.0;
    const int n = static_cast<int>(N);

    auto *memC = MALLOC_MATRIX(float);
    auto *correctMatrix = MALLOC_MATRIX(float);
    float *gpuC, *gpuCPart;
    half *gpuA_half, *gpuB_half, *gpuCSRData, *gpuBCSRData;
    half *gpuCSRDataPart, *gpuBCSRDataPart;
    int *gpuCSRHdr, *gpuCSRIdx, *gpuBCSRHdr, *gpuBCSRIdx;
    int *gpuCSRHdrPart, *gpuCSRIdxPart, *gpuBCSRHdrPart, *gpuBCSRIdxPart;
    cudaEvent_t t1, t2;
    float ms = 0.0f;
    dim3 gridSize, blockSize;
    cudaError_t error;

    const auto *csrA = new CSRMatrix(*matrixA);
    const auto *bcsrA = new BCSRMatrix(*matrixA);

    /* ========================== DENSE ON CPU ========================== */
#ifdef CHECK_CORRECTNESS
    PREPARE_FUNC("Dense on CPU");
    matrixMulCPU(matrixA->data, matrixB->data, correctMatrix);
    END_FUNC("Dense on CPU");
#endif

    /* ========================== DENSE ON GPU ========================== */
    /*gridSize = {
        N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0),
        N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0),
        1
    };
    blockSize = {N_THREADS, N_THREADS, 1};
    PREPARE_FUNC("Dense on GPU");
    denseMatrixMul<<<gridSize, blockSize>>>(gpuA_half, gpuB_half, gpuC, N);
    END_FUNC("Dense on GPU");
    // Use dense on GPU as correct function
    memcpy(correctMatrix, memC, N * N * sizeof(float));*/

    /* ================= DENSE ON GPU WITH COALESCENCE ================== */
    ALLOC_GPU_MEM
    gridSize = {
        CEIL_DIV(N, (N_THREADS * N_THREADS)),
        CEIL_DIV(N, CEIL_DIV(N_THREADS * N_THREADS, N)),
        1
    };
    blockSize = {N_THREADS * N_THREADS, 1, 1};
    RUN_20_AND_REPORT("Dense on GPU Coalescence", denseMatrixMulCo<<<gridSize, blockSize>>>(gpuA_half, gpuB_half, gpuC, N));
    memcpy(correctMatrix, memC, N * N * sizeof(float));

    /* ========================== DENSE WMMA ========================== */
    ALLOC_GPU_MEM
    gridSize = {N / 16, N / 16, 1};
    blockSize = {32, 1, 1};
    RUN_20_AND_REPORT("Dense WMMA", denseMatrixMulTensor<<<gridSize, blockSize>>>(gpuA_half, gpuB_half, gpuC, N));

    /* ========================== SpMM 1 Co ======================== */
    ALLOC_GPU_MEM
    gridSize = {
        N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0),
        N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0),
        1
    };
    blockSize = {N_THREADS, N_THREADS, 1};
    RUN_20_AND_REPORT("SpMM 1 Co", sparseMatrixMult1Co<<<gridSize, blockSize>>>(gpuCSRHdr, gpuCSRIdx, gpuCSRData, gpuB_half, gpuC, N));

    /* ========================== SpMM 1 ========================== */
    /*
    ALLOC_GPU_MEM
    gridSize = {
        N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0),
        N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0),
        1
    };
    blockSize = {N_THREADS, N_THREADS, 1};
    PREPARE_FUNC("SpMM 1");
    sparseMatrixMult1<<<gridSize, blockSize>>>(gpuCSRHdr, gpuCSRIdx,
                                               gpuCSRData, gpuB_half, gpuC, N);
    END_FUNC("SpMM 1");
    */

    /* ========================== SpMM 2 ========================== */
    /*
    ALLOC_GPU_MEM
    gridSize = {
        N / (N_THREADS * N_THREADS) + (N % (N_THREADS * N_THREADS) > 0 ? 1 : 0),
        1, 1
    };
    blockSize = {N_THREADS * N_THREADS, 1, 1};
    PREPARE_FUNC("SpMM 2");
    sparseMatrixMult2<<<gridSize, blockSize>>>(gpuCSRHdr, gpuCSRIdx,
                                               gpuCSRData, gpuB_half, gpuC, N);
    END_FUNC("SpMM 2");
    */

    /* ========================== SpMM 3 ========================== */
    /*
    ALLOC_GPU_MEM
    gridSize = {
        csrA->hdr[N] / (N_THREADS * N_THREADS) + (
            csrA->hdr[N] % (N_THREADS * N_THREADS) > 0 ? 1 : 0),
        1,
        1
    };
    blockSize = {N_THREADS * N_THREADS, 1, 1};
    PREPARE_FUNC("SpMM 3");
    sparseMatrixMult3<<<gridSize, blockSize>>>(gpuCSRHdr, gpuCSRIdx,
                                               gpuCSRData, gpuB_half, gpuC, N);
    END_FUNC("SpMM 3");
    */

    /* ========================= SpMM WITH TENSORS ========================= */
    ALLOC_GPU_MEM
    gridSize = {N / 16, N / 16, 1};
    blockSize = {32, 1, 1};
    RUN_20_AND_REPORT("SpMM with Tensors", sparseMatrixMulTensor<<<gridSize, blockSize>>>(gpuBCSRHdr, gpuBCSRIdx, gpuBCSRData, gpuB_half, gpuC, N));

    /* ==================== SpMM WITH TENSORS OPTIMIZED ==================== */
    ALLOC_GPU_MEM
    gridSize = {N / 16, N / 16, 1};
    blockSize = {32, 1, 1};
    RUN_20_AND_REPORT("SpMM with Tensors Op", sparseMatrixMulTensor1<<<gridSize, blockSize>>>(gpuBCSRHdr, gpuBCSRIdx, gpuBCSRData, gpuB_half, gpuC, N));

    /* ============================== CUBLAS =============================== */

    ALLOC_GPU_MEM
    cublasCreate(&cublasHandle);

    RUN_20_AND_REPORT("cuBLAS GeMM", cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, gpuB_half, CUDA_R_16F, n, gpuA_half, CUDA_R_16F, n, &beta, gpuC, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    cublasDestroy(cublasHandle);

    /* ============================== CUBLAS WITH TENSORS =============================== */

    ALLOC_GPU_MEM
    cublasCreate(&cublasHandle);

    // Enable tensor op math for this handle and do a warm-up GEMM to avoid
    // JIT/cold-start overhead and ensure tensor-core kernels are resident.
    cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
    /* Warm-up call (single, not timed) to prime cuBLAS / kernels */
    cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, gpuB_half, CUDA_R_16F, n, gpuA_half, CUDA_R_16F, n, &beta, gpuC, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    RUN_20_AND_REPORT("cuBLAS GeMM with Tensors", cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, gpuB_half, CUDA_R_16F, n, gpuA_half, CUDA_R_16F, n, &beta, gpuC, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    cublasDestroy(cublasHandle);

    /* ============================= CUSPARSE ============================== */

    ALLOC_GPU_MEM
    cusparseHandle_t cusparseHandle;
    size_t bufferSize;
    void *gpuBuffer = nullptr;
    cusparseMatDescr_t cusparseMatDescr;
    cusparseSpMatDescr_t matDescrA;
    cusparseDnMatDescr_t matDescrB, matDescrC;
    int64_t rows, cols, ld;
    cudaDataType_t dataType;
    cusparseOrder_t order;

    cusparseCreate(&cusparseHandle);

    cusparseCreateMatDescr(&cusparseMatDescr);
    cusparseSetMatType(cusparseMatDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(cusparseMatDescr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreateCsr(&matDescrA, n, n, csrA->hdr[N],
                      gpuCSRHdr, gpuCSRIdx, gpuCSRData,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
    cusparseCreateDnMat(&matDescrB, n, n, n, gpuB_half,
                        CUDA_R_16F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&matDescrC, n, n, n, gpuC,
                        CUDA_R_32F, CUSPARSE_ORDER_ROW);

    cusparseSpMM_bufferSize(cusparseHandle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_TRANSPOSE,
                            &alpha, matDescrA, matDescrB,
                            &beta, matDescrC, CUDA_R_32F,
                            CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&gpuBuffer, bufferSize);

    RUN_20_AND_REPORT("cuSPARSE CSR", cusparseSpMM(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, &alpha, matDescrA, matDescrB, &beta, matDescrC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, gpuBuffer));
    cusparseDnMatGet(matDescrC, &rows, &cols, &ld, reinterpret_cast<void **>(&gpuC), &dataType, &order);

    /* ============================ HYBRID CSR ============================= */

    float sparsityThresholds[] = {0.0, 0.2, 0.4, 0.6, 0.8};
    for (float threshold : sparsityThresholds) {
        ALLOC_GPU_MEM
        // Copy partial things
        const auto *hcsrA = new HCSRMatrix(*matrixA, threshold);
        hcsrA->bcsr->copyToDevice(&gpuBCSRHdrPart, &gpuBCSRIdxPart, &gpuBCSRDataPart);
        hcsrA->csr->copyToDevice(&gpuCSRHdrPart, &gpuCSRIdxPart, &gpuCSRDataPart);

        string functionName = "Hybrid CSR " + to_string(threshold);
        /* Run the 3-kernel hybrid sequence 20 times and report median */
        RUN_20_AND_REPORT(functionName.c_str(), \
            gridSize = {
                N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0),
                N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0),
                1
            }; \
            blockSize = {N_THREADS, N_THREADS, 1}; \
            sparseMatrixMult1Co<<<gridSize, blockSize>>>(gpuCSRHdrPart, gpuCSRIdxPart, gpuCSRDataPart, gpuB_half, gpuCPart, N); \
            gridSize = {N / 16, N / 16, 1}; blockSize = {32, 1, 1}; \
            sparseMatrixMulTensor1<<<gridSize, blockSize>>>(gpuBCSRHdrPart, gpuBCSRIdxPart, gpuBCSRDataPart, gpuB_half, gpuC, N); \
            gridSize = { CEIL_DIV(N, (N_THREADS * N_THREADS)), CEIL_DIV(N, CEIL_DIV(N_THREADS * N_THREADS, N)), 1 }; \
            blockSize = {N_THREADS * N_THREADS, 1, 1}; \
            addMatrices<<<gridSize, blockSize>>>(gpuC, gpuCPart, N) );
    }

    cusparseDestroySpMat(matDescrA);
    cusparseDestroyDnMat(matDescrB);
    cusparseDestroyDnMat(matDescrC);
    cusparseDestroy(cusparseHandle);
    cudaDeviceReset();

    free(memC);
    free(correctMatrix);
    cudaFree(gpuC);
    cudaFree(gpuA_half);
    cudaFree(gpuB_half);
    cudaFree(gpuCSRData);
    cudaFree(gpuCSRHdr);
    cudaFree(gpuCSRIdx);
    cudaFree(gpuBCSRData);
    cudaFree(gpuBCSRHdr);
    cudaFree(gpuBCSRIdx);

    return 0;
}

// vim: ts=4 sw=4
