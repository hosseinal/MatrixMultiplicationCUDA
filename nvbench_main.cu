// Minimal nvbench harness that reuses existing kernels in main.cu
#include <cassert>
#include <iostream>
#include <nvbench/nvbench.cuh>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include "BCSRMatrix.cuh"
#include "CSRMatrix.cuh"
#include "HCSRMatrix.h"
#include "Matrix.cuh"
#include "miscutil.h"

using namespace std;

// Local copy of globals from main.cu
extern unsigned int N;
extern string MATRIX_A_PATH;
extern string MATRIX_B_PATH;
extern string MATRIX_PATTERN;
extern string MATRIX_SPARSITY;

// forward declare kernels from main.cu
__global__ void denseMatrixMulCo(const half *d_A, const half *d_B, float *d_C, const unsigned int n);
__global__ void denseMatrixMulTensor(const half *d_A, const half *d_B, float *d_C, const unsigned int n);
__global__ void sparseMatrixMult1Co(const int *hdr, const int *idx, const half *data, const half *B, float *C, const unsigned int n);
__global__ void sparseMatrixMulTensor(const int *hdr, const int *idx, const half *data, const half *B, float *C, const unsigned int n);
__global__ void sparseMatrixMulTensor1(const int *hdr, const int *idx, const half *data, const half *B, float *C, const unsigned int n);

// Device buffer holder
struct DevBuffers {
    half *gpuA_half = nullptr;
    half *gpuB_half = nullptr;
    float *gpuC = nullptr;
    float *gpuCPart = nullptr;
    int *gpuCSRHdr = nullptr;
    int *gpuCSRIdx = nullptr;
    half *gpuCSRData = nullptr;
    int *gpuBCSRHdr = nullptr;
    int *gpuBCSRIdx = nullptr;
    half *gpuBCSRData = nullptr;
};

static DevBuffers setup_device(const Matrix *matrixA, const Matrix *matrixB, CSRMatrix *&csrA, BCSRMatrix *&bcsrA) {
    DevBuffers dev;
    N = matrixA->cols;
    csrA = new CSRMatrix(*matrixA);
    bcsrA = new BCSRMatrix(*matrixA);

    // copy CSR/BCSR to device
    bcsrA->copyToDevice(&dev.gpuBCSRHdr, &dev.gpuBCSRIdx, &dev.gpuBCSRData);
    csrA->copyToDevice(&dev.gpuCSRHdr, &dev.gpuCSRIdx, &dev.gpuCSRData);

    // allocate dense buffers
    cudaMalloc(reinterpret_cast<void **>(&dev.gpuA_half), N * N * sizeof(half));
    cudaMalloc(reinterpret_cast<void **>(&dev.gpuB_half), N * N * sizeof(half));
    cudaMalloc(reinterpret_cast<void **>(&dev.gpuC), N * N * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&dev.gpuCPart), N * N * sizeof(float));

    cudaMemcpy(dev.gpuA_half, matrixA->data, N * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dev.gpuB_half, matrixB->data, N * N * sizeof(half), cudaMemcpyHostToDevice);

    cudaMemset(dev.gpuC, 0, N * N * sizeof(float));
    cudaMemset(dev.gpuCPart, 0, N * N * sizeof(float));

    return dev;
}

static void teardown_device(DevBuffers &dev, CSRMatrix *csrA, BCSRMatrix *bcsrA) {
    if (dev.gpuA_half) cudaFree(dev.gpuA_half);
    if (dev.gpuB_half) cudaFree(dev.gpuB_half);
    if (dev.gpuC) cudaFree(dev.gpuC);
    if (dev.gpuCPart) cudaFree(dev.gpuCPart);
    if (dev.gpuCSRData) cudaFree(dev.gpuCSRData);
    if (dev.gpuCSRHdr) cudaFree(dev.gpuCSRHdr);
    if (dev.gpuCSRIdx) cudaFree(dev.gpuCSRIdx);
    if (dev.gpuBCSRData) cudaFree(dev.gpuBCSRData);
    if (dev.gpuBCSRHdr) cudaFree(dev.gpuBCSRHdr);
    if (dev.gpuBCSRIdx) cudaFree(dev.gpuBCSRIdx);
    delete csrA;
    delete bcsrA;
}

// Bench: Dense WMMA (uses denseMatrixMulTensor kernel)
static void bench_dense_wmma(nvbench::state &state) {
    const char *a_path = getenv("MATRIX_A_PATH");
    const char *b_path = getenv("MATRIX_B_PATH");
    const Matrix *matrixA = new Matrix(a_path ? a_path : "../medA.mat");
    const Matrix *matrixB = new Matrix(b_path ? b_path : "../medB.mat");
    CSRMatrix *csrA = nullptr; BCSRMatrix *bcsrA = nullptr;
    DevBuffers dev = setup_device(matrixA, matrixB, csrA, bcsrA);

    state.exec([&](nvbench::launch &launch) {
        const int n = static_cast<int>(N);
        dim3 grid(n / 16, n / 16, 1);
        dim3 block(32, 1, 1);
        denseMatrixMulTensor<<<grid, block, 0, launch.get_stream()>>>(dev.gpuA_half, dev.gpuB_half, dev.gpuC, N);
        cudaStreamSynchronize(launch.get_stream());
    });

    teardown_device(dev, csrA, bcsrA);
    delete matrixA; delete matrixB;
}

NVBENCH_BENCH(bench_dense_wmma).set_name("Dense WMMA");

// Bench: SpMM with Tensors
static void bench_spmm_tensors(nvbench::state &state) {
    const char *a_path = getenv("MATRIX_A_PATH");
    const char *b_path = getenv("MATRIX_B_PATH");
    const Matrix *matrixA = new Matrix(a_path ? a_path : "../medA.mat");
    const Matrix *matrixB = new Matrix(b_path ? b_path : "../medB.mat");
    CSRMatrix *csrA = nullptr; BCSRMatrix *bcsrA = nullptr;
    DevBuffers dev = setup_device(matrixA, matrixB, csrA, bcsrA);

    state.exec([&](nvbench::launch &launch) {
        const int n = static_cast<int>(N);
        dim3 grid(n / 16, n / 16, 1);
        dim3 block(32, 1, 1);
        sparseMatrixMulTensor<<<grid, block, 0, launch.get_stream()>>>(dev.gpuBCSRHdr, dev.gpuBCSRIdx, dev.gpuBCSRData, dev.gpuB_half, dev.gpuC, N);
        cudaStreamSynchronize(launch.get_stream());
    });

    teardown_device(dev, csrA, bcsrA);
    delete matrixA; delete matrixB;
}

    // Bench: Dense GPU (coalesced) -> denseMatrixMulCo
    static void bench_dense_coalesced(nvbench::state &state) {
        const Matrix *matrixA = new Matrix(MATRIX_A_PATH);
        const Matrix *matrixB = new Matrix(MATRIX_B_PATH);
        CSRMatrix *csrA = nullptr; BCSRMatrix *bcsrA = nullptr;
        DevBuffers dev = setup_device(matrixA, matrixB, csrA, bcsrA);

        state.exec([&](nvbench::launch &launch) {
            const int n = static_cast<int>(N);
            dim3 grid(CEIL_DIV(n, (N_THREADS * N_THREADS)), CEIL_DIV(n, CEIL_DIV(N_THREADS * N_THREADS, n)), 1);
            dim3 block(N_THREADS * N_THREADS, 1, 1);
            denseMatrixMulCo<<<grid, block, 0, launch.get_stream()>>>(dev.gpuA_half, dev.gpuB_half, dev.gpuC, N);
            cudaStreamSynchronize(launch.get_stream());
        });

        teardown_device(dev, csrA, bcsrA);
        delete matrixA; delete matrixB;
    }

    NVBENCH_BENCH(bench_dense_coalesced).set_name("Dense on GPU Coalescence");

    // Bench: SpMM 1 Co (CSR * dense) -> sparseMatrixMult1Co
    static void bench_spmm1_co(nvbench::state &state) {
        const Matrix *matrixA = new Matrix(MATRIX_A_PATH);
        const Matrix *matrixB = new Matrix(MATRIX_B_PATH);
        CSRMatrix *csrA = nullptr; BCSRMatrix *bcsrA = nullptr;
        DevBuffers dev = setup_device(matrixA, matrixB, csrA, bcsrA);

        state.exec([&](nvbench::launch &launch) {
            dim3 grid( CEIL_DIV(N, N_THREADS), CEIL_DIV(N, N_THREADS), 1 );
            dim3 block(N_THREADS, N_THREADS, 1);
            sparseMatrixMult1Co<<<grid, block, 0, launch.get_stream()>>>(dev.gpuCSRHdr, dev.gpuCSRIdx, dev.gpuCSRData, dev.gpuB_half, dev.gpuC, N);
            cudaStreamSynchronize(launch.get_stream());
        });

        teardown_device(dev, csrA, bcsrA);
        delete matrixA; delete matrixB;
    }

    NVBENCH_BENCH(bench_spmm1_co).set_name("SpMM 1 Co");

NVBENCH_BENCH(bench_spmm_tensors).set_name("SpMM with Tensors");

// Bench: SpMM with Tensors Opt
static void bench_spmm_tensors_op(nvbench::state &state) {
    const char *a_path = getenv("MATRIX_A_PATH");
    const char *b_path = getenv("MATRIX_B_PATH");
    const Matrix *matrixA = new Matrix(a_path ? a_path : "../medA.mat");
    const Matrix *matrixB = new Matrix(b_path ? b_path : "../medB.mat");
    CSRMatrix *csrA = nullptr; BCSRMatrix *bcsrA = nullptr;
    DevBuffers dev = setup_device(matrixA, matrixB, csrA, bcsrA);

    state.exec([&](nvbench::launch &launch) {
        const int n = static_cast<int>(N);
        dim3 grid(n / 16, n / 16, 1);
        dim3 block(32, 1, 1);
        sparseMatrixMulTensor1<<<grid, block, 0, launch.get_stream()>>>(dev.gpuBCSRHdr, dev.gpuBCSRIdx, dev.gpuBCSRData, dev.gpuB_half, dev.gpuC, N);
        cudaStreamSynchronize(launch.get_stream());
    });

    teardown_device(dev, csrA, bcsrA);
    delete matrixA; delete matrixB;
}

NVBENCH_BENCH(bench_spmm_tensors_op).set_name("SpMM with Tensors Op");

// Bench: cuBLAS GeMM (with tensor ops)
static void bench_cublas_gemm_tensor(nvbench::state &state) {
    const char *a_path = getenv("MATRIX_A_PATH");
    const char *b_path = getenv("MATRIX_B_PATH");
    const Matrix *matrixA = new Matrix(a_path ? a_path : "../medA.mat");
    const Matrix *matrixB = new Matrix(b_path ? b_path : "../medB.mat");
    CSRMatrix *csrA = nullptr; BCSRMatrix *bcsrA = nullptr;
    DevBuffers dev = setup_device(matrixA, matrixB, csrA, bcsrA);

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);

    state.exec([&](nvbench::launch &launch) {
        const int n = static_cast<int>(N);
        cublasSetStream(cublasHandle, launch.get_stream());
        const float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dev.gpuB_half, CUDA_R_16F, n, dev.gpuA_half, CUDA_R_16F, n, &beta, dev.gpuC, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    });

    cublasDestroy(cublasHandle);
    teardown_device(dev, csrA, bcsrA);
    delete matrixA; delete matrixB;
}

NVBENCH_BENCH(bench_cublas_gemm_tensor).set_name("cuBLAS GeMM with Tensors");

// Bench: cuSPARSE SpMM
static void bench_cusparse_spmm(nvbench::state &state) {
    const char *a_path = getenv("MATRIX_A_PATH");
    const char *b_path = getenv("MATRIX_B_PATH");
    const Matrix *matrixA = new Matrix(a_path ? a_path : "../medA.mat");
    const Matrix *matrixB = new Matrix(b_path ? b_path : "../medB.mat");
    CSRMatrix *csrA = nullptr; BCSRMatrix *bcsrA = nullptr;
    DevBuffers dev = setup_device(matrixA, matrixB, csrA, bcsrA);

    // Prepare cuSPARSE descriptors and workspace once, outside the timed region
    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);

    const int n = static_cast<int>(N);
    cusparseMatDescr_t cusparseMatDescr;
    cusparseCreateMatDescr(&cusparseMatDescr);
    cusparseSetMatType(cusparseMatDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(cusparseMatDescr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseSpMatDescr_t matDescrA;
    cusparseDnMatDescr_t matDescrB, matDescrC;
    cusparseCreateCsr(&matDescrA, n, n, csrA->hdr[N], dev.gpuCSRHdr, dev.gpuCSRIdx, dev.gpuCSRData, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
    cusparseCreateDnMat(&matDescrB, n, n, n, dev.gpuB_half, CUDA_R_16F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&matDescrC, n, n, n, dev.gpuC, CUDA_R_32F, CUSPARSE_ORDER_ROW);

    size_t bufferSize = 0;
    cusparseSpMM_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, (const float *)&(const float){1.0f}, matDescrA, matDescrB, (const float *)&(const float){0.0f}, matDescrC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    void *gpuBuffer = nullptr;
    if (bufferSize > 0) cudaMalloc(&gpuBuffer, bufferSize);

    // Timed region: only invoke cusparseSpMM on the nvbench stream
    state.exec([&](nvbench::launch &launch) {
        cusparseSetStream(cusparseHandle, launch.get_stream());
        cusparseSpMM(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, (const float *)&(const float){1.0f}, matDescrA, matDescrB, (const float *)&(const float){0.0f}, matDescrC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, gpuBuffer);
        cudaStreamSynchronize(launch.get_stream());
    });

    if (gpuBuffer) cudaFree(gpuBuffer);
    cusparseDestroySpMat(matDescrA);
    cusparseDestroyDnMat(matDescrB);
    cusparseDestroyDnMat(matDescrC);
    cusparseDestroy(cusparseHandle);
    teardown_device(dev, csrA, bcsrA);
    delete matrixA; delete matrixB;
}

NVBENCH_BENCH(bench_cusparse_spmm).set_name("cuSPARSE CSR");

NVBENCH_MAIN();

