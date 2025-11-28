#include "CSRMatrix.cuh"

#include <cassert>
#include <iostream>

#define CHECK_CUDA_ERRORS(_where) \
    error = cudaGetLastError(); \
    if (error != cudaSuccess) \
        std::cout << _where << " CUDA " \
        "error: " << cudaGetErrorString(error) << "\n"; \
    assert(error == cudaSuccess);

CSRMatrix::CSRMatrix(const Matrix &matrix) {
    N = matrix.rows;
    hdr = static_cast<int *>(malloc((matrix.rows + 1) * sizeof(int)));
    hdr[0] = 0;

    for (int i = 0; i < matrix.rows; i++) {
        hdr[i + 1] = hdr[i];
        for (int j = 0; j < matrix.cols; j++) {
            if (matrix.data[i * matrix.cols + j]) {
                hdr[i + 1]++;
            }
        }
    }

    idx = static_cast<int *>(malloc(hdr[matrix.rows] * sizeof(int)));
    data = static_cast<half *>(malloc(hdr[matrix.rows] * sizeof(half)));

    for (int i = 0, j = 0; i < matrix.rows * matrix.cols; i++) {
        if (matrix.data[i]) {
            idx[j] = i % matrix.rows;
            data[j] = matrix.data[i];
            j++;
        }
    }
}

CSRMatrix::~CSRMatrix() {
    free(hdr);
    free(idx);
    free(data);
}

void CSRMatrix::copyToDevice(int **gpuHdr, int **gpuIdx, half **gpuData) const {
    cudaError error;

    cudaMalloc(reinterpret_cast<void **>(gpuData), hdr[N] * sizeof(half));
    CHECK_CUDA_ERRORS("CSRMatrix::copyToDevice malloc gpuData");
    cudaMalloc(reinterpret_cast<void **>(gpuHdr), (N + 1) * sizeof(int));
    CHECK_CUDA_ERRORS("CSRMatrix::copyToDevice malloc gpuHdr");
    cudaMalloc(reinterpret_cast<void **>(gpuIdx), hdr[N] * sizeof(int));
    CHECK_CUDA_ERRORS("CSRMatrix::copyToDevice malloc gpuIdx");
    cudaMemcpy(*gpuData, data, hdr[N] * sizeof(half),
               cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS("CSRMatrix::copyToDevice gpuData");
    cudaMemcpy(*gpuHdr, hdr, (N + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS("CSRMatrix::copyToDevice gpuHdr");
    cudaMemcpy(*gpuIdx, idx, hdr[N] * sizeof(int),
               cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS("CSRMatrix::copyToDevice gpuIdx");
}
