//
// Created by huertasg on 7/17/25.
//

#include "BCSRMatrix.cuh"

#include <cassert>
#include <iostream>

#include "miscutil.h"

extern const int BLOCK_SIZE;

#define ASSERT_CUDA_SUCCESS error = cudaGetLastError(); \
                            if (error != cudaSuccess) \
                            cout << "BCSRMatrix::copyToDevice CUDA " \
                            "error: " << cudaGetErrorString(error) << '\n'; \
                            assert(error == cudaSuccess);


BCSRMatrix::BCSRMatrix(const Matrix &matrix) {
    // find dense 16x16 blocks
    blockRows = matrix.rows / BLOCK_SIZE;
    hdr = static_cast<int *>(malloc((blockRows + 1) * sizeof(int)));
    hdr[0] = 0;

    for (int i = 0; i < matrix.rows; i += BLOCK_SIZE) {
        hdr[i / BLOCK_SIZE + 1] = hdr[i / BLOCK_SIZE];
        for (int j = 0; j < matrix.cols; j += BLOCK_SIZE) {
            if (blockDensity(matrix, i, j) > 0.0f) {
                hdr[i / BLOCK_SIZE + 1]++;
            }
        }
    }

    idx = static_cast<int *>(malloc(hdr[blockRows] * sizeof(int)));
    data = static_cast<half *>(malloc(
        hdr[blockRows] * sizeof(half) * BLOCK_SIZE * BLOCK_SIZE));

    int k = 0;
    for (int i = 0; i < matrix.rows; i += BLOCK_SIZE) {
        for (int j = 0; j < matrix.cols; j += BLOCK_SIZE) {
            if (blockDensity(matrix, i, j) > 0.0f) {
                idx[k] = j / BLOCK_SIZE;
                // obtain fragment
                for (int x = 0; x < BLOCK_SIZE; x++) {
                    for (int y = 0; y < BLOCK_SIZE; y++) {
                        data[k * BLOCK_SIZE * BLOCK_SIZE + x * BLOCK_SIZE + y] =
                                matrix.data[(i + x) * matrix.cols + j + y];
                    }
                }
                nonZeros += BLOCK_SIZE * BLOCK_SIZE;
                k++;
            }
        }
    }

    //assert(nonZeros == matrix.nonZeros);
}

BCSRMatrix::~BCSRMatrix() {
    free(hdr);
    free(idx);
    free(data);
}

void BCSRMatrix::print() const {
    std::cout << "hdr:\n\t";
    for (int i = 0; i < blockRows; i++) {
        std::cout << hdr[i] << " ";
    }
    std::cout << "\nidx:\n\t";
    for (int i = 0; i < hdr[blockRows]; i++) {
        std::cout << idx[i] << " ";
    }
    std::cout << "\ndata:\n\t";
    for (int i = 0; i < hdr[blockRows]; i++) {
        std::cout << "=== Block " << i << " ===\n";
        for (int j = 0; j < BLOCK_SIZE; j++) {
            cout << '\t';
            for (int k = 0; k < BLOCK_SIZE; k++) {
                cout << __half2float(
                    data[i * BLOCK_SIZE * BLOCK_SIZE + j * BLOCK_SIZE + k]) <<" ";
            }
            cout << '\n';
        }
    }
}

void BCSRMatrix::copyToDevice(int **gpuHdr, int **gpuIdx, half **gpuData)
const {
    cudaError error;

    cudaMalloc(reinterpret_cast<void **>(gpuHdr),
               (blockRows + 1) * sizeof(int));
    ASSERT_CUDA_SUCCESS;

    cudaMalloc(reinterpret_cast<void **>(gpuIdx),
               hdr[blockRows] * sizeof(int));
    ASSERT_CUDA_SUCCESS;

    cudaMalloc(reinterpret_cast<void **>(gpuData),
               hdr[blockRows] * sizeof(half) * BLOCK_SIZE * BLOCK_SIZE);
    ASSERT_CUDA_SUCCESS;

    cudaMemcpy(*gpuHdr, hdr, (blockRows + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    ASSERT_CUDA_SUCCESS;
    cudaMemcpy(*gpuIdx, idx, hdr[blockRows] * sizeof(int),
               cudaMemcpyHostToDevice);
    ASSERT_CUDA_SUCCESS;
    cudaMemcpy(*gpuData, data,
               hdr[blockRows] * sizeof(half) * BLOCK_SIZE * BLOCK_SIZE,
               cudaMemcpyHostToDevice);
    ASSERT_CUDA_SUCCESS;
}
