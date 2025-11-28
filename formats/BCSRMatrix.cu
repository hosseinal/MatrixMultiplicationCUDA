//
// Created by huertasg on 7/17/25.
//

#include "BCSRMatrix.cuh"

#include <cassert>
#include <iostream>
#include <stdexcept>

#include "../miscutil.h"

extern const int BLOCK_SIZE;

#define ASSERT_CUDA_SUCCESS error = cudaGetLastError(); \
                            if (error != cudaSuccess) \
                            cout << "BCSRMatrix::copyToDevice CUDA " \
                            "error: " << cudaGetErrorString(error) << "\n"; \
                            assert(error == cudaSuccess);


BCSRMatrix::BCSRMatrix(const Matrix &matrix, int blockSizeRow, int blockSizeCol) 
    : blockSizeRow(blockSizeRow), blockSizeCol(blockSizeCol) {
    // Validate block sizes - WMMA requires at least 16x16 blocks
    if (blockSizeRow < 16 || blockSizeCol < 16) {
        throw std::invalid_argument("Block sizes must be at least 16x16 for WMMA compatibility. "
                                  "Got " + std::to_string(blockSizeRow) + "x" + std::to_string(blockSizeCol));
    }
    
    // Validate block sizes are multiples of 16 - required for WMMA alignment
    if (blockSizeRow % 16 != 0 || blockSizeCol % 16 != 0) {
        throw std::invalid_argument("Block sizes must be multiples of 16 for WMMA alignment. "
                                  "Got " + std::to_string(blockSizeRow) + "x" + std::to_string(blockSizeCol));
    }
    
    // find dense blocks of specified size
    blockRows = matrix.rows / blockSizeRow;
    blockCols = matrix.cols / blockSizeCol;
    hdr = static_cast<int *>(malloc((blockRows + 1) * sizeof(int)));
    hdr[0] = 0;

    for (int i = 0; i < matrix.rows; i += blockSizeRow) {
        hdr[i / blockSizeRow + 1] = hdr[i / blockSizeRow];
        for (int j = 0; j < matrix.cols; j += blockSizeCol) {
            if (blockDensity(matrix, i, j, blockSizeRow, blockSizeCol) > 0.0f) {
                hdr[i / blockSizeRow + 1]++;
            }
        }
    }

    idx = static_cast<int *>(malloc(hdr[blockRows] * sizeof(int)));
    data = static_cast<half *>(malloc(
        hdr[blockRows] * sizeof(half) * blockSizeRow * blockSizeCol));

    int k = 0;
    for (int i = 0; i < matrix.rows; i += blockSizeRow) {
        for (int j = 0; j < matrix.cols; j += blockSizeCol) {
            if (blockDensity(matrix, i, j, blockSizeRow, blockSizeCol) > 0.0f) {
                idx[k] = j / blockSizeCol;
                // obtain fragment
                for (int x = 0; x < blockSizeRow; x++) {
                    for (int y = 0; y < blockSizeCol; y++) {
                        data[k * blockSizeRow * blockSizeCol + x * blockSizeCol + y] =
                                matrix.data[(i + x) * matrix.cols + j + y];
                    }
                }
                nonZeros += blockSizeRow * blockSizeCol;
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
    std::cout << "Block size: " << blockSizeRow << "x" << blockSizeCol << std::endl;
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
        for (int j = 0; j < blockSizeRow; j++) {
            cout << '\t';
            for (int k = 0; k < blockSizeCol; k++) {
                cout << __half2float(
                    data[i * blockSizeRow * blockSizeCol + j * blockSizeCol + k]) <<" ";
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
               hdr[blockRows] * sizeof(half) * blockSizeRow * blockSizeCol);
    ASSERT_CUDA_SUCCESS;

    cudaMemcpy(*gpuHdr, hdr, (blockRows + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    ASSERT_CUDA_SUCCESS;
    cudaMemcpy(*gpuIdx, idx, hdr[blockRows] * sizeof(int),
               cudaMemcpyHostToDevice);
    ASSERT_CUDA_SUCCESS;
    cudaMemcpy(*gpuData, data,
               hdr[blockRows] * sizeof(half) * blockSizeRow * blockSizeCol,
               cudaMemcpyHostToDevice);
    ASSERT_CUDA_SUCCESS;
}
