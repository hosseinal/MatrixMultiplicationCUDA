//
// Created by huertasg on 8/23/25.
//

#include "HCSRMatrix.h"

#include "miscutil.h"

extern const int BLOCK_SIZE;

HCSRMatrix::HCSRMatrix(const Matrix &matrix, float threshold) {
    const auto *matrix1 = new Matrix(matrix.rows, matrix.cols);
    const auto *matrix2 = new Matrix(matrix.rows, matrix.cols);

    for (int i = 0; i < matrix.rows; i += BLOCK_SIZE) {
        for (int j = 0; j < matrix.cols; j += BLOCK_SIZE) {
            if (blockDensity(matrix, i, j) >= threshold) {
                // Copy this block to the matrix1
                for (int i1 = i; i1 < i + BLOCK_SIZE; i1 ++) {
                    for (int j1 = j; j1 < j + BLOCK_SIZE; j1 ++) {
                        matrix1->data[i1 * matrix1->cols + j1] = matrix.data[i1 * matrix.cols + j1];
                    }
                }
            } else {
                // Copy to matrix2
                for (int i1 = i; i1 < i + BLOCK_SIZE; i1 ++) {
                    for (int j1 = j; j1 < j + BLOCK_SIZE; j1 ++) {
                        matrix2->data[i1 * matrix2->cols + j1] = matrix.data[i1 * matrix.cols + j1];
                    }
                }
            }
        }
    }

    bcsr = new BCSRMatrix(*matrix1);
    csr = new CSRMatrix(*matrix2);
}
