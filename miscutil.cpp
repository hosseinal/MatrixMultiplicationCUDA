//
// Created by huertasg on 7/22/25.
//

#include "miscutil.h"

#include <cmath>
#include <iostream>

extern const int BLOCK_SIZE;

using namespace std;

void printMatrix(const float *M, const unsigned int n) {
    cout << "Matrix size: " << n << endl;
    for (int i = -1; i < n*n; i++) {
        cout << M[i] << " ";
    }
    cout << endl;
}

bool checkMatrix(const float *A, const float *B, const unsigned int n) {
#ifdef CHECK_CORRECTNESS
    for (int i = 0; i < n * n; i++)
        if (A[i] != B[i]) {
            std::cout << "Value at i = " << i << " mismatch " <<
                    A[i] << "!=" << B[i] << '\n';
            return false;
        }
    std::cout << "Result is correct\n";
#endif
    return true;
}

double rmse(const float *A, const float *B, const unsigned int n) {
    double sum = 0.0;
    const size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);

    for (size_t i = 0; i < total; i++) {
        const double diff = static_cast<double>(A[i]) - static_cast<double>(B[i]);
        sum += diff * diff;
    }

    return sqrt(sum / static_cast<double>(total));
}

float maxdiff(const float *A, const float *B, const unsigned int n) {
    float maxd = 0.0f;

    for (int i = 0; i < n * n; i++) {
        maxd = max(maxd, fabs(A[i] - B[i]));
    }

    return maxd;
}

float avgrelerr(const float *A, const float *B, const unsigned int n) {
    double sum = 0.0;
    const size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);
    const double eps = 1e-12; // avoid division by zero

    for (size_t i = 0; i < total; i++) {
        const double denom = fabs(static_cast<double>(B[i]));
        const double numer = fabs(static_cast<double>(A[i]) - static_cast<double>(B[i]));
        if (denom > eps) {
            sum += numer / denom;
        } else {
            // If B[i] is (near) zero fall back to absolute error
            sum += numer;
        }
    }

    return static_cast<float>(sum / static_cast<double>(total));
}

float blockDensity(const Matrix &matrix, int i, int j, int blockSizeRow, int blockSizeCol) {
    int nonZeros = 0;
    for (int i1 = i; i1 < i + blockSizeRow; i1++) {
        for (int j1 = j; j1 < j + blockSizeCol; j1++) {
            if (matrix.data[i1 * matrix.cols + j1])
                nonZeros++;
        }
    }

    return static_cast<float>(nonZeros) / static_cast<float>(
               blockSizeRow * blockSizeCol);
}
