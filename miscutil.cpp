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
    double sum = 0.0f;

    for (int i = 0; i < n * n; i++) {
        sum += static_cast<double>(A[i] - B[i]) * static_cast<double>(A[i] - B[i]);
    }

    return sqrt(sum / n);
}

float maxdiff(const float *A, const float *B, const unsigned int n) {
    float maxd = 0.0f;

    for (int i = 0; i < n * n; i++) {
        maxd = max(maxd, fabs(A[i] - B[i]));
    }

    return maxd;
}

float avgrelerr(const float *A, const float *B, const unsigned int n) {
    double sum = 0.0f;

    for (int i = 0; i < n * n; i++) {
        sum += fabs(A[i] - B[i]) / B[i];
    }

    return sum / n;
}

float blockDensity(const Matrix &matrix, int i, int j) {
    int nonZeros = 0;
    for (int i1 = i; i1 < i + BLOCK_SIZE; i1++) {
        for (int j1 = j; j1 < j + BLOCK_SIZE; j1++) {
            if (matrix.data[i1 * matrix.cols + j1])
                nonZeros++;
        }
    }

    return static_cast<float>(nonZeros) / static_cast<float>(
               BLOCK_SIZE * BLOCK_SIZE);
}
