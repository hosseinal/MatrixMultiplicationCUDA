//
// Created by huertasg on 7/22/25.
//

#ifndef MISCUTIL_H
#define MISCUTIL_H
#include "Matrix.cuh"

void printMatrix(const float *M, const unsigned int n);

bool checkMatrix(const float *A, const float *B, const unsigned int n);

double rmse(const float *A, const float *B, const unsigned int n);

float maxdiff(const float *A, const float *B, const unsigned int n);

float avgrelerr(const float *A, const float *B, const unsigned int n);

float blockDensity(const Matrix &matrix, int i, int j, int blockSizeRow = 16, int blockSizeCol = 16);

#endif //MISCUTIL_H
