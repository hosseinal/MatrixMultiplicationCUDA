//
// Created by huertasg on 7/17/25.
//

#ifndef BCSRMATRIX_CUH
#define BCSRMATRIX_CUH
#include "Matrix.cuh"

class BCSRMatrix {
public:
    int *hdr = nullptr;
    int *idx = nullptr;
    half *data = nullptr;
    int blockRows = 0;
    int nonZeros = 0;
    explicit BCSRMatrix(const Matrix &matrix);
    ~BCSRMatrix();

    void print() const;
    void copyToDevice(int **gpuHdr, int **gpuIdx, half **gpuData) const;
};


#endif //BCSRMATRIX_CUH