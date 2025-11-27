//
// Created by huertasg on 7/17/25.
//

#ifndef CSRMATRIX_CUH
#define CSRMATRIX_CUH

#include <cuda_fp16.h>
#include <string>

#include "../Matrix.cuh"

class CSRMatrix {
public:
    int N;
    int *hdr = nullptr;
    int *idx = nullptr;
    half *data = nullptr;
    explicit CSRMatrix(const Matrix &matrix);
    ~CSRMatrix();
    void copyToDevice(int **gpuHdr, int **gpuIdx, half **gpuData) const;
};


#endif //CSRMATRIX_CUH
