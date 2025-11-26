//
// Created by huertasg on 8/23/25.
//

#ifndef HCSRMATRIX_H
#define HCSRMATRIX_H
#include "formats/BCSRMatrix.cuh"
#include "formats/CSRMatrix.cuh"

/**
 * Hybrid BCSR Matrix
 *
 * Puts blocks with a density above a threshold in BCSR and otherwise in CSR
 *
 */
class HCSRMatrix {
public:
    BCSRMatrix *bcsr;
    CSRMatrix *csr;
    explicit HCSRMatrix(const Matrix &matrix, float threshold);
};



#endif //HCSRMATRIX_H
