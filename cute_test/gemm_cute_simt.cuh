#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void cute_gemm_simt_invoke(char transA, char transB,
                           int m, int n, int k,
                           float alpha,
                           const float* A, int ldA,
                           const float* B, int ldB,
                           float beta,
                           float* C, int ldC,
                           cudaStream_t stream);

#ifdef __cplusplus
}
#endif
