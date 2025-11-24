#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// Macro for ceiling division
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

// Forward declarations of all CUDA kernels

// Dense matrix multiplication kernels
__global__ void denseMatrixMul(const half *d_A, const half *d_B, float *d_C,
                               const unsigned int M, const unsigned int N, const unsigned int Z);

__global__ void denseMatrixMul(const half *d_A, const half *d_B, float *d_C,
                               const unsigned int n);

__global__ void denseMatrixMulCo(const half *d_A, const half *d_B, float *d_C,
                                 const unsigned int n);

__global__ void denseMatrixMulCo(const half *d_A, const half *d_B, float *d_C,
                                 const unsigned int M, const unsigned int N, const unsigned int Z);

__global__ void denseMatrixMulTensor(const half *d_A, const half *d_B,
                                     float *d_C, const unsigned int M, const unsigned int N, const unsigned int Z);

// Sparse matrix multiplication kernels
__global__ void sparseMatrixMult1Co(const int *hdr, const int *idx,
                                    const half *data, const half *B, float *C,
                                    const unsigned int n);

__global__ void sparseMatrixMult1(const int *hdr, const int *idx,
                                  const half *data, const half *B, float *C,
                                  const unsigned int n);

__global__ void sparseMatrixMulTensor(const int *hdr, const int *idx,
                                      const half *data, const half *B,
                                      float *C, const unsigned int M, const unsigned int N);

__global__ void sparseMatrixMulTensor1(const int *hdr, const int *idx,
                                       const half *data, const half *B,
                                       float *C, const unsigned int n);

__global__ void sparseMatrixMulTensor_v2(const int *hdr, const int *idx,
                                         const half *data, const half *B,
                                         float *C, const unsigned int M, const unsigned int N);

__global__ void sparseMatrixMulTensor_v3(const int *hdr, const int *idx,
                                         const half *data, const half *B,
                                         float *C, const unsigned int M, const unsigned int N);

__global__ void sparseMatrixMulTensor_v1_improved(const int * __restrict__ hdr, const int * __restrict__ idx,
                                                  const half * __restrict__ data, const half * __restrict__ B,
                                                  float * __restrict__ C, const unsigned int M, const unsigned int N);

__global__ void sparseMatrixMulTensor_option2_ldmatrix_sm80(const int *hdr, const int *idx,
                                                            const half *data, const half *B,
                                                            float *C, const unsigned int M, const unsigned int N);

// Large random pattern specific kernel (64x16 blocks)
__global__ void sparseMatrixMulTensor64x16(const int *hdr, const int *idx,
                                                 const half *data, const half *B,
                                                 float *C, const unsigned int M, const unsigned int N);

// 64x16 block size specific kernel v2 (similar to v2 but for 64x16 blocks, uses 64 threads)
__global__ void sparseMatrixMulTensor64x16_v2(const int *hdr, const int *idx,
                                              const half *data, const half *B,
                                              float *C, const unsigned int M, const unsigned int N);

// 32x16 block size specific kernel
__global__ void sparseMatrixMulTensor32x16(const int *hdr, const int *idx,
                                          const half *data, const half *B,
                                          float *C, const unsigned int M, const unsigned int N);

// Variant: same as sparseMatrixMulTensor32x16 but stages A blocks into shared
// memory first so each warp loads A from fast on-chip memory via WMMA API.
// Launch with shared memory size at least 16*16*sizeof(half).
__global__ void sparseMatrixMulTensor32x16_shared(const int *hdr, const int *idx,
                                                  const half *data, const half *B,
                                                  float *C, const unsigned int M, const unsigned int N);

// Variant: stages B into shared memory using vectorized loads and stages A from BCSR into shared memory
__global__ void sparseMatrixMulTensor32x16_shared_vectorized(const int *hdr, const int *idx,
                                                              const half *data, const half *B,
                                                              float *C, const unsigned int M, const unsigned int N);

// 32x16 block size specific kernel v2 (v2-style for 32x16 blocks, uses 64 threads)
__global__ void sparseMatrixMulTensor32x16_v2(const int *hdr, const int *idx,
                                              const half *data, const half *B,
                                              float *C, const unsigned int M, const unsigned int N);

// Adaptive BCSR kernels that balance load based on matrix aspect ratio
__global__ void sparseMatrixMulTensorAdaptive(const int *hdr, const int *idx,
                                              const half *data, const half *B,
                                              float *C, const unsigned int M, const unsigned int N,
                                              const unsigned int colBlocksPerRow);

__global__ void sparseMatrixMulTensorAdaptive32x16(const int *hdr, const int *idx,
                                                   const half *data, const half *B,
                                                   float *C, const unsigned int M, const unsigned int N,
                                                   const unsigned int colBlocksPerRow);

__global__ void sparseMatrixMulTensorAdaptive64x16(const int *hdr, const int *idx,
                                                   const half *data, const half *B,
                                                   float *C, const unsigned int M, const unsigned int N,
                                                   const unsigned int colBlocksPerRow);

// Utility kernels
__global__ void addMatrices(float *C, const float *CPart, const unsigned int n);

#endif // CUDA_KERNELS_CUH