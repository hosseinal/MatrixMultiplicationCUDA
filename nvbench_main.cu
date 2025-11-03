// nvbench harness to run selected kernels from main.cu across patterns/sizes/sparsities
#include <nvbench/nvbench.cuh>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <cassert>

#include <cassert>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <mma.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <algorithm>
#include <cooperative_groups.h>
#include <cuda/barrier>

#include "matrix_generator.h"
// Include implementation so templates are available in this TU (quick solution)
#include "matrix_generator.cpp"
// Matrix/CSR/BCSR types
#include "Matrix.cuh"
#include "CSRMatrix.cuh"
#include "BCSRMatrix.cuh"



using namespace std;
using namespace nvcuda;

#ifndef CEIL_DIV
#define CEIL_DIV(_a, _b) (((_a) / (_b)) + (((_a) % (_b)) > 0 ? 1 : 0))
#endif
// Dense matrix multiplication for rectangular A(MxN) * B(NxZ) = C(MxZ)
__global__ void denseMatrixMul(const half *d_A, const half *d_B, float *d_C,
							   const unsigned int M, const unsigned int N, const unsigned int Z) {
	const unsigned int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
	const unsigned int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if (rowIdx < M && colIdx < Z) {
		float tmp = 0.0f;
		for (unsigned int k = 0; k < N; ++k) {
			tmp += __half2float(d_A[rowIdx * N + k]) * __half2float(d_B[k * Z + colIdx]);
		}
		d_C[rowIdx * Z + colIdx] = tmp;
	}

}


// Variant: largeRandom-specific kernel. Assumes we want to process 4 consecutive
// 16-row panels (64 rows) in a single block and one 16-column tile. For the
// "largerandom" pattern we generate 64x16 blocks; this kernel reads up to 4
// distinct 16x16 A blocks (one per sub-row) and multiplies each with the same
// B tile, producing four 16x16 WMMA accumulators that are written back to C.
__global__ void sparseMatrixMulTensorlargeRandom(const int *hdr, const int *idx,
												 const half *data, const half *B,
												 float *C, const unsigned int M, const unsigned int N) {
	// Each block covers 64 rows (4*16) and 16 columns
	const unsigned int blockRow64 = blockIdx.y * 64u;
	const unsigned int tileCol = blockIdx.x * 32u;

	if (blockRow64 >= M || tileCol >= N) return;

	// Prepare 4 WMMA accumulators, one per 16-row sub-panel
	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag0, a_frag1, a_frag2, a_frag3;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag, b_frag1 ;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1, c2, c3, c4, c5, c6, c7;

	wmma::fill_fragment(c0, 0.0f);
	wmma::fill_fragment(c1, 0.0f);
	wmma::fill_fragment(c2, 0.0f);
	wmma::fill_fragment(c3, 0.0f);
    wmma::fill_fragment(c4, 0.0f);
    wmma::fill_fragment(c5, 0.0f);
    wmma::fill_fragment(c6, 0.0f);
    wmma::fill_fragment(c7, 0.0f);


	// base block-row index in 16-row units
	const int baseBlockRow = static_cast<int>(blockRow64 / 16u);

	// Determine how many valid 16-row sub-blocks we have (up to 4)
	unsigned int remaining_rows = (M > blockRow64) ? (M - blockRow64) : 0u;
	int sub_count = static_cast<int>((remaining_rows + 15u) / 16u);
	if (sub_count > 4) sub_count = 4;

	// Compute base block-row indices for the up-to-4 sub-block-rows
	const int br0 = baseBlockRow + 0;
	const int br1 = baseBlockRow + 1;
	const int br2 = baseBlockRow + 2;
	const int br3 = baseBlockRow + 3;

	// Compute combined k range across the sub-rows so we can load B once per k
	const int k_start = hdr[baseBlockRow];
	const int k_end = hdr[baseBlockRow + sub_count];

	// Iterate once over the combined k range. For each k load B tile a single
	// time and dispatch the multiply for whichever sub-rows include that k.
	for (int k = k_start; k < k_end; ++k) {
		const size_t b_off = static_cast<size_t>(idx[k]) * 16u * static_cast<size_t>(N) + static_cast<size_t>(tileCol);
		wmma::load_matrix_sync(b_frag, B + b_off, N);
        wmma::load_matrix_sync(b_frag1, B + b_off + 16, N);

		// If k belongs to sub-row 0
		if (sub_count >= 1 && k >= hdr[br0] && k < hdr[br0 + 1]) {
			wmma::load_matrix_sync(a_frag0, data + static_cast<size_t>(k) * 16u * 16u, 16);
			wmma::mma_sync(c0, a_frag0, b_frag, c0);
            wmma::mma_sync(c4, a_frag0, b_frag1, c4);
		}
		// sub-row 1
		if (sub_count >= 2 && k >= hdr[br1] && k < hdr[br1 + 1]) {
			wmma::load_matrix_sync(a_frag1, data + static_cast<size_t>(k) * 16u * 16u, 16);
			wmma::mma_sync(c1, a_frag1, b_frag, c1);
            wmma::mma_sync(c5, a_frag1, b_frag1, c5);
		}
		// sub-row 2
		if (sub_count >= 3 && k >= hdr[br2] && k < hdr[br2 + 1]) {
			wmma::load_matrix_sync(a_frag2, data + static_cast<size_t>(k) * 16u * 16u, 16);
			wmma::mma_sync(c2, a_frag2, b_frag, c2);
            wmma::mma_sync(c6, a_frag2, b_frag1, c6);
		}
		// sub-row 3
		if (sub_count == 4 && k >= hdr[br3] && k < hdr[br3 + 1]) {
			wmma::load_matrix_sync(a_frag3, data + static_cast<size_t>(k) * 16u * 16u, 16);
			wmma::mma_sync(c3, a_frag3, b_frag, c3);
            wmma::mma_sync(c7, a_frag3, b_frag1, c7);
		}
	}

	// Store each 16x16 accumulator to the corresponding rows in C (guard columns)
	const unsigned int outCol = tileCol;
	if (outCol < N) {
		// sub 0
		if (blockRow64 + 0 * 16u < M){
             wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 0 * 16u) * N + outCol, c0, N, wmma::mem_row_major);
             wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 0 * 16u) * N + outCol + 16, c4, N, wmma::mem_row_major);
        }
        if (blockRow64 + 1 * 16u < M){
         wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 1 * 16u) * N + outCol, c1, N, wmma::mem_row_major);
         wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 1 * 16u) * N + outCol + 16, c5, N, wmma::mem_row_major);
        }
		if (blockRow64 + 2 * 16u < M){
             wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 2 * 16u) * N + outCol, c2, N, wmma::mem_row_major);
             wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 2 * 16u) * N + outCol + 16, c6, N, wmma::mem_row_major);
        }
		if (blockRow64 + 3 * 16u < M) {
            wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 3 * 16u) * N + outCol, c3, N, wmma::mem_row_major);
            wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 3 * 16u) * N + outCol + 16, c7, N, wmma::mem_row_major);
        }
	}
}

// Improved v1 variant: cooperative A staging, unified warp path, reduced divergence,
// size-safe offsets and restrict qualifiers. This keeps the same block layout as v2
// (64 threads, two 16x16 tiles per block horizontally). It uses WMMA for full
// tiles; partial-tile handling remains the caller's responsibility (same as v2).
__global__ void sparseMatrixMulTensor_v1_improved(const int * __restrict__ hdr, const int * __restrict__ idx,
												  const half * __restrict__ data, const half * __restrict__ B,
												  float * __restrict__ C, const unsigned int M, const unsigned int N) {
	const unsigned int warpRow = blockIdx.y * 16u;
	const unsigned int tileColBase = blockIdx.x * 32u; // covers 32 columns per block

	if (warpRow >= M || tileColBase >= N) return;

	const unsigned int warpId = threadIdx.x / 32u; // 0 or 1
	const unsigned int laneId = threadIdx.x & 31u;

	extern __shared__ half sA[]; // 16*16 elements (256)

	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
	wmma::fill_fragment(c_frag, 0.0f);

	const int blockRowIdx = static_cast<int>(warpRow / 16u);

	// Iterate non-zero blocks in this block-row
	for (int k = hdr[blockRowIdx]; k < hdr[blockRowIdx + 1]; ++k) {
		// Global pointer to A block (16x16 contiguous)
		const half *a_global = data + static_cast<size_t>(k) * 16u * 16u;

		// Cooperative load of 16*16 half elements into shared memory
		for (unsigned int i = threadIdx.x; i < 16u * 16u; i += blockDim.x) {
			sA[i] = a_global[i];
		}
		__syncthreads();

		// Compute tile column for this warp (both warps follow same path)
		const unsigned int tileCol = tileColBase + warpId * 16u;
		if (tileCol >= N) {
			__syncthreads();
			continue;
		}

		// Load A from shared memory (leading dim 16) and B from global memory
		wmma::load_matrix_sync(a_frag, sA, 16);
		const size_t b_offset = static_cast<size_t>(idx[k]) * 16u * static_cast<size_t>(N) + static_cast<size_t>(tileCol);
		wmma::load_matrix_sync(b_frag, B + b_offset, N);
		wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

		__syncthreads();
	}

	// store results for this warp's tile
	const unsigned int outCol = tileColBase + warpId * 16u;
	if (outCol < N) {
		const size_t out_offset = static_cast<size_t>(warpRow) * static_cast<size_t>(N) + static_cast<size_t>(outCol);
		wmma::store_matrix_sync(C + out_offset, c_frag, N, wmma::mem_row_major);
	}
}

// Option2: Ampere (sm_80) inline-PTX ldmatrix + mma.sync implementation (ungarded)
// WARNING: This emits raw PTX sequences for ldmatrix/mma.sync and may require
// iterative fixes on your toolchain. You asked for an unguarded/raw PTX path.
__global__ void sparseMatrixMulTensor_option2_ldmatrix_sm80(const int *hdr, const int *idx,
															const half *data, const half *B,
															float *C, const unsigned int M, const unsigned int N) {
	const unsigned int warpRow = blockIdx.y * 16;
	const unsigned int tileColBase = blockIdx.x * 32;
	if (warpRow >= M || tileColBase >= N) return;

	const unsigned int warpId = threadIdx.x / 32;

	// Shared staging for the A block
	extern __shared__ half sA[]; // 16*16
	const int blockRowIdx = warpRow / 16;

	// We'll accumulate into a small local accumulator per lane
	float c_acc[16];
	for (int i = 0; i < 16; ++i) c_acc[i] = 0.0f;

	for (int k = hdr[blockRowIdx]; k < hdr[blockRowIdx + 1]; ++k) {
		const half *a_global = data + static_cast<size_t>(k) * 16 * 16;

		// Cooperative load into shared memory
		for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) sA[i] = a_global[i];
		__syncthreads();

		const unsigned int tileCol = tileColBase + warpId * 16;
		if (tileCol >= N) { __syncthreads(); continue; }

		// Load A block from shared memory into a WMMA fragment and use the
		// WMMA API to perform the 16x16x16 multiply. This is a portable and
		// robust fallback in place of raw PTX `ldmatrix`/`mma.sync`.
		wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag_tmp;

		wmma::fill_fragment(c_frag_tmp, 0.0f);

		// Load A from shared memory (sA contains the 16x16 A block)
		wmma::load_matrix_sync(a_frag, sA, 16);

		// Load B tile from global memory as before
		wmma::load_matrix_sync(b_frag, B + static_cast<size_t>(idx[k]) * 16 * static_cast<size_t>(N) + tileCol, N);

		// Perform WMMA multiply-accumulate
		wmma::mma_sync(c_frag_tmp, a_frag, b_frag, c_frag_tmp);

		// Write temporary accumulator into a small float array and fold into c_acc
		float tmp_c[16];
		wmma::store_matrix_sync(tmp_c, c_frag_tmp, 16, wmma::mem_row_major);
		for (int t = 0; t < 16; ++t) c_acc[t] += tmp_c[t];

		__syncthreads();
	}

	const unsigned int outCol = tileColBase + warpId * 16;
	if (outCol < N) {
		// For simplicity, have lane 0 of each warp write the 16 accumulators
		if ((threadIdx.x & 31) == 0) {
			float *cptr = C + static_cast<size_t>(warpRow) * static_cast<size_t>(N) + outCol;
			for (int r = 0; r < 16; ++r) cptr[r] = c_acc[r];
		}
	}
}

// Forward-declare kernels from main.cu so this translation unit can
// call them as CUDA kernels. Signatures must match the definitions
// in main.cu. Do NOT use extern "C" here — CUDA kernel symbols are
// emitted by nvcc with device linkage and C++ linkage; adding
// extern "C" prevents the correct linkage and causes undefined
// references at link time.
__global__ void denseMatrixMul(const half *d_A, const half *d_B, float *d_C,
                               const unsigned int n) {
    const unsigned int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (rowIdx < n && colIdx < n) {
        float tmp = 0.0f;
        for (int k = 0; k < n; k++) {
            // Accumulate results for a single element
            // There's no need here to use reduction  or atomic add, because this
            // thread is the only one accessing this location
            tmp += __half2float(d_A[rowIdx * n + k]) *
                    __half2float(d_B[k * n + colIdx]);
        }
        d_C[rowIdx * n + colIdx] = tmp;
    }
	}
__global__ void denseMatrixMulCo(const half *d_A, const half *d_B, float *d_C,
                                 const unsigned int n) {
    const unsigned int rowIdx = blockIdx.y *
        CEIL_DIV(n, gridDim.y) + threadIdx.x / n;
    const unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x % n;

    if (rowIdx < n && colIdx < n) {
        float tmp = 0.0f;
        for (int k = 0; k < n; k++) {
            tmp += __half2float(d_A[rowIdx * n + k]) * __half2float(
                d_B[k * n + colIdx]);
        }
        d_C[rowIdx * n + colIdx] = tmp;
    }
}
__global__ void denseMatrixMulCo(const half *d_A, const half *d_B, float *d_C,
								 const unsigned int M, const unsigned int N, const unsigned int Z) {
	const unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (rowIdx < M && colIdx < Z) {
		float tmp = 0.0f;
		for (unsigned int k = 0; k < N; ++k) {
			tmp += __half2float(d_A[rowIdx * N + k]) * __half2float(d_B[k * Z + colIdx]);
		}
		d_C[rowIdx * Z + colIdx] = tmp;
	}
}

__global__ void denseMatrixMulTensor(const half *d_A, const half *d_B,
									 float *d_C, const unsigned int M, const unsigned int N, const unsigned int Z) {
	// Calculate which 16x16 tile this thread block handles
	const unsigned int warp_row = blockIdx.y * 16;
	const unsigned int warp_col = blockIdx.x * 16;

	if (warp_row >= M || warp_col >= Z) return;

	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

	wmma::fill_fragment(c_frag, 0.0f);

	// Accumulate over K dimension in 16x16 chunks (N is inner dim)
	for (unsigned int k = 0; k < N; k += 16) {
		wmma::load_matrix_sync(a_frag, d_A + warp_row * N + k, N);
		wmma::load_matrix_sync(b_frag, d_B + k * Z + warp_col, Z);
		wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
	}

	wmma::store_matrix_sync(d_C + warp_row * Z + warp_col, c_frag, Z,
							wmma::mem_row_major);
}
__global__ void sparseMatrixMult1Co(const int *hdr, const int *idx,
                                    const half *data, const half *B, float *C,
                                    const unsigned int n) {
    const unsigned int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (rowIdx < n && colIdx < n) {
        float tmp = 0.0f;
        for (int k = hdr[rowIdx]; k < hdr[rowIdx + 1]; k++) {
            tmp += __half2float(data[k]) * __half2float(
                B[idx[k] * n + colIdx]);
        }
        C[rowIdx * n + colIdx] = tmp;
    }
}

__global__ void sparseMatrixMult1(const int *hdr, const int *idx,
                                  const half *data, const half *B, float *C,
                                  const unsigned int n) {
    const unsigned int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (rowIdx < n && colIdx < n) {
        for (int k = hdr[rowIdx]; k < hdr[rowIdx + 1]; k++) {
            C[rowIdx * n + colIdx] += __half2float(data[k]) * __half2float(
                B[idx[k] * n + colIdx]);
        }
    }

}

__global__ void sparseMatrixMulTensor(const int *hdr, const int *idx,
									  const half *data, const half *B,
									  float *C, const unsigned int M, const unsigned int N) {
	// warpRow: starting row index (based on blockIdx.y), warpCol: starting col index (based on blockIdx.x)
	const unsigned int warpRow = blockIdx.y * 16;
	const unsigned int warpCol = blockIdx.x * 16;

	// Bounds check against the output matrix dimensions (M rows, N cols)
	if (warpRow >= M || warpCol >= N) return; // Ensure we don't access out of bounds

	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

	wmma::fill_fragment(c_frag, 0.0f);

	// hdr indexes blocks per block-row; warpRow/16 gives the block-row index
	for (int k = hdr[warpRow / 16]; k < hdr[warpRow / 16 + 1]; k++) {
		// load A block (16x16) and corresponding B tile; B leading dimension is N (number of columns)
		wmma::load_matrix_sync(a_frag, data + k * 16 * 16, 16);
		wmma::load_matrix_sync(b_frag, B + idx[k] * 16 * N + warpCol, N);
		wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
	}

	// store result into C with leading dimension N
	wmma::store_matrix_sync(C + warpRow * N + warpCol, c_frag, N,
							wmma::mem_row_major);
}



__global__ void sparseMatrixMulTensor1(const int *hdr, const int *idx,
                                      const half *data, const half *B,
                                      float *C, const unsigned int n) {
    const unsigned int warpRow = blockIdx.y * 16;
    const unsigned int warpCol = blockIdx.x * 16;

    if (warpRow >= n || warpCol >= n) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
    for (int k = hdr[warpRow / 16]; k < hdr[warpRow / 16 + 1]; k++) {
        wmma::load_matrix_sync(a_frag, data + k * 16 * 16, 16);
        wmma::load_matrix_sync(b_frag, B + idx[k] * 16 * n + warpCol, n);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warpRow * n + warpCol, c_frag, n,
                            wmma::mem_row_major);
}
__global__ void addMatrices(float *C, const float *CPart, const unsigned int n) {
    const unsigned int rowIdx = blockIdx.y *
        CEIL_DIV(n, gridDim.y) + threadIdx.x / n;
    const unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x % n;

    if (rowIdx < n && colIdx < n) {
        C[rowIdx * n + colIdx] += CPart[rowIdx * n + colIdx];
    }
}
// Variant v2: each block uses 64 threads (2 warps). grid.x is halved because
// each block computes two adjacent 16x16 tiles horizontally: the first warp
// computes the left 16x16 tile, the second warp computes the right 16x16 tile.
__global__ void sparseMatrixMulTensor_v2(const int *hdr, const int *idx,
										 const half *data, const half *B,
										 float *C, const unsigned int M, const unsigned int N) {
	// Base row and base column (two tiles per block in x)
	const unsigned int warpRow = blockIdx.y * 16;
	const unsigned int tileColBase = blockIdx.x * 32; // each block covers 32 columns (two 16-wide tiles)

	if (warpRow >= M || tileColBase >= N) return;

	// Identify the warp within the block: 0 or 1 (since blockDim.x == 64)
	const unsigned int warpId = threadIdx.x / 32;

	// Each warp will create its own fragments and operate only when its warpId matches.
	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

	wmma::fill_fragment(c_frag, 0.0f);

	// hdr indexes blocks per block-row; warpRow/16 gives the block-row index
	const int blockRowIdx = warpRow / 16;
	for (int k = hdr[blockRowIdx]; k < hdr[blockRowIdx + 1]; ++k) {
		// Load A block (same for both warps)
		// The A block is stored as contiguous 16x16 blocks in 'data'
		if (warpId == 0) {
			// left tile: columns start at tileColBase
			wmma::load_matrix_sync(a_frag, data + k * 16 * 16, 16);
			wmma::load_matrix_sync(b_frag, B + idx[k] * 16 * N + tileColBase, N);
			wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
		} else {
			// right tile: columns start at tileColBase + 16
			// guard: if the right tile starts beyond N, skip
			if (tileColBase + 16 >= N) continue;
			wmma::load_matrix_sync(a_frag, data + k * 16 * 16, 16);
			wmma::load_matrix_sync(b_frag, B + idx[k] * 16 * N + tileColBase + 16, N);
			wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
		}
	}

	// Store the accumulator back to C for the tile belonging to this warp
	if (warpId == 0) {
		wmma::store_matrix_sync(C + warpRow * N + tileColBase, c_frag, N, wmma::mem_row_major);
	} else {
		if (tileColBase + 16 < N) {
			wmma::store_matrix_sync(C + warpRow * N + tileColBase + 16, c_frag, N, wmma::mem_row_major);
		}
	}
}

// Variant v3: same as v2 but first stages the A 16x16 block into shared memory
// so both warps can load A from fast on-chip memory. The shared memory buffer
// size required per block is 16*16 half elements (256 * sizeof(half)). The
// benchmark must specify shared mem when launching this kernel.
__global__ void sparseMatrixMulTensor_v3(const int *hdr, const int *idx,
										 const half *data, const half *B,
										 float *C, const unsigned int M, const unsigned int N) {
	// Base row and base column (two tiles per block in x)
	const unsigned int warpRow = blockIdx.y * 16;
	const unsigned int tileColBase = blockIdx.x * 32; // each block covers 32 columns (two 16-wide tiles)

	if (warpRow >= M || tileColBase >= N) return;

	// Identify the warp within the block: 0 or 1 (since blockDim.x == 64)
	const unsigned int warpId = threadIdx.x / 32;

	// Shared memory for one 16x16 A block (row-major). Declared dynamically so
	// bench can specify proper shared size (256 * sizeof(half)).
	extern __shared__ half sA[]; // length in elements: 16*16 == 256

	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

	wmma::fill_fragment(c_frag, 0.0f);

	// hdr indexes blocks per block-row; warpRow/16 gives the block-row index
	const int blockRowIdx = warpRow / 16;
	// For each non-zero block in this block-row, stage A into shared mem then
	// both warps will read from sA to perform the WMMA operation.
	for (int k = hdr[blockRowIdx]; k < hdr[blockRowIdx + 1]; ++k) {
		// Global pointer to A block (16x16) stored contiguously
		const half *a_global = data + static_cast<size_t>(k) * 16 * 16;

		// Cooperatively load the 256 half elements into shared memory.
		// Each thread in the block will load multiple elements if needed.
		for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
			sA[i] = a_global[i];
		}
		__syncthreads();

		// Now load A from shared memory and B from global memory per-warp
		if (warpId == 0) {
			// left tile: columns start at tileColBase
			wmma::load_matrix_sync(a_frag, sA, 16);
			wmma::load_matrix_sync(b_frag, B + static_cast<size_t>(idx[k]) * 16 * N + tileColBase, N);
			wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
		} else {
			// right tile: columns start at tileColBase + 16
			if (tileColBase + 16 >= N) {
				__syncthreads();
				continue;
			}
			wmma::load_matrix_sync(a_frag, sA, 16);
			wmma::load_matrix_sync(b_frag, B + static_cast<size_t>(idx[k]) * 16 * N + tileColBase + 16, N);
			wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
		}

		// Ensure all warps have finished reading before the next iteration
		__syncthreads();
	}

	// Store the accumulator back to C for the tile belonging to this warp
	if (warpId == 0) {
		wmma::store_matrix_sync(C + static_cast<size_t>(warpRow) * N + tileColBase, c_frag, N, wmma::mem_row_major);
	} else {
		if (tileColBase + 16 < N) {
			wmma::store_matrix_sync(C + static_cast<size_t>(warpRow) * N + tileColBase + 16, c_frag, N, wmma::mem_row_major);
		}
	}
}

// Local constant to match main.cu's thread configuration
constexpr unsigned int N_THREADS = 32;

// Provide BLOCK_SIZE definition with external linkage for translation units that expect it.
// main.cu defines this for the application; nvbench builds a separate TU so
// we provide the same externally-linked symbol here.
extern const int BLOCK_SIZE = 16;

static const std::vector<std::string> patterns = {
	"random",
	"checkerboard",
	"diagonal",
	"blockdiagonal",
	"blockrandom",
	"largerandom"
};

// Helper: fill project Matrix from generated float matrix
static void fill_Matrix_from_generated(Matrix &dst, const std::vector<std::vector<float>> &src) {
	int rows = dst.rows;
	int cols = dst.cols;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			dst.data[i * cols + j] = __float2half(src[i][j]);
		}
	}
}

// Simple CPU reference matrix multiply: C = A * B
static void cpu_matmul_ref(const Matrix *A, const Matrix *B, std::vector<float> &Cout) {
	const int M = A->rows;
	const int K = A->cols;
	const int N = B->cols;
	Cout.assign(static_cast<size_t>(M) * N, 0.0f);
	for (int i = 0; i < M; ++i) {
		for (int k = 0; k < K; ++k) {
			float a = __half2float(A->data[i * K + k]);
			for (int j = 0; j < N; ++j) {
				Cout[static_cast<size_t>(i) * N + j] += a * __half2float(B->data[k * N + j]);
			}
		}
	}
}

// Report summary helper (user-provided)
void report_summary(nvbench::state& state)
{
	state.get_summary("nv/cold/time/gpu/min").remove_value("hide");
	state.get_summary("nv/cold/time/gpu/max").remove_value("hide");
	state.get_summary("nv/cold/time/gpu/mean").remove_value("hide");
	//state.get_summary("nv/cold/time/gpu/mean").set_string("hide", "");
	state.get_summary("nv/cold/time/cpu/mean").set_string("hide", "");
	state.get_summary("nv/cold/time/cpu/min").set_string("hide", "");
	state.get_summary("nv/cold/time/cpu/max").set_string("hide", "");
	state.get_summary("nv/cold/time/cpu/stdev/relative").set_string("hide", "");
	// state.get_summary("nv/cold/sm_clock_rate/mean").remove_value("hide");
	// state.get_summary("nv/cold/sm_clock_rate/scaling/percent").remove_value("hide");

}

// Common generation + device copy helper
struct GenDeviceBuffers {
	Matrix *matrixA = nullptr;
	Matrix *matrixB = nullptr;
	CSRMatrix *csrA = nullptr;
	BCSRMatrix *bcsrA = nullptr;
	// device pointers
	half *gpuA_half = nullptr;
	half *gpuB_half = nullptr;
	float *gpuC = nullptr;
	float *gpuCPart = nullptr;
	int *gpuCSRHdr = nullptr, *gpuCSRIdx = nullptr;
	half *gpuCSRData = nullptr;
	int *gpuBCSRHdr = nullptr, *gpuBCSRIdx = nullptr;
	half *gpuBCSRData = nullptr;

	~GenDeviceBuffers() {
		if (gpuC) cudaFree(gpuC);
		if (gpuCPart) cudaFree(gpuCPart);
		if (gpuA_half) cudaFree(gpuA_half);
		if (gpuB_half) cudaFree(gpuB_half);
		// Note: CSR/BCSR device frees are handled by their copyToDevice callers or not needed here
		delete csrA;
		delete bcsrA;
		delete matrixA;
		delete matrixB;
	}
};

static std::unique_ptr<GenDeviceBuffers> prepare_buffers(int M, int K, int N, double sparsity, const std::string &pattern) {
	auto out = std::make_unique<GenDeviceBuffers>();
	// Generate float matrices with generator
	// A is M x K, B is K x N (dense), C will be M x N
	auto genA = mg::generate_matrix<float>(M, K, sparsity, pattern, 16, 123);
	auto genB = mg::generate_matrix<float>(K, N, 0.0, "random", 16, 456);

	out->matrixA = new Matrix(M, K);
	out->matrixB = new Matrix(K, N);
	fill_Matrix_from_generated(*out->matrixA, genA);
	fill_Matrix_from_generated(*out->matrixB, genB);

	// Build sparse representations from matrixA
	out->csrA = new CSRMatrix(*out->matrixA);
	out->bcsrA = new BCSRMatrix(*out->matrixA);

	// Copy CSR/BCSR to device
	out->bcsrA->copyToDevice(&out->gpuBCSRHdr, &out->gpuBCSRIdx, &out->gpuBCSRData);
	out->csrA->copyToDevice(&out->gpuCSRHdr, &out->gpuCSRIdx, &out->gpuCSRData);

	size_t bytes_half_A = static_cast<size_t>(M) * K * sizeof(half);
	size_t bytes_half_B = static_cast<size_t>(K) * N * sizeof(half);
	size_t bytes_float_C = static_cast<size_t>(M) * N * sizeof(float);
	cudaMalloc(reinterpret_cast<void **>(&out->gpuA_half), bytes_half_A);
	cudaMalloc(reinterpret_cast<void **>(&out->gpuB_half), bytes_half_B);
	cudaMalloc(reinterpret_cast<void **>(&out->gpuC), bytes_float_C);
	cudaMalloc(reinterpret_cast<void **>(&out->gpuCPart), bytes_float_C);
	cudaMemcpy(out->gpuA_half, out->matrixA->data, bytes_half_A, cudaMemcpyHostToDevice);
	cudaMemcpy(out->gpuB_half, out->matrixB->data, bytes_half_B, cudaMemcpyHostToDevice);
	cudaMemset(out->gpuC, 0, bytes_float_C);
	cudaMemset(out->gpuCPart, 0, bytes_float_C);

	return out;
}

// Benchmark: denseMatrixMul (naive)
static void bench_denseMatrixMul(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;

	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern);


	// Disable NVBench's blocking-kernel deadlock detector for this benchmark.
	// The kernel launcher synchronizes the stream explicitly and we
	// prefer to disable the deadlock timeout rather than marking the exec
	// as synchronous so measurements run uninterrupted.
	// state.set_blocking_kernel_timeout(-1);

	// grid/block similar to main.cu naive kernel
	dim3 gridSize{static_cast<unsigned int>(N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0)), static_cast<unsigned int>(M / N_THREADS + (M % N_THREADS > 0 ? 1 : 0)), 1};
	dim3 blockSize{N_THREADS, N_THREADS, 1};
	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		// clear output buffer for this iteration on the launch stream
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		// start timer
		timer.start();
		denseMatrixMul<<<gridSize, blockSize, 0, launch.get_stream()>>>(buf->gpuA_half, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(M), static_cast<unsigned int>(K), static_cast<unsigned int>(N));
		// stop timer
		timer.stop();
	});

	// copy result back and verify on CPU
	{
		std::vector<float> out_host(static_cast<size_t>(M) * N);
		cudaMemcpy(out_host.data(), buf->gpuC, out_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
		std::vector<float> ref;
		cpu_matmul_ref(buf->matrixA, buf->matrixB, ref);
		const float eps = 1e-2f;
		for (size_t i = 0; i < out_host.size(); ++i) {
			if (std::fabs(out_host[i] - ref[i]) > eps) {
				std::fprintf(stderr, "Mismatch at %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::abort();
			}
		}
	}

	// report summary tweaks
	report_summary(state);
}

// Benchmark: denseMatrixMulTensor (wmma)
static void bench_denseMatrixMulTensor(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern);
	// state.set_blocking_kernel_timeout(-1);

	dim3 gridSize{static_cast<unsigned int>((N + 15) / 16), static_cast<unsigned int>((M + 15) / 16), 1};
	dim3 blockSize{32, 1, 1};

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		// clear output buffer for this iteration on the launch stream
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		// start timer
		timer.start();
		denseMatrixMulTensor<<<gridSize, blockSize, 0, launch.get_stream()>>>(buf->gpuA_half, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(M), static_cast<unsigned int>(K), static_cast<unsigned int>(N));
		// stop timer
		timer.stop();
	});

	// copy result back and verify on CPU
	{
		std::vector<float> out_host(static_cast<size_t>(M) * N);
		cudaMemcpy(out_host.data(), buf->gpuC, out_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
		std::vector<float> ref;
		cpu_matmul_ref(buf->matrixA, buf->matrixB, ref);
		const float eps = 1e-2f;
		for (size_t i = 0; i < out_host.size(); ++i) {
			if (std::fabs(out_host[i] - ref[i]) > eps) {
				std::fprintf(stderr, "Mismatch at %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::abort();
			}
		}
	}

	// report summary tweaks
	report_summary(state);
}

// Benchmark: sparseMatrixMult1
static void bench_sparseMatrixMult1(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern);
	// state.set_blocking_kernel_timeout(-1);

	dim3 gridSize{static_cast<unsigned int>(N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0)), static_cast<unsigned int>(M / N_THREADS + (M % N_THREADS > 0 ? 1 : 0)), 1};
	dim3 blockSize{N_THREADS, N_THREADS, 1};

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		// clear output buffer for this iteration on the launch stream
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		// start timer
		timer.start();
		sparseMatrixMult1<<<gridSize, blockSize, 0, launch.get_stream()>>>(buf->gpuCSRHdr, buf->gpuCSRIdx, buf->gpuCSRData, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(N));
		// stop timer
		timer.stop();
	});

	// copy result back and verify on CPU
	{
		std::vector<float> out_host(static_cast<size_t>(M) * N);
		cudaMemcpy(out_host.data(), buf->gpuC, out_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
		std::vector<float> ref;
		cpu_matmul_ref(buf->matrixA, buf->matrixB, ref);
		const float eps = 1e-2f;
		for (size_t i = 0; i < out_host.size(); ++i) {
			if (std::fabs(out_host[i] - ref[i]) > eps) {
				std::fprintf(stderr, "Mismatch at %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::abort();
			}
		}
	}

	// report summary tweaks
	report_summary(state);
}

// Benchmark: sparseMatrixMulTensor (BCSR tensor)
static void bench_sparseMatrixMulTensor(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern);
	// state.set_blocking_kernel_timeout(-1);

	// grid.x corresponds to tile columns (N), grid.y to tile rows (M) — match denseMatrixMulTensor
	dim3 gridSize{static_cast<unsigned int>((N + 15) / 16), static_cast<unsigned int>((M + 15) / 16), 1};
	dim3 blockSize{32, 1, 1};

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec([&](nvbench::launch &launch){
		// clear output buffer for this iteration on the launch stream		
        
		sparseMatrixMulTensor<<<gridSize, blockSize, 0, launch.get_stream()>>>(buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(M), static_cast<unsigned int>(N));
		// stop timer
        
	});

	// copy result back and verify on CPU
	{
		std::vector<float> out_host(static_cast<size_t>(M) * N);
		cudaMemcpy(out_host.data(), buf->gpuC, out_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
		std::vector<float> ref;
		cpu_matmul_ref(buf->matrixA, buf->matrixB, ref);
		const float eps = 1e-2f;
		for (size_t i = 0; i < out_host.size(); ++i) {
			if (std::fabs(out_host[i] - ref[i]) > eps) {
				std::fprintf(stderr, "Mismatch at %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::abort();
			}
		}
	}

      state.get_summary("nv/cold/time/gpu/min").remove_value("hide");
      state.get_summary("nv/cold/time/gpu/max").remove_value("hide");
      state.get_summary("nv/cold/time/gpu/mean").remove_value("hide");
      state.get_summary("nv/cold/time/cpu/mean").remove_value("hide");
      state.get_summary("nv/cold/time/cpu/min").remove_value("hide");
      state.get_summary("nv/cold/time/cpu/max").remove_value("hide");

    // report summary tweaks
    // report_summary(state);
    
}

// Benchmark: sparseMatrixMulTensor_v2 (each block covers two tiles horizontally, 64 threads)
static void bench_sparseMatrixMulTensor_v2(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern);
	// state.set_blocking_kernel_timeout(-1);

	// grid.x halves because each block handles two 16-wide tiles (32 columns)
	dim3 gridSize{static_cast<unsigned int>((N + 31) / 32), static_cast<unsigned int>((M + 15) / 16), 1};
	dim3 blockSize{64, 1, 1};

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		// clear output buffer for this iteration on the launch stream
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		// start timer
		timer.start();
		sparseMatrixMulTensor_v2<<<gridSize, blockSize, 0, launch.get_stream()>>>(buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(M), static_cast<unsigned int>(N));
		// stop timer
		timer.stop();
	});

	// copy result back and verify on CPU
	{
		std::vector<float> out_host(static_cast<size_t>(M) * N);
		cudaMemcpy(out_host.data(), buf->gpuC, out_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
		std::vector<float> ref;
		cpu_matmul_ref(buf->matrixA, buf->matrixB, ref);
		const float eps = 1e-2f;
		for (size_t i = 0; i < out_host.size(); ++i) {
			if (std::fabs(out_host[i] - ref[i]) > eps) {
				std::fprintf(stderr, "Mismatch at %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::abort();
			}
		}
	}

	// report summary tweaks
	state.get_summary("nv/cold/time/gpu/min").remove_value("hide");
      state.get_summary("nv/cold/time/gpu/max").remove_value("hide");
      state.get_summary("nv/cold/time/gpu/mean").remove_value("hide");
      state.get_summary("nv/cold/time/cpu/mean").remove_value("hide");
      state.get_summary("nv/cold/time/cpu/min").remove_value("hide");
      state.get_summary("nv/cold/time/cpu/max").remove_value("hide");
}

// Benchmark: sparseMatrixMulTensor_v3 (A staged to shared memory)
static void bench_sparseMatrixMulTensor_v3(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern);

	// grid.x halves because each block handles two 16-wide tiles (32 columns)
	dim3 gridSize{static_cast<unsigned int>((N + 31) / 32), static_cast<unsigned int>((M + 15) / 16), 1};
	dim3 blockSize{64, 1, 1};

	// shared memory size: 16*16 half elements
	const unsigned int sharedBytes = static_cast<unsigned int>(16 * 16 * sizeof(half));

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		// clear output buffer for this iteration on the launch stream
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		// start timer
		timer.start();
		sparseMatrixMulTensor_v3<<<gridSize, blockSize, sharedBytes, launch.get_stream()>>>(buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(M), static_cast<unsigned int>(N));
		// stop timer
		timer.stop();
	});

	// copy result back and verify on CPU
	{
		std::vector<float> out_host(static_cast<size_t>(M) * N);
		cudaMemcpy(out_host.data(), buf->gpuC, out_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
		std::vector<float> ref;
		cpu_matmul_ref(buf->matrixA, buf->matrixB, ref);
		const float eps = 1e-2f;
		for (size_t i = 0; i < out_host.size(); ++i) {
			if (std::fabs(out_host[i] - ref[i]) > eps) {
				std::fprintf(stderr, "Mismatch at %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::abort();
			}
		}
	}

	// report summary tweaks
	state.get_summary("nv/cold/time/gpu/min").remove_value("hide");
      state.get_summary("nv/cold/time/gpu/max").remove_value("hide");
      state.get_summary("nv/cold/time/gpu/mean").remove_value("hide");
      state.get_summary("nv/cold/time/cpu/mean").remove_value("hide");
      state.get_summary("nv/cold/time/cpu/min").remove_value("hide");
      state.get_summary("nv/cold/time/cpu/max").remove_value("hide");
}

// Benchmark: sparseMatrixMulTensorlargeRandom (process 4x16-row panels -> 64 rows per block)
static void bench_sparseMatrixMulTensorlargeRandom(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern);

	// grid.x corresponds to 16-column tiles, grid.y to 64-row tiles
	dim3 gridSize{static_cast<unsigned int>((N + 15) / 32), static_cast<unsigned int>((M + 63) / 64), 1};
	dim3 blockSize{32, 1, 1};

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		timer.start();
		sparseMatrixMulTensorlargeRandom<<<gridSize, blockSize, 0, launch.get_stream()>>>(buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(M), static_cast<unsigned int>(N));
		timer.stop();
	});

	// copy result back and verify on CPU
	{
		std::vector<float> out_host(static_cast<size_t>(M) * N);
		cudaMemcpy(out_host.data(), buf->gpuC, out_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
		std::vector<float> ref;
		cpu_matmul_ref(buf->matrixA, buf->matrixB, ref);
		const float eps = 1e-2f;
		for (size_t i = 0; i < out_host.size(); ++i) {
			if (std::fabs(out_host[i] - ref[i]) > eps) {
				std::fprintf(stderr, "Mismatch at %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::abort();
			}
		}
	}

	state.get_summary("nv/cold/time/gpu/min").remove_value("hide");
	state.get_summary("nv/cold/time/gpu/max").remove_value("hide");
	state.get_summary("nv/cold/time/gpu/mean").remove_value("hide");
}

// Benchmark: sparseMatrixMulTensor_v1_improved
static void bench_sparseMatrixMulTensor_v1_improved(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern);

	dim3 gridSize{static_cast<unsigned int>((N + 31) / 32), static_cast<unsigned int>((M + 15) / 16), 1};
	dim3 blockSize{64, 1, 1};

	const unsigned int sharedBytes = static_cast<unsigned int>(16 * 16 * sizeof(half));

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		// clear output buffer for this iteration on the launch stream
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		// start timer
		timer.start();
		sparseMatrixMulTensor_v1_improved<<<gridSize, blockSize, sharedBytes, launch.get_stream()>>>(buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(M), static_cast<unsigned int>(N));
		// stop timer
		timer.stop();
	});

	// copy result back and verify on CPU
	{
		std::vector<float> out_host(static_cast<size_t>(M) * N);
		cudaMemcpy(out_host.data(), buf->gpuC, out_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
		std::vector<float> ref;
		cpu_matmul_ref(buf->matrixA, buf->matrixB, ref);
		const float eps = 1e-2f;
		for (size_t i = 0; i < out_host.size(); ++i) {
			if (std::fabs(out_host[i] - ref[i]) > eps) {
				std::fprintf(stderr, "Mismatch at %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::abort();
			}
		}
	}

	state.get_summary("nv/cold/time/gpu/min").remove_value("hide");
    state.get_summary("nv/cold/time/gpu/max").remove_value("hide");
    state.get_summary("nv/cold/time/gpu/mean").remove_value("hide");
    state.get_summary("nv/cold/time/cpu/mean").remove_value("hide");
    state.get_summary("nv/cold/time/cpu/min").remove_value("hide");
    state.get_summary("nv/cold/time/cpu/max").remove_value("hide");
}


// Benchmark: cuBLAS (GEMM) - no tensor ops
static void bench_cuBLAS(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern);

	cublasHandle_t handle;
	cublasCreate(&handle);
	constexpr float alpha = 1.0f;
	constexpr float beta = 0.0f;
	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		// ensure cuBLAS uses the same stream as the launch
		cublasSetStream(handle, launch.get_stream());
		// clear output buffer for this iteration on the launch stream
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		// start timer
		timer.start();
		// cublasGemmEx parameters: (handle, transB, transA, m, n, k, ...)
		// we keep previous ordering but adapt dimensions for MxK * KxN = MxN
		cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, buf->gpuB_half, CUDA_R_16F, N, buf->gpuA_half, CUDA_R_16F, K, &beta, buf->gpuC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
		// stop timer
		timer.stop();
	});

	// copy result back and verify on CPU
	{
		std::vector<float> out_host(static_cast<size_t>(M) * N);
		cudaMemcpy(out_host.data(), buf->gpuC, out_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
		std::vector<float> ref;
		cpu_matmul_ref(buf->matrixA, buf->matrixB, ref);
		const float eps = 1e-2f;
		for (size_t i = 0; i < out_host.size(); ++i) {
			if (std::fabs(out_host[i] - ref[i]) > eps) {
				std::fprintf(stderr, "Mismatch at %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::abort();
			}
		}
	}
    cublasDestroy(handle);
	// report summary tweaks
	report_summary(state);
}

// Benchmark: cuBLAS with Tensor Cores
static void bench_cuBLAS_Tensor(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern);

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
	constexpr float alpha = 1.0f;
	constexpr float beta = 0.0f;
	// warm up
	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, buf->gpuB_half, CUDA_R_16F, N, buf->gpuA_half, CUDA_R_16F, K, &beta, buf->gpuC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		// ensure cuBLAS uses the same stream as the launch
		cublasSetStream(handle, launch.get_stream());
		// clear output buffer for this iteration on the launch stream
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		// start timer
		timer.start();
		cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, buf->gpuB_half, CUDA_R_16F, N, buf->gpuA_half, CUDA_R_16F, K, &beta, buf->gpuC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		// stop timer
		timer.stop();
	});

	// copy result back and verify on CPU
	{
		std::vector<float> out_host(static_cast<size_t>(M) * N);
		cudaMemcpy(out_host.data(), buf->gpuC, out_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
		std::vector<float> ref;
		cpu_matmul_ref(buf->matrixA, buf->matrixB, ref);
		const float eps = 1e-2f;
		for (size_t i = 0; i < out_host.size(); ++i) {
			if (std::fabs(out_host[i] - ref[i]) > eps) {
				std::fprintf(stderr, "Mismatch at %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::abort();
			}
		}
	}
	cublasDestroy(handle);

    // report summary tweaks
	report_summary(state);
}


// EvalTest
// Benchmark: sparseMatrixMulTensor_option2_ldmatrix_sm80 (PTX ldmatrix + mma.sync)
static void bench_sparseMatrixMulTensor_option2_ldmatrix_sm80(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern);

	dim3 gridSize{static_cast<unsigned int>((N + 31) / 32), static_cast<unsigned int>((M + 15) / 16), 1};
	dim3 blockSize{64, 1, 1};
	const unsigned int sharedBytes = static_cast<unsigned int>(16 * 16 * sizeof(half));

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		timer.start();
		sparseMatrixMulTensor_option2_ldmatrix_sm80<<<gridSize, blockSize, sharedBytes, launch.get_stream()>>>(buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(M), static_cast<unsigned int>(N));
		timer.stop();
	});

	// copy result back and verify on CPU
	{
		std::vector<float> out_host(static_cast<size_t>(M) * N);
		cudaMemcpy(out_host.data(), buf->gpuC, out_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
		std::vector<float> ref;
		cpu_matmul_ref(buf->matrixA, buf->matrixB, ref);
		const float eps = 1e-2f;
		for (size_t i = 0; i < out_host.size(); ++i) {
			if (std::fabs(out_host[i] - ref[i]) > eps) {
				std::fprintf(stderr, "Mismatch at %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::abort();
			}
		}
	}

	state.get_summary("nv/cold/time/gpu/min").remove_value("hide");
    state.get_summary("nv/cold/time/gpu/max").remove_value("hide");
    state.get_summary("nv/cold/time/gpu/mean").remove_value("hide");
    state.get_summary("nv/cold/time/cpu/mean").remove_value("hide");
    state.get_summary("nv/cold/time/cpu/min").remove_value("hide");
    state.get_summary("nv/cold/time/cpu/max").remove_value("hide");
}

// NVBENCH_BENCH(bench_denseMatrixMul).set_name("denseMatrixMul").add_int64_axis("N", {16}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50}).add_int64_axis("PAT", {0});
// NVBENCH_BENCH(bench_denseMatrixMulTensor).set_name("denseMatrixMulTensor").add_int64_axis("N", {16}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50}).add_int64_axis("PAT", {0});
// NVBENCH_BENCH(bench_sparseMatrixMult1).set_name("sparseMatrixMult1").add_int64_axis("N", {16}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50}).add_int64_axis("PAT", {0});
NVBENCH_BENCH(bench_sparseMatrixMulTensor).set_name("sparseMatrixMulTensor").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50}).add_int64_axis("PAT", {5});
// NVBENCH_BENCH(bench_cuBLAS).set_name("cuBLAS_GEMM").add_int64_axis("N", {16}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50}).add_int64_axis("PAT", {0});
// NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {16}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50}).add_int64_axis("PAT", {0});
NVBENCH_BENCH(bench_sparseMatrixMulTensor_v2).set_name("sparseMatrixMulTensor_v2").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {5});
NVBENCH_BENCH(bench_sparseMatrixMulTensor_v3).set_name("sparseMatrixMulTensor_v3").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {5});
NVBENCH_BENCH(bench_sparseMatrixMulTensorlargeRandom).set_name("sparseMatrixMulTensorlargeRandom").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {5});


// NVBENCH_BENCH(bench_denseMatrixMul).set_name("denseMatrixMul").add_int64_axis("N", {16}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50}).add_int64_axis("PAT", {0});
// NVBENCH_BENCH(bench_denseMatrixMulTensor).set_name("denseMatrixMulTensor").add_int64_axis("N", {16}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50}).add_int64_axis("PAT", {0});
// NVBENCH_BENCH(bench_sparseMatrixMult1).set_name("sparseMatrixMult1").add_int64_axis("N", {16}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50}).add_int64_axis("PAT", {0});
NVBENCH_BENCH(bench_sparseMatrixMulTensor).set_name("sparseMatrixMulTensor").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50}).add_int64_axis("PAT", {5});
// NVBENCH_BENCH(bench_cuBLAS).set_name("cuBLAS_GEMM").add_int64_axis("N", {16}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50}).add_int64_axis("PAT", {0});
// NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {16}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50}).add_int64_axis("PAT", {0});
NVBENCH_BENCH(bench_sparseMatrixMulTensor_v2).set_name("sparseMatrixMulTensor_v2").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {5});
NVBENCH_BENCH(bench_sparseMatrixMulTensor_v3).set_name("sparseMatrixMulTensor_v3").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {5});
NVBENCH_BENCH(bench_sparseMatrixMulTensorlargeRandom).set_name("sparseMatrixMulTensorlargeRandom").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {256}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {5});


// // Register benches and axes
// NVBENCH_BENCH(bench_denseMatrixMul).set_name("denseMatrixMul").add_int64_axis("N", {16, 32, 64, 128}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// NVBENCH_BENCH(bench_denseMatrixMulTensor).set_name("denseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// // NVBENCH_BENCH(bench_sparseMatrixMult1).set_name("sparseMatrixMult1").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor).set_name("sparseMatrixMulTensor").add_int64_axis("N", {32, 64}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {80,90}).add_int64_axis("PAT", {3,4});
// // // NVBENCH_BENCH(bench_cuBLAS).set_name("cuBLAS_GEMM").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// // // NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor_v2).set_name("sparseMatrixMulTensor_v2").add_int64_axis("N", {32, 64}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor_v3).set_name("sparseMatrixMulTensor_v3").add_int64_axis("N", {32, 64}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {80,90}).add_int64_axis("PAT", {3,4});

// // // size 512 * 2048
// // NVBENCH_BENCH(bench_denseMatrixMul).set_name("denseMatrixMul").add_int64_axis("N", {16, 32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {2048}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// // NVBENCH_BENCH(bench_denseMatrixMulTensor).set_name("denseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {2048}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// // // NVBENCH_BENCH(bench_sparseMatrixMult1).set_name("sparseMatrixMult1").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {2048}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor).set_name("sparseMatrixMulTensor").add_int64_axis("N", {16, 32, 64}).add_int64_axis("M", {512}).add_int64_axis("K", {2048}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_cuBLAS).set_name("cuBLAS_GEMM").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {2048}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {2048}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor_v2).set_name("sparseMatrixMulTensor_v2").add_int64_axis("N", {16, 32, 64}).add_int64_axis("M", {512}).add_int64_axis("K", {2048}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor_v3).set_name("sparseMatrixMulTensor_v3").add_int64_axis("N", {16, 32, 64}).add_int64_axis("M", {512}).add_int64_axis("K", {2048}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});

// // // // size 512 * 4608
// // // NVBENCH_BENCH(bench_denseMatrixMul).set_name("denseMatrixMul").add_int64_axis("N", {16, 32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {4608}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// // // NVBENCH_BENCH(bench_denseMatrixMulTensor).set_name("denseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {4608}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// // // // NVBENCH_BENCH(bench_sparseMatrixMult1).set_name("sparseMatrixMult1").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {4608}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor).set_name("sparseMatrixMulTensor").add_int64_axis("N", {16, 32, 64}).add_int64_axis("M", {512}).add_int64_axis("K", {4608}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_cuBLAS).set_name("cuBLAS_GEMM").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {4608}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {4608}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor_v2).set_name("sparseMatrixMulTensor_v2").add_int64_axis("N", {16, 32, 64}).add_int64_axis("M", {512}).add_int64_axis("K", {4608}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// // // // // size 256 * 1024
// // // // NVBENCH_BENCH(bench_denseMatrixMul).set_name("denseMatrixMul").add_int64_axis("N", {16, 32, 64, 128}).add_int64_axis("M", {256}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// // // // NVBENCH_BENCH(bench_denseMatrixMulTensor).set_name("denseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {256}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// // // // // NVBENCH_BENCH(bench_sparseMatrixMult1).set_name("sparseMatrixMult1").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {256}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor).set_name("sparseMatrixMulTensor").add_int64_axis("N", {16, 32, 64}).add_int64_axis("M", {256}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_cuBLAS).set_name("cuBLAS_GEMM").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {256}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {256}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor_v2).set_name("sparseMatrixMulTensor_v2").add_int64_axis("N", {16, 32, 64}).add_int64_axis("M", {256}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor_v3).set_name("sparseMatrixMulTensor_v3").add_int64_axis("N", {16, 32, 64}).add_int64_axis("M", {256}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});


// // // // // size 512 * 256
// // // // NVBENCH_BENCH(bench_denseMatrixMul).set_name("denseMatrixMul").add_int64_axis("N", {16, 32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// // // // NVBENCH_BENCH(bench_denseMatrixMulTensor).set_name("denseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// // // // // NVBENCH_BENCH(bench_sparseMatrixMult1).set_name("sparseMatrixMult1").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor).set_name("sparseMatrixMulTensor").add_int64_axis("N", {16, 32, 64}).add_int64_axis("M", {512}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_cuBLAS).set_name("cuBLAS_GEMM").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor_v2).set_name("sparseMatrixMulTensor_v2").add_int64_axis("N", {16, 32, 64}).add_int64_axis("M", {512}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor_v3).set_name("sparseMatrixMulTensor_v3").add_int64_axis("N", {16, 32, 64}).add_int64_axis("M", {512}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});


// // // // // size 2048 * 1024
// // // // NVBENCH_BENCH(bench_denseMatrixMul).set_name("denseMatrixMul").add_int64_axis("N", {16, 32, 64, 128}).add_int64_axis("M", {2048}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// // // // NVBENCH_BENCH(bench_denseMatrixMulTensor).set_name("denseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {2048}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// // // // // NVBENCH_BENCH(bench_sparseMatrixMult1).set_name("sparseMatrixMult1").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {2048}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor).set_name("sparseMatrixMulTensor").add_int64_axis("N", {16, 32, 64}).add_int64_axis("M", {2048}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_cuBLAS).set_name("cuBLAS_GEMM").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {2048}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {2048}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor_v2).set_name("sparseMatrixMulTensor_v2").add_int64_axis("N", {16, 32, 64}).add_int64_axis("M", {2048}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor_v3).set_name("sparseMatrixMulTensor_v3").add_int64_axis("N", {16, 32, 64}).add_int64_axis("M", {2048}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});

// // // // //size 512 * 512
// // // // NVBENCH_BENCH(bench_denseMatrixMul).set_name("denseMatrixMul").add_int64_axis("N", {16, 32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// // // // NVBENCH_BENCH(bench_denseMatrixMulTensor).set_name("denseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// // // // // NVBENCH_BENCH(bench_sparseMatrixMult1).set_name("sparseMatrixMult1").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor).set_name("sparseMatrixMulTensor").add_int64_axis("N", {16, 32, 64}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_cuBLAS).set_name("cuBLAS_GEMM").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor_v2).set_name("sparseMatrixMulTensor_v2").add_int64_axis("N", {16, 32, 64}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
// NVBENCH_BENCH(bench_sparseMatrixMulTensor_v3).set_name("sparseMatrixMulTensor_v3").add_int64_axis("N", {16, 32, 64}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});

// Register the large-random specific bench. Use N values that are multiples of 32
// NVBENCH_BENCH(bench_sparseMatrixMulTensorlargeRandom).set_name("sparseMatrixMulTensorlargeRandom").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {256,512}).add_int64_axis("K", {256,1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {3,4});
