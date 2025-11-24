#include "cuda_kernels.cuh"

// Vectorized loader for B (dense) into shared tile.
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_K,
		  size_t NUM_THREADS, size_t BLOCK_TILE_SKEW_SIZE_X = 0U,
		  typename VECTOR_TYPE = int4>
__device__ void load_B_to_shared_vectorized(
	T const* B, size_t ldb,
	T B_thread_block_tile[BLOCK_TILE_SIZE_K]
						 [BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
	size_t thread_block_tile_idx, size_t thread_linear_idx, size_t n, size_t k)
{
	constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTOR_TYPE) / sizeof(T)};
	static_assert(sizeof(VECTOR_TYPE) % sizeof(T) == 0U);
	static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
	static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);
	constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_X{BLOCK_TILE_SIZE_X / NUM_VECTOR_UNITS};

	static_assert((BLOCK_TILE_SIZE_X) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
	static_assert((BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);

	// Each thread cooperatively loads vectors forming the B tile.
#pragma unroll
	for (size_t load_idx{0U};
		 load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) / NUM_THREADS;
		 ++load_idx)
	{
		size_t const B_thread_block_tile_row_idx{
			(thread_linear_idx + load_idx * NUM_THREADS) / VECTORIZED_BLOCK_TILE_SIZE_X};
		size_t const B_thread_block_tile_col_idx{
			((thread_linear_idx + load_idx * NUM_THREADS) % VECTORIZED_BLOCK_TILE_SIZE_X) * NUM_VECTOR_UNITS};

		size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K + B_thread_block_tile_row_idx};
		size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X + B_thread_block_tile_col_idx};

		VECTOR_TYPE B_row_vector_vals{};
		if (B_row_idx < k && B_col_idx < n) {
			B_row_vector_vals = *reinterpret_cast<VECTOR_TYPE const*>(&B[B_row_idx * ldb + B_col_idx]);
		}
		if (B_col_idx + NUM_VECTOR_UNITS > n) {
			size_t const num_invalid_elements{B_col_idx + NUM_VECTOR_UNITS - n};
			T* const B_row_vector_vals_ptr{reinterpret_cast<T*>(&B_row_vector_vals)};
			for (size_t i{0U}; i < num_invalid_elements; ++i) {
				B_row_vector_vals_ptr[NUM_VECTOR_UNITS - 1U - i] = static_cast<T>(0);
			}
		}

		if (B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K &&
			B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_X) {
			*reinterpret_cast<VECTOR_TYPE*>(&B_thread_block_tile[B_thread_block_tile_row_idx]
											  [B_thread_block_tile_col_idx]) = B_row_vector_vals;
		}
	}
}

// Vectorized loader for a 32x16 A block into shared memory (row-major).
// - block_ptr: pointer to global block (32 rows x 16 cols) stored row-major
// - sA_ptr: pointer to shared memory buffer (same layout)
// - thread_linear_idx, num_threads: used to distribute vector loads across threads
template <typename T, typename VECTOR_TYPE = int4>
__device__ void load_A_block_32x16_vectorized(const T* block_ptr, T* sA_ptr,
											  unsigned int thread_linear_idx,
											  unsigned int num_threads)
{
	using VECTOR_TYPE_LOCAL = VECTOR_TYPE;
	constexpr size_t BLOCK_ROWS = 32u;
	constexpr size_t BLOCK_COLS = 16u;
	constexpr size_t NUM_VECTOR_UNITS = sizeof(VECTOR_TYPE_LOCAL) / sizeof(T);
	static_assert(BLOCK_COLS % NUM_VECTOR_UNITS == 0U, "BLOCK_COLS must be divisible by vector width");
	constexpr size_t VECTORIZED_COLS = BLOCK_COLS / NUM_VECTOR_UNITS;

	const size_t total_vector_loads = BLOCK_ROWS * VECTORIZED_COLS;
	for (size_t load = thread_linear_idx; load < total_vector_loads; load += num_threads) {
		const size_t row = load / VECTORIZED_COLS;
		const size_t vcol = load % VECTORIZED_COLS;
		const size_t col = vcol * NUM_VECTOR_UNITS;
		// read vector from global block (row-major)
		VECTOR_TYPE_LOCAL vec = *reinterpret_cast<const VECTOR_TYPE_LOCAL*>(block_ptr + row * BLOCK_COLS + col);
		// write vector into shared memory (same row-major layout)
		*reinterpret_cast<VECTOR_TYPE_LOCAL*>(sA_ptr + row * BLOCK_COLS + col) = vec;
	}
}


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

// Original largerandom kernel implementation preserved for comparison/testing
__global__ void sparseMatrixMulTensorlargeRandom_old(const int *hdr, const int *idx,
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
}// Dense matrix multiplication for square matrices

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

// Variant: largeRandom-specific kernel. Processes 64x16 BCSR blocks directly.
// Each BCSR block is a single 64x16 dense block, not 4 separate 16x16 blocks.
// We decompose the 64x16 block into 4 WMMA operations vertically (64/16=4).
__global__ void sparseMatrixMulTensor64x16(const int *hdr, const int *idx,
												 const half *data, const half *B,
												 float *C, const unsigned int M, const unsigned int N) {
	// Each block covers 64 rows and 32 columns (for B reuse)
	const unsigned int blockRow64 = blockIdx.y * 64u;
	const unsigned int tileCol = blockIdx.x * 32u;

	if (blockRow64 >= M || tileCol >= N) return;

	// For 64x16 BCSR blocks, we need 4 WMMA operations vertically and 2 horizontally for B reuse
	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag0, a_frag1, a_frag2, a_frag3;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag, b_frag1;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1, c2, c3, c4, c5, c6, c7;

	wmma::fill_fragment(c0, 0.0f);  // rows 0-15, cols 0-15
	wmma::fill_fragment(c1, 0.0f);  // rows 16-31, cols 0-15
	wmma::fill_fragment(c2, 0.0f);  // rows 32-47, cols 0-15
	wmma::fill_fragment(c3, 0.0f);  // rows 48-63, cols 0-15
	wmma::fill_fragment(c4, 0.0f);  // rows 0-15, cols 16-31
	wmma::fill_fragment(c5, 0.0f);  // rows 16-31, cols 16-31
	wmma::fill_fragment(c6, 0.0f);  // rows 32-47, cols 16-31
	wmma::fill_fragment(c7, 0.0f);  // rows 48-63, cols 16-31

	// Block row index for the 64-row block (each BCSR entry represents one 64x16 block)
	const int blockRowIdx = static_cast<int>(blockRow64 / 64u);

	// Iterate over non-zero 64x16 BCSR blocks in this block-row
	for (int k = hdr[blockRowIdx]; k < hdr[blockRowIdx + 1]; ++k) {
		// Load B fragments (same for all A sub-blocks)
		const size_t b_off = static_cast<size_t>(idx[k]) * 16u * static_cast<size_t>(N) + static_cast<size_t>(tileCol);
		wmma::load_matrix_sync(b_frag, B + b_off, N);
		if (tileCol + 16 < N) {
			wmma::load_matrix_sync(b_frag1, B + b_off + 16, N);
		}

		// The 64x16 BCSR block is stored as a contiguous 64x16 matrix in data
		// We need to load 4 different 16x16 sub-blocks from this 64x16 block
		const half *block_data = data + static_cast<size_t>(k) * 64u * 16u;

		// Load A fragments from different vertical positions within the 64x16 block
		wmma::load_matrix_sync(a_frag0, block_data + 0 * 16u * 16u, 16);   // rows 0-15
		wmma::load_matrix_sync(a_frag1, block_data + 1 * 16u * 16u, 16);   // rows 16-31
		wmma::load_matrix_sync(a_frag2, block_data + 2 * 16u * 16u, 16);   // rows 32-47
		wmma::load_matrix_sync(a_frag3, block_data + 3 * 16u * 16u, 16);   // rows 48-63

		// Perform WMMA operations for first column tile (0-15)
		wmma::mma_sync(c0, a_frag0, b_frag, c0);
		wmma::mma_sync(c1, a_frag1, b_frag, c1);
		wmma::mma_sync(c2, a_frag2, b_frag, c2);
		wmma::mma_sync(c3, a_frag3, b_frag, c3);

		// Perform WMMA operations for second column tile (16-31) if within bounds
		if (tileCol + 16 < N) {
			wmma::mma_sync(c4, a_frag0, b_frag1, c4);
			wmma::mma_sync(c5, a_frag1, b_frag1, c5);
			wmma::mma_sync(c6, a_frag2, b_frag1, c6);
			wmma::mma_sync(c7, a_frag3, b_frag1, c7);
		}
	}

	// Store results for all 8 accumulator fragments
	const unsigned int outCol = tileCol;
	if (outCol < N) {
		// Store first column tile (0-15)
		wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 0 * 16u) * N + outCol, c0, N, wmma::mem_row_major);
		wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 1 * 16u) * N + outCol, c1, N, wmma::mem_row_major);
		wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 2 * 16u) * N + outCol, c2, N, wmma::mem_row_major);
		wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 3 * 16u) * N + outCol, c3, N, wmma::mem_row_major);

		// Store second column tile (16-31) if within bounds
		if (tileCol + 16 < N) {
			wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 0 * 16u) * N + outCol + 16, c4, N, wmma::mem_row_major);
			wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 1 * 16u) * N + outCol + 16, c5, N, wmma::mem_row_major);
			wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 2 * 16u) * N + outCol + 16, c6, N, wmma::mem_row_major);
			wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 3 * 16u) * N + outCol + 16, c7, N, wmma::mem_row_major);
		}
	}
}

// Kernel optimized for 32x16 BCSR blocks - handles 32 rows and 32 columns per thread block
// Each block covers 32 rows and 32 columns, using 2 WMMA operations vertically and 2 horizontally
__global__ void sparseMatrixMulTensor32x16(const int *hdr, const int *idx,
											const half *data, const half *B,
											float *C, const unsigned int M, const unsigned int N) {
	// Each block covers 32 rows and 32 columns (for B reuse)
	const unsigned int blockRow32 = blockIdx.y * 32u;
	const unsigned int tileCol = blockIdx.x * 32u;

	if (blockRow32 >= M || tileCol >= N) return;

	// For 32x16 BCSR blocks, we need 2 WMMA operations vertically and 2 horizontally for B reuse
	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag0, a_frag1;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag, b_frag1;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1, c2, c3;

	wmma::fill_fragment(c0, 0.0f);  // rows 0-15, cols 0-15
	wmma::fill_fragment(c1, 0.0f);  // rows 16-31, cols 0-15
	wmma::fill_fragment(c2, 0.0f);  // rows 0-15, cols 16-31
	wmma::fill_fragment(c3, 0.0f);  // rows 16-31, cols 16-31

	// Block row index for the 32-row block (each BCSR entry represents one 32x16 block)
	const int blockRowIdx = static_cast<int>(blockRow32 / 32u);

	// Iterate over non-zero 32x16 BCSR blocks in this block-row
	for (int k = hdr[blockRowIdx]; k < hdr[blockRowIdx + 1]; ++k) {
		// Load B fragments (same for all A sub-blocks)
		const size_t b_off = static_cast<size_t>(idx[k]) * 16u * static_cast<size_t>(N) + static_cast<size_t>(tileCol);
		wmma::load_matrix_sync(b_frag, B + b_off, N);
		if (tileCol + 16 < N) {
			wmma::load_matrix_sync(b_frag1, B + b_off + 16, N);
		}

		// The 32x16 BCSR block is stored as a contiguous 32x16 matrix in data
		// We need to load 2 different 16x16 sub-blocks from this 32x16 block
		const half *block_data = data + static_cast<size_t>(k) * 32u * 16u;

		// Load A fragments from different vertical positions within the 32x16 block
		wmma::load_matrix_sync(a_frag0, block_data + 0 * 16u * 16u, 16);   // rows 0-15
		wmma::load_matrix_sync(a_frag1, block_data + 1 * 16u * 16u, 16);   // rows 16-31

		// Perform WMMA operations for first column tile (0-15)
		wmma::mma_sync(c0, a_frag0, b_frag, c0);
		wmma::mma_sync(c1, a_frag1, b_frag, c1);

		// Perform WMMA operations for second column tile (16-31) if within bounds
		if (tileCol + 16 < N) {
			wmma::mma_sync(c2, a_frag0, b_frag1, c2);
			wmma::mma_sync(c3, a_frag1, b_frag1, c3);
		}
	}

	// Store results for all 4 accumulator fragments
	const unsigned int outCol = tileCol;
	if (outCol < N) {
		// Store first column tile (0-15)
		wmma::store_matrix_sync(C + static_cast<size_t>(blockRow32 + 0 * 16u) * N + outCol, c0, N, wmma::mem_row_major);
		wmma::store_matrix_sync(C + static_cast<size_t>(blockRow32 + 1 * 16u) * N + outCol, c1, N, wmma::mem_row_major);

		// Store second column tile (16-31) if within bounds
		if (tileCol + 16 < N) {
			wmma::store_matrix_sync(C + static_cast<size_t>(blockRow32 + 0 * 16u) * N + outCol + 16, c2, N, wmma::mem_row_major);
			wmma::store_matrix_sync(C + static_cast<size_t>(blockRow32 + 1 * 16u) * N + outCol + 16, c3, N, wmma::mem_row_major);
		}
	}
}

			// Variant: same as `sparseMatrixMulTensor32x16` but stages the 32x16 A block into
			// shared memory first so both fragments (rows 0-15 and 16-31) are loaded from
			// fast on-chip memory via the WMMA API. Launch with shared memory size at
			// least `32*16*sizeof(half)` per block.
			__global__ void sparseMatrixMulTensor32x16_shared(const int *hdr, const int *idx,
															  const half *data, const half *B,
															  float *C, const unsigned int M, const unsigned int N) {
				// Each block covers 32 rows and 32 columns (for B reuse)
				const unsigned int blockRow32 = blockIdx.y * 32u;
				const unsigned int tileCol = blockIdx.x * 32u;

				if (blockRow32 >= M || tileCol >= N) return;

				// Shared memory buffer for the 32x16 A block (row-major): 32*16 half elements
				extern __shared__ half sA[]; // size (elements) = 32*16 == 512

				// WMMA fragments
				wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag0, a_frag1;
				wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag, b_frag1;
				wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1, c2, c3;

				wmma::fill_fragment(c0, 0.0f);
				wmma::fill_fragment(c1, 0.0f);
				wmma::fill_fragment(c2, 0.0f);
				wmma::fill_fragment(c3, 0.0f);

				const int blockRowIdx = static_cast<int>(blockRow32 / 32u);

				// Iterate over non-zero 32x16 BCSR blocks in this block-row
				for (int k = hdr[blockRowIdx]; k < hdr[blockRowIdx + 1]; ++k) {
					// Pointer to the contiguous 32x16 block in global memory
					const half *block_data = data + static_cast<size_t>(k) * 32u * 16u;

					// Cooperative load of the whole 32x16 block into shared memory
					const unsigned int elems = 32u * 16u; // 512
					for (unsigned int i = threadIdx.x; i < elems; i += blockDim.x) {
						sA[i] = block_data[i];
					}
					__syncthreads();

					// Load B fragments (same for both A sub-blocks)
					const size_t b_off = static_cast<size_t>(idx[k]) * 16u * static_cast<size_t>(N) + static_cast<size_t>(tileCol);
					wmma::load_matrix_sync(b_frag, B + b_off, N);
					if (tileCol + 16 < N) {
						wmma::load_matrix_sync(b_frag1, B + b_off + 16, N);
					}

					// Load A fragments from shared memory: first 16x16 (rows 0-15), then next 16x16 (rows 16-31)
					// Each 16x16 block occupies 256 half elements; layout matches original contiguous layout.
					const half *sA0 = sA + 0 * (16u * 16u);
					const half *sA1 = sA + 1 * (16u * 16u);

					wmma::load_matrix_sync(a_frag0, sA0, 16);
					wmma::load_matrix_sync(a_frag1, sA1, 16);

					// Perform WMMA operations for first column tile (0-15)
					wmma::mma_sync(c0, a_frag0, b_frag, c0);
					wmma::mma_sync(c1, a_frag1, b_frag, c1);

					// Perform WMMA operations for second column tile (16-31) if within bounds
					if (tileCol + 16 < N) {
						wmma::mma_sync(c2, a_frag0, b_frag1, c2);
						wmma::mma_sync(c3, a_frag1, b_frag1, c3);
					}

					__syncthreads();
				}

				// Store results for all 4 accumulator fragments
				const unsigned int outCol = tileCol;
				if (outCol < N) {
					// Store first column tile (0-15)
					wmma::store_matrix_sync(C + static_cast<size_t>(blockRow32 + 0 * 16u) * N + outCol, c0, N, wmma::mem_row_major);
					wmma::store_matrix_sync(C + static_cast<size_t>(blockRow32 + 1 * 16u) * N + outCol, c1, N, wmma::mem_row_major);

					// Store second column tile (16-31) if within bounds
					if (tileCol + 16 < N) {
						wmma::store_matrix_sync(C + static_cast<size_t>(blockRow32 + 0 * 16u) * N + outCol + 16, c2, N, wmma::mem_row_major);
						wmma::store_matrix_sync(C + static_cast<size_t>(blockRow32 + 1 * 16u) * N + outCol + 16, c3, N, wmma::mem_row_major);
					}
				}
			}

// Kernel: sparseMatrixMulTensor32x16_shared_vectorized
// Stages B (16x32 tile) into shared memory using vectorized loads, and uses
// existing A block layout (32x16) from global `data`. Shared memory layout:
//  - sA: 32x16 elements (for A staging, optional)
//  - sB: 16x32 elements (for B tile, vectorized load)
// Launch with blockSize {64,1,1} and sharedBytes = (32*16 + 16*32) * sizeof(half)
__global__ void sparseMatrixMulTensor32x16_shared_vectorized(const int *hdr, const int *idx,
															const half *data, const half *B,
															float *C, const unsigned int M, const unsigned int N) {
	const unsigned int blockRow32 = blockIdx.y * 32u;
	const unsigned int tileCol = blockIdx.x * 32u;

	if (blockRow32 >= M || tileCol >= N) return;

	// dynamic shared buffer: first sA (32*16), then sB (16*32)
	extern __shared__ half s[];
	half *sA = s;                        // 32*16 elements
	half *sB_raw = s + 32 * 16;         // points to 16*32 elements
	half (*sB)[32] = reinterpret_cast<half (*)[32]>(sB_raw);

	// WMMA fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag0, a_frag1;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag, b_frag1;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1, c2, c3;

	wmma::fill_fragment(c0, 0.0f);
	wmma::fill_fragment(c1, 0.0f);
	wmma::fill_fragment(c2, 0.0f);
	wmma::fill_fragment(c3, 0.0f);

	const int blockRowIdx = static_cast<int>(blockRow32 / 32u);

	// Load A block into shared memory using the vectorized helper

	// Iterate over non-zero 32x16 BCSR blocks in this block-row
	for (int k = hdr[blockRowIdx]; k < hdr[blockRowIdx + 1]; ++k) {
		// Pointer to the contiguous 32x16 block in global memory (A)
		const half *block_data = data + static_cast<size_t>(k) * 32u * 16u;

		// Vectorized load A block into shared memory sA
		load_A_block_32x16_vectorized<half, int4>(block_data, sA, threadIdx.x, blockDim.x);
		__syncthreads();

		// Vectorized load B tile (16 rows x 32 cols) into shared memory sB
		// Use template with BLOCK_TILE_SIZE_X=32, BLOCK_TILE_SIZE_K=16, NUM_THREADS=64
		load_B_to_shared_vectorized<half, 32, 16, 64, 0, int4>(B, static_cast<size_t>(N),
															  reinterpret_cast<half (*)[32]>(sB),
															  static_cast<size_t>(idx[k]), static_cast<size_t>(threadIdx.x),
															  static_cast<size_t>(N), static_cast<size_t>(-1));
		__syncthreads();

		// Load B fragments from shared memory. Leading dimension is 32.
		wmma::load_matrix_sync(b_frag, &sB[0][0], 32);
		if (tileCol + 16 < N) {
			wmma::load_matrix_sync(b_frag1, &sB[0][16], 32);
		}

		// Load A fragments from shared memory: first 16x16 (rows 0-15), then next 16x16 (rows 16-31)
		const half *sA0 = sA + 0 * (16u * 16u);
		const half *sA1 = sA + 1 * (16u * 16u);

		wmma::load_matrix_sync(a_frag0, sA0, 16);
		wmma::load_matrix_sync(a_frag1, sA1, 16);

		// Perform WMMA operations for first column tile (0-15)
		wmma::mma_sync(c0, a_frag0, b_frag, c0);
		wmma::mma_sync(c1, a_frag1, b_frag, c1);

		// Perform WMMA operations for second column tile (16-31) if within bounds
		if (tileCol + 16 < N) {
			wmma::mma_sync(c2, a_frag0, b_frag1, c2);
			wmma::mma_sync(c3, a_frag1, b_frag1, c3);
		}

		__syncthreads();
	}

	// Store results for all 4 accumulator fragments
	const unsigned int outCol = tileCol;
	if (outCol < N) {
		wmma::store_matrix_sync(C + static_cast<size_t>(blockRow32 + 0 * 16u) * N + outCol, c0, N, wmma::mem_row_major);
		wmma::store_matrix_sync(C + static_cast<size_t>(blockRow32 + 1 * 16u) * N + outCol, c1, N, wmma::mem_row_major);
		if (tileCol + 16 < N) {
			wmma::store_matrix_sync(C + static_cast<size_t>(blockRow32 + 0 * 16u) * N + outCol + 16, c2, N, wmma::mem_row_major);
			wmma::store_matrix_sync(C + static_cast<size_t>(blockRow32 + 1 * 16u) * N + outCol + 16, c3, N, wmma::mem_row_major);
		}
	}
}

// Kernel variant: sparseMatrixMulTensor64x16_v2 (v2-style for 64x16 blocks with 64 threads)
// Each block has 2 warps (64 threads), each warp handles 16 columns
// Grid covers 32 columns per block (like v2 pattern): warp 0 handles cols 0-15, warp 1 handles cols 16-31
__global__ void sparseMatrixMulTensor64x16_v2(const int *hdr, const int *idx,
											  const half *data, const half *B,
											  float *C, const unsigned int M, const unsigned int N) {
	// Each block covers 64 rows and 32 columns (two 16-column tiles per block)
	const unsigned int blockRow64 = blockIdx.y * 64u;
	const unsigned int tileColBase = blockIdx.x * 32u; // each block covers 32 columns

	if (blockRow64 >= M || tileColBase >= N) return;

	// Identify the warp within the block: 0 or 1 (since blockDim.x == 64)
	const unsigned int warpId = threadIdx.x / 32;

	// Each warp creates fragments for the 64x16 block computation
	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag0, a_frag1, a_frag2, a_frag3;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1, c2, c3;

	wmma::fill_fragment(c0, 0.0f);  // rows 0-15
	wmma::fill_fragment(c1, 0.0f);  // rows 16-31
	wmma::fill_fragment(c2, 0.0f);  // rows 32-47
	wmma::fill_fragment(c3, 0.0f);  // rows 48-63

	// Block row index for the 64-row block (each BCSR entry represents one 64x16 block)
	const int blockRowIdx = static_cast<int>(blockRow64 / 64u);

	// Calculate which 16-column tile this warp handles
	const unsigned int tileCol = tileColBase + warpId * 16u;
	if (tileCol >= N) return; // Guard against out-of-bounds columns

	// Iterate over non-zero 64x16 BCSR blocks in this block-row
	for (int k = hdr[blockRowIdx]; k < hdr[blockRowIdx + 1]; ++k) {
		// Load B fragment for the 16 columns this warp handles
		const size_t b_off = static_cast<size_t>(idx[k]) * 16u * static_cast<size_t>(N) + static_cast<size_t>(tileCol);
		wmma::load_matrix_sync(b_frag, B + b_off, N);

		// The 64x16 BCSR block is stored as a contiguous 64x16 matrix in data
		// We need to load 4 different 16x16 sub-blocks from this 64x16 block
		const half *block_data = data + static_cast<size_t>(k) * 64u * 16u;

		// Load A fragments from different vertical positions within the 64x16 block
		wmma::load_matrix_sync(a_frag0, block_data + 0 * 16u * 16u, 16);   // rows 0-15
		wmma::load_matrix_sync(a_frag1, block_data + 1 * 16u * 16u, 16);   // rows 16-31
		wmma::load_matrix_sync(a_frag2, block_data + 2 * 16u * 16u, 16);   // rows 32-47
		wmma::load_matrix_sync(a_frag3, block_data + 3 * 16u * 16u, 16);   // rows 48-63

		// Perform WMMA operations for the 16-column tile this warp handles
		wmma::mma_sync(c0, a_frag0, b_frag, c0);
		wmma::mma_sync(c1, a_frag1, b_frag, c1);
		wmma::mma_sync(c2, a_frag2, b_frag, c2);
		wmma::mma_sync(c3, a_frag3, b_frag, c3);
	}

	// Store results for all 4 accumulator fragments for this warp's tile
	wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 0 * 16u) * N + tileCol, c0, N, wmma::mem_row_major);
	wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 1 * 16u) * N + tileCol, c1, N, wmma::mem_row_major);
	wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 2 * 16u) * N + tileCol, c2, N, wmma::mem_row_major);
	wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 3 * 16u) * N + tileCol, c3, N, wmma::mem_row_major);
}

// Kernel variant: sparseMatrixMulTensor32x16_v2 (v2-style for 32x16 blocks with 64 threads)
// Each block has 2 warps (64 threads), each warp handles 16 columns
// Grid covers 32 columns per block (like v2 pattern): warp 0 handles cols 0-15, warp 1 handles cols 16-31
__global__ void sparseMatrixMulTensor32x16_v2(const int *hdr, const int *idx,
											  const half *data, const half *B,
											  float *C, const unsigned int M, const unsigned int N) {
	// Each block covers 32 rows and 32 columns (two 16-column tiles per block)
	const unsigned int blockRow32 = blockIdx.y * 32u;
	const unsigned int tileColBase = blockIdx.x * 32u; // each block covers 32 columns

	if (blockRow32 >= M || tileColBase >= N) return;

	// Identify the warp within the block: 0 or 1 (since blockDim.x == 64)
	const unsigned int warpId = threadIdx.x / 32;

	// Each warp creates fragments for the 32x16 block computation
	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag0, a_frag1;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;

	wmma::fill_fragment(c0, 0.0f);  // rows 0-15
	wmma::fill_fragment(c1, 0.0f);  // rows 16-31

	// Block row index for the 32-row block (each BCSR entry represents one 32x16 block)
	const int blockRowIdx = static_cast<int>(blockRow32 / 32u);

	// Calculate which 16-column tile this warp handles
	const unsigned int tileCol = tileColBase + warpId * 16u;
	if (tileCol >= N) return; // Guard against out-of-bounds columns

	// Iterate over non-zero 32x16 BCSR blocks in this block-row
	for (int k = hdr[blockRowIdx]; k < hdr[blockRowIdx + 1]; ++k) {
		// Load B fragment for the 16 columns this warp handles
		const size_t b_off = static_cast<size_t>(idx[k]) * 16u * static_cast<size_t>(N) + static_cast<size_t>(tileCol);
		wmma::load_matrix_sync(b_frag, B + b_off, N);

		// The 32x16 BCSR block is stored as a contiguous 32x16 matrix in data
		// We need to load 2 different 16x16 sub-blocks from this 32x16 block
		const half *block_data = data + static_cast<size_t>(k) * 32u * 16u;

		// Load A fragments from different vertical positions within the 32x16 block
		wmma::load_matrix_sync(a_frag0, block_data + 0 * 16u * 16u, 16);   // rows 0-15
		wmma::load_matrix_sync(a_frag1, block_data + 1 * 16u * 16u, 16);   // rows 16-31

		// Perform WMMA operations for the 16-column tile this warp handles
		wmma::mma_sync(c0, a_frag0, b_frag, c0);
		wmma::mma_sync(c1, a_frag1, b_frag, c1);
	}

	// Store results for all 2 accumulator fragments for this warp's tile
	wmma::store_matrix_sync(C + static_cast<size_t>(blockRow32 + 0 * 16u) * N + tileCol, c0, N, wmma::mem_row_major);
	wmma::store_matrix_sync(C + static_cast<size_t>(blockRow32 + 1 * 16u) * N + tileCol, c1, N, wmma::mem_row_major);
}

// Adaptive BCSR kernel for 16x16 blocks (v2-style) that distributes column work across multiple thread blocks for "fat" matrices
// Uses 2 warps per block (64 threads) to process two adjacent 16-column tiles horizontally
// colBlocksPerRow: number of thread blocks assigned to each BCSR row (based on aspect ratio)
__global__ void sparseMatrixMulTensorAdaptive(const int *hdr, const int *idx,
                                              const half *data, const half *B,
                                              float *C, const unsigned int M, const unsigned int N,
                                              const unsigned int colBlocksPerRow) {
	// Calculate which BCSR row this thread block is working on
	const unsigned int bcsrRow = blockIdx.y;
	const unsigned int colBlockId = blockIdx.x; // which column segment within the row
	const unsigned int totalColTiles32 = (N + 31) / 32; // total 32-column tiles in matrix (each block handles 32 cols)
	const unsigned int tilesPerBlock = totalColTiles32 / colBlocksPerRow; // 32-column tiles per thread block
	const unsigned int extraTiles = totalColTiles32 % colBlocksPerRow; // remainder tiles
	
	// Calculate column range for this thread block (in terms of 32-column tiles)
	unsigned int startTile, endTile;
	if (colBlockId < extraTiles) {
		// First 'extraTiles' blocks get one extra tile
		startTile = colBlockId * (tilesPerBlock + 1);
		endTile = (colBlockId + 1) * (tilesPerBlock + 1);
	} else {
		// Remaining blocks get standard tiles
		startTile = extraTiles * (tilesPerBlock + 1) + (colBlockId - extraTiles) * tilesPerBlock;
		endTile = startTile + tilesPerBlock;
	}
	
	if (bcsrRow >= M / 16 || colBlockId >= colBlocksPerRow) return;
	
	const unsigned int warpRow = bcsrRow * 16;
	const unsigned int warpId = threadIdx.x / 32; // 0 or 1 (since blockDim.x == 64)
	
	// Process each 32-column tile within this thread block's assigned range
	for (unsigned int tileIdx = startTile; tileIdx < endTile; ++tileIdx) {
		const unsigned int tileColBase = tileIdx * 32;
		if (tileColBase >= N) break;
		
		// Each warp handles one 16-column subtile within the 32-column tile
		const unsigned int tileCol = tileColBase + warpId * 16;
		if (tileCol >= N) continue;
		
		wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
		
		wmma::fill_fragment(c_frag, 0.0f);
		
		// Process all non-zero blocks in this BCSR row
		for (int k = hdr[bcsrRow]; k < hdr[bcsrRow + 1]; k++) {
			wmma::load_matrix_sync(a_frag, data + static_cast<size_t>(k) * 16 * 16, 16);
			wmma::load_matrix_sync(b_frag, B + static_cast<size_t>(idx[k]) * 16 * N + tileCol, N);
			wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
		}
		
		wmma::store_matrix_sync(C + static_cast<size_t>(warpRow) * N + tileCol, c_frag, N, wmma::mem_row_major);
	}
}

// Adaptive BCSR kernel for 32x16 blocks (v2-style) 
// Uses 2 warps per block (64 threads) to process two adjacent 16-column tiles horizontally
__global__ void sparseMatrixMulTensorAdaptive32x16(const int *hdr, const int *idx,
                                                   const half *data, const half *B,
                                                   float *C, const unsigned int M, const unsigned int N,
                                                   const unsigned int colBlocksPerRow) {
	// Calculate which BCSR row this thread block is working on (32-row blocks)
	const unsigned int bcsrRow = blockIdx.y;
	const unsigned int colBlockId = blockIdx.x;
	const unsigned int totalColTiles32 = (N + 31) / 32; // total 32-column tiles in matrix
	const unsigned int tilesPerBlock = totalColTiles32 / colBlocksPerRow;
	const unsigned int extraTiles = totalColTiles32 % colBlocksPerRow;
	
	// Calculate column range for this thread block (in terms of 32-column tiles)
	unsigned int startTile, endTile;
	if (colBlockId < extraTiles) {
		startTile = colBlockId * (tilesPerBlock + 1);
		endTile = (colBlockId + 1) * (tilesPerBlock + 1);
	} else {
		startTile = extraTiles * (tilesPerBlock + 1) + (colBlockId - extraTiles) * tilesPerBlock;
		endTile = startTile + tilesPerBlock;
	}
	
	if (bcsrRow >= M / 32 || colBlockId >= colBlocksPerRow) return;
	
	const unsigned int blockRow32 = bcsrRow * 32;
	const unsigned int warpId = threadIdx.x / 32; // 0 or 1
	
	// Process each 32-column tile within this thread block's assigned range
	for (unsigned int tileIdx = startTile; tileIdx < endTile; ++tileIdx) {
		const unsigned int tileColBase = tileIdx * 32;
		if (tileColBase >= N) break;
		
		// Each warp handles one 16-column subtile within the 32-column tile
		const unsigned int tileCol = tileColBase + warpId * 16;
		if (tileCol >= N) continue;
		
		// For 32x16 BCSR blocks, we need 2 WMMA operations vertically per warp
		wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag0, a_frag1;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
		
		wmma::fill_fragment(c0, 0.0f);  // rows 0-15
		wmma::fill_fragment(c1, 0.0f);  // rows 16-31
		
		// Process all non-zero 32x16 blocks in this BCSR row
		for (int k = hdr[bcsrRow]; k < hdr[bcsrRow + 1]; ++k) {
			// Load B fragment for this warp's column range
			wmma::load_matrix_sync(b_frag, B + static_cast<size_t>(idx[k]) * 16u * N + tileCol, N);
			
			// Load A fragments from different vertical positions within the 32x16 block
			const half *block_data = data + static_cast<size_t>(k) * 32u * 16u;
			wmma::load_matrix_sync(a_frag0, block_data + 0 * 16u * 16u, 16);   // rows 0-15
			wmma::load_matrix_sync(a_frag1, block_data + 1 * 16u * 16u, 16);   // rows 16-31
			
			// Perform WMMA operations
			wmma::mma_sync(c0, a_frag0, b_frag, c0);
			wmma::mma_sync(c1, a_frag1, b_frag, c1);
		}
		
		// Store results for this warp's column range
		wmma::store_matrix_sync(C + static_cast<size_t>(blockRow32 + 0 * 16u) * N + tileCol, c0, N, wmma::mem_row_major);
		wmma::store_matrix_sync(C + static_cast<size_t>(blockRow32 + 1 * 16u) * N + tileCol, c1, N, wmma::mem_row_major);
	}
}

// Adaptive BCSR kernel for 64x16 blocks (v2-style)
// Uses 2 warps per block (64 threads) to process two adjacent 16-column tiles horizontally
__global__ void sparseMatrixMulTensorAdaptive64x16(const int *hdr, const int *idx,
                                                   const half *data, const half *B,
                                                   float *C, const unsigned int M, const unsigned int N,
                                                   const unsigned int colBlocksPerRow) {
	// Calculate which BCSR row this thread block is working on (64-row blocks)
	const unsigned int bcsrRow = blockIdx.y;
	const unsigned int colBlockId = blockIdx.x;
	const unsigned int totalColTiles32 = (N + 31) / 32; // total 32-column tiles in matrix
	const unsigned int tilesPerBlock = totalColTiles32 / colBlocksPerRow;
	const unsigned int extraTiles = totalColTiles32 % colBlocksPerRow;
	
	// Calculate column range for this thread block (in terms of 32-column tiles)
	unsigned int startTile, endTile;
	if (colBlockId < extraTiles) {
		startTile = colBlockId * (tilesPerBlock + 1);
		endTile = (colBlockId + 1) * (tilesPerBlock + 1);
	} else {
		startTile = extraTiles * (tilesPerBlock + 1) + (colBlockId - extraTiles) * tilesPerBlock;
		endTile = startTile + tilesPerBlock;
	}
	
	if (bcsrRow >= M / 64 || colBlockId >= colBlocksPerRow) return;
	
	const unsigned int blockRow64 = bcsrRow * 64;
	const unsigned int warpId = threadIdx.x / 32; // 0 or 1
	
	// Process each 32-column tile within this thread block's assigned range
	for (unsigned int tileIdx = startTile; tileIdx < endTile; ++tileIdx) {
		const unsigned int tileColBase = tileIdx * 32;
		if (tileColBase >= N) break;
		
		// Each warp handles one 16-column subtile within the 32-column tile
		const unsigned int tileCol = tileColBase + warpId * 16;
		if (tileCol >= N) continue;
		
		// For 64x16 BCSR blocks, we need 4 WMMA operations vertically per warp
		wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag0, a_frag1, a_frag2, a_frag3;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1, c2, c3;
		
		wmma::fill_fragment(c0, 0.0f);  // rows 0-15
		wmma::fill_fragment(c1, 0.0f);  // rows 16-31
		wmma::fill_fragment(c2, 0.0f);  // rows 32-47
		wmma::fill_fragment(c3, 0.0f);  // rows 48-63
		
		// Process all non-zero 64x16 blocks in this BCSR row
		for (int k = hdr[bcsrRow]; k < hdr[bcsrRow + 1]; ++k) {
			// Load B fragment for this warp's column range
			wmma::load_matrix_sync(b_frag, B + static_cast<size_t>(idx[k]) * 16u * N + tileCol, N);
			
			// Load A fragments from different vertical positions within the 64x16 block
			const half *block_data = data + static_cast<size_t>(k) * 64u * 16u;
			wmma::load_matrix_sync(a_frag0, block_data + 0 * 16u * 16u, 16);   // rows 0-15
			wmma::load_matrix_sync(a_frag1, block_data + 1 * 16u * 16u, 16);   // rows 16-31
			wmma::load_matrix_sync(a_frag2, block_data + 2 * 16u * 16u, 16);   // rows 32-47
			wmma::load_matrix_sync(a_frag3, block_data + 3 * 16u * 16u, 16);   // rows 48-63
			
			// Perform WMMA operations
			wmma::mma_sync(c0, a_frag0, b_frag, c0);
			wmma::mma_sync(c1, a_frag1, b_frag, c1);
			wmma::mma_sync(c2, a_frag2, b_frag, c2);
			wmma::mma_sync(c3, a_frag3, b_frag, c3);
		}
		
		// Store results for this warp's column range
		wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 0 * 16u) * N + tileCol, c0, N, wmma::mem_row_major);
		wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 1 * 16u) * N + tileCol, c1, N, wmma::mem_row_major);
		wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 2 * 16u) * N + tileCol, c2, N, wmma::mem_row_major);
		wmma::store_matrix_sync(C + static_cast<size_t>(blockRow64 + 3 * 16u) * N + tileCol, c3, N, wmma::mem_row_major);
	}
}