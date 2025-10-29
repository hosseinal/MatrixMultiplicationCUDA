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

#ifndef CEIL_DIV
#define CEIL_DIV(_a, _b) (((_a) / (_b)) + (((_a) % (_b)) > 0 ? 1 : 0))
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
                                      float *C, const unsigned int n) {
    const unsigned int warpRow = blockIdx.y * 16;
    const unsigned int warpCol = blockIdx.x * 16;

    if (warpRow >= n || warpCol >= n) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = hdr[warpRow / 16]; k < hdr[warpRow / 16 + 1]; k++) {
        wmma::load_matrix_sync(a_frag, data + k * 16 * 16, 16);
        wmma::load_matrix_sync(b_frag, B + idx[k] * 16 * n + warpCol, n);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warpRow * n + warpCol, c_frag, n,
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
// Local constant to match main.cu's thread configuration
constexpr unsigned int N_THREADS = 32;

static const std::vector<std::string> patterns = {
	"random",
	"checkerboard",
	"diagonal",
	"blockdiagonal",
	"blockrandom"
};

// Explicit (M, K) pairs — one pair per line as requested
static const std::vector<std::pair<int,int>> mk_pairs = {
	{1024, 256},
	{512, 2048},
	{512, 4608},
	{256, 1024},
	{512, 256},
	{2048, 1024},
	{512, 512}
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
	state.get_summary("nv/cold/sm_clock_rate/mean").remove_value("hide");
	state.get_summary("nv/cold/sm_clock_rate/scaling/percent").remove_value("hide");

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

	dim3 gridSize{static_cast<unsigned int>((N + 15) / 16), static_cast<unsigned int>((M + 15) / 16), 1};
	dim3 blockSize{32, 1, 1};

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		// clear output buffer for this iteration on the launch stream
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		// start timer
		timer.start();
		sparseMatrixMulTensor<<<gridSize, blockSize, 0, launch.get_stream()>>>(buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(N));
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
	state.exec([&](nvbench::launch &launch){
		// clear output buffer for this iteration
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, buf->gpuB_half, CUDA_R_16F, N, buf->gpuA_half, CUDA_R_16F, K, &beta, buf->gpuC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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
}

// Register benches and axes
NVBENCH_BENCH(bench_denseMatrixMul).set_name("denseMatrixMul").add_int64_axis("N", {16, 32, 64, 128}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_denseMatrixMulTensor).set_name("denseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_sparseMatrixMult1).set_name("sparseMatrixMult1").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_sparseMatrixMulTensor).set_name("sparseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_cuBLAS).set_name("cuBLAS_GEMM").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});


// size 512 * 2048
NVBENCH_BENCH(bench_denseMatrixMul).set_name("denseMatrixMul").add_int64_axis("N", {16, 32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {2048}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_denseMatrixMulTensor).set_name("denseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {2048}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_sparseMatrixMult1).set_name("sparseMatrixMult1").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {2048}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_sparseMatrixMulTensor).set_name("sparseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {2048}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_cuBLAS).set_name("cuBLAS_GEMM").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {2048}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {2048}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});

// size 512 * 4608
NVBENCH_BENCH(bench_denseMatrixMul).set_name("denseMatrixMul").add_int64_axis("N", {16, 32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {4608}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_denseMatrixMulTensor).set_name("denseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {4608}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_sparseMatrixMult1).set_name("sparseMatrixMult1").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {4608}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_sparseMatrixMulTensor).set_name("sparseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {4608}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_cuBLAS).set_name("cuBLAS_GEMM").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {4608}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {4608}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});

// size 256 * 1024
NVBENCH_BENCH(bench_denseMatrixMul).set_name("denseMatrixMul").add_int64_axis("N", {16, 32, 64, 128}).add_int64_axis("M", {256}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_denseMatrixMulTensor).set_name("denseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {256}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_sparseMatrixMult1).set_name("sparseMatrixMult1").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {256}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_sparseMatrixMulTensor).set_name("sparseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {256}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_cuBLAS).set_name("cuBLAS_GEMM").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {256}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {256}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});

// size 512 * 256
NVBENCH_BENCH(bench_denseMatrixMul).set_name("denseMatrixMul").add_int64_axis("N", {16, 32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_denseMatrixMulTensor).set_name("denseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_sparseMatrixMult1).set_name("sparseMatrixMult1").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_sparseMatrixMulTensor).set_name("sparseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_cuBLAS).set_name("cuBLAS_GEMM").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});

// size 2048 * 1024
NVBENCH_BENCH(bench_denseMatrixMul).set_name("denseMatrixMul").add_int64_axis("N", {16, 32, 64, 128}).add_int64_axis("M", {2048}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_denseMatrixMulTensor).set_name("denseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {2048}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_sparseMatrixMult1).set_name("sparseMatrixMult1").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {2048}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_sparseMatrixMulTensor).set_name("sparseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {2048}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_cuBLAS).set_name("cuBLAS_GEMM").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {2048}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {2048}).add_int64_axis("K", {1024}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});

//size 512 * 512
NVBENCH_BENCH(bench_denseMatrixMul).set_name("denseMatrixMul").add_int64_axis("N", {16, 32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_denseMatrixMulTensor).set_name("denseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_sparseMatrixMult1).set_name("sparseMatrixMult1").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_sparseMatrixMulTensor).set_name("sparseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_cuBLAS).set_name("cuBLAS_GEMM").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {128, 256, 512}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
