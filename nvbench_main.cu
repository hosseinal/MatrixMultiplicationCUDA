// nvbench harness to run selected kernels from main.cu across patterns/sizes/sparsities
#include <nvbench/nvbench.cuh>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <thrust/host_vector.h>
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

// Include our CUDA kernels
#include "src/cuda_kernels.cuh"
#include <cuda/barrier>

#include "matrix_generator.h"
// Include implementation so templates are available in this TU (quick solution)
#include "matrix_generator.cpp"
// Matrix/CSR/BCSR types
#include "Matrix.cuh"
#include "formats/CSRMatrix.cuh"
#include "formats/BCSRMatrix.cuh"

// // CUTE GEMM wrapper prototypes
// #include "cute_test/gemm_cute_tensor.cuh"
// #include "cute_test/gemm_cute_simt.cuh"
// #include "cutlass/semaphore.h"



using namespace std;
using namespace nvcuda;

#ifndef CEIL_DIV
#define CEIL_DIV(_a, _b) (((_a) / (_b)) + (((_a) % (_b)) > 0 ? 1 : 0))
#endif

// Local constant to match main.cu's thread configuration
constexpr unsigned int N_THREADS = 32;

// Provide BLOCK_SIZE for this TU. Keep internal linkage to avoid ODR violations
// when building multiple translation units that may also define BLOCK_SIZE.
static const int BLOCK_SIZE = 16;

static const std::vector<std::string> patterns = {
	"random",
	"checkerboard",
	"diagonal",
	"blockdiagonal",
	"blockrandom",
	"pattern64by16",
	"pattern32by16"
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

// Allow callers to specify block sizes for BCSR construction. Defaults kept at 16x16
static std::unique_ptr<GenDeviceBuffers> prepare_buffers(int M, int K, int N, double sparsity, const std::string &pattern,
														 int blockSizeRow = 16, int blockSizeCol = 16) {
	auto out = std::make_unique<GenDeviceBuffers>();
	// Generate float matrices with generator
	// A is M x K, B is K x N (dense), C will be M x N
	// pass block size into the generator so patterns that depend on block size behave consistently
	auto genA = mg::generate_matrix<float>(M, K, sparsity, pattern, blockSizeRow, 123);
	auto genB = mg::generate_matrix<float>(K, N, 0.0, "random", blockSizeRow, 456);

	out->matrixA = new Matrix(M, K);
	out->matrixB = new Matrix(K, N);
	fill_Matrix_from_generated(*out->matrixA, genA);
	fill_Matrix_from_generated(*out->matrixB, genB);

	// Build sparse representations from matrixA
	out->csrA = new CSRMatrix(*out->matrixA);
	// construct BCSR with explicit block sizes
	out->bcsrA = new BCSRMatrix(*out->matrixA, blockSizeRow, blockSizeCol);

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

	auto buf = prepare_buffers(M, K, N, spars, pattern, 16, 16);


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
				std::fprintf(stderr, "[denseMatrixMul kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s\n", M, K, N, spars*100.0, pattern.c_str());
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

	auto buf = prepare_buffers(M, K, N, spars, pattern, 16, 16);
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
				std::fprintf(stderr, "[denseMatrixMulTensor kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s\n", M, K, N, spars*100.0, pattern.c_str());
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

	auto buf = prepare_buffers(M, K, N, spars, pattern, 16, 16);
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
				std::fprintf(stderr, "[sparseMatrixMult1 kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s\n", M, K, N, spars*100.0, pattern.c_str());
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

	auto buf = prepare_buffers(M, K, N, spars, pattern, 16, 16);
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
				std::fprintf(stderr, "[sparseMatrixMulTensor kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s block_size=16x16\n", M, K, N, spars*100.0, pattern.c_str());
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

	auto buf = prepare_buffers(M, K, N, spars, pattern, 16, 16);
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
				std::fprintf(stderr, "[sparseMatrixMulTensor_v2 kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s block_size=16x16\n", M, K, N, spars*100.0, pattern.c_str());
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

	auto buf = prepare_buffers(M, K, N, spars, pattern, 16, 16);

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
				std::fprintf(stderr, "[sparseMatrixMulTensor_v3 kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s block_size=16x16\n", M, K, N, spars*100.0, pattern.c_str());
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

// Function to print the first 64x16 blocks of matrix A for debugging
void print_matrix_A_blocks(Matrix* matrixA, int M, int K, int num_blocks_to_print = 4) {
    std::cout << "\n=== Printing first " << num_blocks_to_print << " blocks (64x16 each) of Matrix A ===" << std::endl;
    std::cout << "Matrix A dimensions: " << M << " x " << K << std::endl;
    
    // Calculate how many 64x16 blocks we can fit
    int blocks_in_M = M / 64;
    int blocks_in_K = K / 16;
    
    std::cout << "Total blocks in M direction: " << blocks_in_M << std::endl;
    std::cout << "Total blocks in K direction: " << blocks_in_K << std::endl;
    
    // Print the first few blocks
    int blocks_printed = 0;
    for (int block_m = 0; block_m < blocks_in_M && blocks_printed < num_blocks_to_print; block_m++) {
        for (int block_k = 0; block_k < blocks_in_K && blocks_printed < num_blocks_to_print; block_k++) {
            std::cout << "\n--- Block (" << block_m << ", " << block_k << ") ---" << std::endl;
            std::cout << "Rows " << (block_m * 64) << " to " << (block_m * 64 + 63) 
                      << ", Cols " << (block_k * 16) << " to " << (block_k * 16 + 15) << std::endl;
            
            // Print the 64x16 block
            for (int row = 0; row < 64; row++) {
                int global_row = block_m * 64 + row;
                if (global_row >= M) break;
                
                std::cout << "Row " << std::setw(3) << global_row << ": ";
                for (int col = 0; col < 16; col++) {
                    int global_col = block_k * 16 + col;
                    if (global_col >= K) break;
                    
                    // Convert half to float for printing
                    half val_half = matrixA->data[global_row * K + global_col];
                    float val = __half2float(val_half);
                    std::cout << std::setw(6) << std::fixed << std::setprecision(2) << val << " ";
                }
                std::cout << std::endl;
            }
            blocks_printed++;
        }
    }
    std::cout << "=== End of Matrix A blocks ===" << std::endl;
}

// Benchmark: sparseMatrixMulTensor64x16 (process 4x16-row panels -> 64 rows per block)
static void bench_sparseMatrixMulTensor64x16(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern, 64, 16);

	// Print debug info only for small test cases
	if (M <= 256 && K <= 256) {
		print_matrix_A_blocks(buf->matrixA, M, K, 2);
	}

	// grid.x corresponds to 32-column tiles (2x16), grid.y to 64-row tiles
	dim3 gridSize{static_cast<unsigned int>((N + 31) / 32), static_cast<unsigned int>((M + 63) / 64), 1};
	dim3 blockSize{32, 1, 1};

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		timer.start();
		sparseMatrixMulTensor64x16<<<gridSize, blockSize, 0, launch.get_stream()>>>(buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(M), static_cast<unsigned int>(N));
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
				std::fprintf(stderr, "[sparseMatrixMulTensor64x16 kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s block_size=64x16\n", M, K, N, spars*100.0, pattern.c_str());
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

// Benchmark: sparseMatrixMulTensor_v1_improved
static void bench_sparseMatrixMulTensor_v1_improved(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern, 16, 16);

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
				std::fprintf(stderr, "[sparseMatrixMulTensor_v1_improved kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s block_size=16x16\n", M, K, N, spars*100.0, pattern.c_str());
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

// Benchmark: sparseMatrixMulTensor32x16 (optimized for 32x16 blocks)
static void bench_sparseMatrixMulTensor32x16(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern, 32, 16);

	// grid.x corresponds to 32-column tiles, grid.y to 32-row tiles
	dim3 gridSize{static_cast<unsigned int>((N + 31) / 32), static_cast<unsigned int>((M + 31) / 32), 1};
	dim3 blockSize{32, 1, 1};

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		timer.start();
		sparseMatrixMulTensor32x16<<<gridSize, blockSize, 0, launch.get_stream()>>>(buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(M), static_cast<unsigned int>(N));
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
				std::fprintf(stderr, "[sparseMatrixMulTensor32x16 kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s block_size=32x16\n", M, K, N, spars*100.0, pattern.c_str());
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

// Benchmark: CUTE GEMM (calls cute/gemm_cute_tensor.cu's gemm)
static void bench_cute_gemm_tensor(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	// Generate float matrices (dense) using the same generator used elsewhere
	auto genA = mg::generate_matrix<float>(M, K, spars, pattern, 16, 123);
	auto genB = mg::generate_matrix<float>(K, N, 0.0, "random", 16, 456);

	// Allocate device buffers for floats
	float *dA = nullptr, *dB = nullptr, *dC = nullptr;
	size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
	size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
	size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);
	cudaMalloc(reinterpret_cast<void**>(&dA), bytesA);
	cudaMalloc(reinterpret_cast<void**>(&dB), bytesB);
	cudaMalloc(reinterpret_cast<void**>(&dC), bytesC);

	// Copy generated host floats to device by flattening the vectors
	thrust::host_vector<float> hA_flat(static_cast<size_t>(M) * K);
	thrust::host_vector<float> hB_flat(static_cast<size_t>(K) * N);
	for (int i = 0; i < M; ++i) for (int j = 0; j < K; ++j) hA_flat[static_cast<size_t>(i) * K + j] = genA[i][j];
	for (int i = 0; i < K; ++i) for (int j = 0; j < N; ++j) hB_flat[static_cast<size_t>(i) * N + j] = genB[i][j];
	cudaMemcpy(dA, hA_flat.data(), bytesA, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB_flat.data(), bytesB, cudaMemcpyHostToDevice);
	cudaMemset(dC, 0, bytesC);

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		// clear output buffer for this iteration on the launch stream
		cudaMemsetAsync(dC, 0, bytesC, launch.get_stream());
		timer.start();
		// call the C wrapper that invokes the templated cute gemm
		cute_gemm_tensor_invoke('N', 'T', M, N, K, 1.0f, dA, M, dB, N, 0.0f, dC, M, launch.get_stream());
		timer.stop();
	});

	// Copy result back and verify on CPU
	{
		std::vector<float> out_host(static_cast<size_t>(M) * N);
		cudaMemcpy(out_host.data(), dC, out_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
		// Build CPU reference from generated matrices
		std::vector<float> ref(static_cast<size_t>(M) * N, 0.0f);
		for (int i = 0; i < M; ++i) {
			for (int k = 0; k < K; ++k) {
				float a = genA[i][k];
				for (int j = 0; j < N; ++j) {
					ref[static_cast<size_t>(i) * N + j] += a * genB[k][j];
				}
			}
		}
		const float eps = 1e-2f;
		for (size_t i = 0; i < out_host.size(); ++i) {
			if (std::fabs(out_host[i] - ref[i]) > eps) {
				std::fprintf(stderr, "[CUTE GEMM] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::abort();
			}
		}
	}

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	report_summary(state);
}

// Benchmark: CUTE GEMM SIMT (calls cute/gemm_cute_simt.cu's gemm)
static void bench_cute_gemm_simt(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	// Generate float matrices (dense) using the same generator used elsewhere
	auto genA = mg::generate_matrix<float>(M, K, spars, pattern, 16, 123);
	auto genB = mg::generate_matrix<float>(K, N, 0.0, "random", 16, 456);

	// Allocate device buffers for floats
	float *dA = nullptr, *dB = nullptr, *dC = nullptr;
	size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
	size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
	size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);
	cudaMalloc(reinterpret_cast<void**>(&dA), bytesA);
	cudaMalloc(reinterpret_cast<void**>(&dB), bytesB);
	cudaMalloc(reinterpret_cast<void**>(&dC), bytesC);

	// Copy generated host floats to device by flattening the vectors
	thrust::host_vector<float> hA_flat(static_cast<size_t>(M) * K);
	thrust::host_vector<float> hB_flat(static_cast<size_t>(K) * N);
	for (int i = 0; i < M; ++i) for (int j = 0; j < K; ++j) hA_flat[static_cast<size_t>(i) * K + j] = genA[i][j];
	for (int i = 0; i < K; ++i) for (int j = 0; j < N; ++j) hB_flat[static_cast<size_t>(i) * N + j] = genB[i][j];
	cudaMemcpy(dA, hA_flat.data(), bytesA, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB_flat.data(), bytesB, cudaMemcpyHostToDevice);
	cudaMemset(dC, 0, bytesC);

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		// clear output buffer for this iteration on the launch stream
		cudaMemsetAsync(dC, 0, bytesC, launch.get_stream());
		timer.start();
		// call the C wrapper that invokes the templated simt cute gemm
		cute_gemm_simt_invoke('N', 'T', M, N, K, 1.0f, dA, M, dB, N, 0.0f, dC, M, launch.get_stream());
		timer.stop();
	});

	// Copy result back and verify on CPU
	{
		std::vector<float> out_host(static_cast<size_t>(M) * N);
		cudaMemcpy(out_host.data(), dC, out_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
		// Build CPU reference from generated matrices
		std::vector<float> ref(static_cast<size_t>(M) * N, 0.0f);
		for (int i = 0; i < M; ++i) {
			for (int k = 0; k < K; ++k) {
				float a = genA[i][k];
				for (int j = 0; j < N; ++j) {
					ref[static_cast<size_t>(i) * N + j] += a * genB[k][j];
				}
			}
		}
		const float eps = 1e-2f;
		for (size_t i = 0; i < out_host.size(); ++i) {
			if (std::fabs(out_host[i] - ref[i]) > eps) {
				std::fprintf(stderr, "[CUTE GEMM SIMT] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::abort();
			}
		}
	}

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	report_summary(state);
}

// Benchmark: sparseMatrixMulTensor32x16_v2 (v2-style for 32x16 blocks)
static void bench_sparseMatrixMulTensor32x16_v2(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern, 32, 16);

	// grid.x corresponds to 32-column tiles (v2 pattern), grid.y to 32-row tiles
	dim3 gridSize{static_cast<unsigned int>((N + 31) / 32), static_cast<unsigned int>((M + 31) / 32), 1};
	dim3 blockSize{64, 1, 1};

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		timer.start();
		sparseMatrixMulTensor32x16_v2<<<gridSize, blockSize, 0, launch.get_stream()>>>(buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(M), static_cast<unsigned int>(N));
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
				std::fprintf(stderr, "[sparseMatrixMulTensor32x16_v2 kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s block_size=32x16\n", M, K, N, spars*100.0, pattern.c_str());
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

// Benchmark: sparseMatrixMulTensor32x16_shared (shared-memory staged A then WMMA)
static void bench_sparseMatrixMulTensor32x16_shared(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern, 32, 16);

	// grid.x corresponds to 32-column tiles (v2 pattern), grid.y to 32-row tiles
	dim3 gridSize{static_cast<unsigned int>((N + 31) / 32), static_cast<unsigned int>((M + 31) / 32), 1};
	dim3 blockSize{64, 1, 1};
	const unsigned int sharedBytes = static_cast<unsigned int>(32 * 16 * sizeof(half));

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		timer.start();
		sparseMatrixMulTensor32x16_shared<<<gridSize, blockSize, sharedBytes, launch.get_stream()>>>(
			buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC,
			static_cast<unsigned int>(M), static_cast<unsigned int>(N));
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
				std::fprintf(stderr, "[sparseMatrixMulTensor32x16_shared kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s block_size=32x16(shared)\n", M, K, N, spars*100.0, pattern.c_str());
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

// Benchmark: sparseMatrixMulTensor32x16_shared_vectorized (shared B staged with vectorized loads)
static void bench_sparseMatrixMulTensor32x16_shared_vectorized(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern, 32, 16);

	// grid.x corresponds to 32-column tiles (v2 pattern), grid.y to 32-row tiles
	dim3 gridSize{static_cast<unsigned int>((N + 31) / 32), static_cast<unsigned int>((M + 31) / 32), 1};
	dim3 blockSize{64, 1, 1};
	const unsigned int sharedBytes = static_cast<unsigned int>((32 * 16 + 16 * 32) * sizeof(half));

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		timer.start();
		sparseMatrixMulTensor32x16_shared_vectorized<<<gridSize, blockSize, sharedBytes, launch.get_stream()>>>(
			buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC,
			static_cast<unsigned int>(M), static_cast<unsigned int>(N));
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
				std::fprintf(stderr, "[sparseMatrixMulTensor32x16_shared_vectorized kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s block_size=32x16(shared_vectorized)\n", M, K, N, spars*100.0, pattern.c_str());
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

// Benchmark: sparseMatrixMulTensor32x16_v2_shared_vectorized (v2-style, 64 threads, shared vectorized staging)
static void bench_sparseMatrixMulTensor32x16_v2_shared_vectorized(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern, 32, 16);

	// grid.x corresponds to 32-column tiles (v2 pattern), grid.y to 32-row tiles
	dim3 gridSize{static_cast<unsigned int>((N + 31) / 32), static_cast<unsigned int>((M + 31) / 32), 1};
	dim3 blockSize{64, 1, 1};
	const unsigned int sharedBytes = static_cast<unsigned int>((32 * 16 + 16 * 32) * sizeof(half));

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		timer.start();
		sparseMatrixMulTensor32x16_v2_shared_vectorized<<<gridSize, blockSize, sharedBytes, launch.get_stream()>>>(
			buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC,
			static_cast<unsigned int>(M), static_cast<unsigned int>(N));
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
				std::fprintf(stderr, "[sparseMatrixMulTensor32x16_v2_shared_vectorized kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s block_size=32x16(v2_shared_vectorized)\n", M, K, N, spars*100.0, pattern.c_str());
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

// Benchmark: sparseMatrixMulTensor64x16_v2 (v2-style for 64x16 blocks)
static void bench_sparseMatrixMulTensor64x16_v2(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern, 64, 16);

	// grid.x corresponds to 32-column tiles (v2 pattern), grid.y to 64-row tiles
	dim3 gridSize{static_cast<unsigned int>((N + 31) / 32), static_cast<unsigned int>((M + 63) / 64), 1};
	dim3 blockSize{64, 1, 1};

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		timer.start();
		sparseMatrixMulTensor64x16_v2<<<gridSize, blockSize, 0, launch.get_stream()>>>(buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(M), static_cast<unsigned int>(N));
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
				std::fprintf(stderr, "[sparseMatrixMulTensor64x16_v2 kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s block_size=64x16\n", M, K, N, spars*100.0, pattern.c_str());
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

	auto buf = prepare_buffers(M, K, N, spars, pattern, 16, 16);

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
				std::fprintf(stderr, "[cuBLAS GEMM kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s (dense reference)\n", M, K, N, spars*100.0, pattern.c_str());
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

	auto buf = prepare_buffers(M, K, N, spars, pattern, 16, 16);

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
				std::fprintf(stderr, "[cuBLAS GEMM Tensor kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s (dense reference)\n", M, K, N, spars*100.0, pattern.c_str());
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

	auto buf = prepare_buffers(M, K, N, spars, pattern, 16, 16);

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
				std::fprintf(stderr, "[sparseMatrixMulTensor_option2_ldmatrix_sm80 kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s block_size=16x16\n", M, K, N, spars*100.0, pattern.c_str());
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

// Adaptive BCSR benchmark functions that balance load based on matrix aspect ratio
static void bench_sparseMatrixMulTensorAdaptive(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern, 16, 16);

	// Calculate aspect ratio and determine load balancing strategy
	const float aspectRatio = static_cast<float>(N) / static_cast<float>(M);
	unsigned int colBlocksPerRow = 1;
	
	if (aspectRatio >= 3.0f) {
		colBlocksPerRow = 3;
	} else if (aspectRatio >= 2.0f) {
		colBlocksPerRow = 2;
	}

	// Adjusted grid size (v2-style): X dimension is colBlocksPerRow, Y stays as BCSR rows
	const unsigned int bcsrRows = (M + 15) / 16;
	dim3 gridSize{colBlocksPerRow, bcsrRows, 1};
	dim3 blockSize{64, 1, 1}; // v2-style: 64 threads per block (2 warps)

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		timer.start();
		sparseMatrixMulTensorAdaptive<<<gridSize, blockSize, 0, launch.get_stream()>>>(
			buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC, 
			static_cast<unsigned int>(M), static_cast<unsigned int>(N), colBlocksPerRow);
		timer.stop();
	});

	// Verify results
	{
		std::vector<float> out_host(static_cast<size_t>(M) * N);
		cudaMemcpy(out_host.data(), buf->gpuC, out_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
		std::vector<float> ref;
		cpu_matmul_ref(buf->matrixA, buf->matrixB, ref);
		const float eps = 1e-2f;
		for (size_t i = 0; i < out_host.size(); ++i) {
			if (std::fabs(out_host[i] - ref[i]) > eps) {
				std::fprintf(stderr, "[Adaptive BCSR 16x16 kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s aspect_ratio=%.2f colBlocksPerRow=%u\n", 
					M, K, N, spars*100.0, pattern.c_str(), aspectRatio, colBlocksPerRow);
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

static void bench_sparseMatrixMulTensorAdaptive32x16(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern, 32, 16);

	// Calculate aspect ratio and determine load balancing strategy
	const float aspectRatio = static_cast<float>(N) / static_cast<float>(M);
	unsigned int colBlocksPerRow = 1;
	
	if (aspectRatio >= 3.0f) {
		colBlocksPerRow = 3;
	} else if (aspectRatio >= 2.0f) {
		colBlocksPerRow = 2;
	}

	// Adjusted grid size for 32x16 blocks (v2-style)
	const unsigned int bcsrRows = (M + 31) / 32;
	dim3 gridSize{colBlocksPerRow, bcsrRows, 1};
	dim3 blockSize{64, 1, 1}; // v2-style: 64 threads per block (2 warps)

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		timer.start();
		sparseMatrixMulTensorAdaptive32x16<<<gridSize, blockSize, 0, launch.get_stream()>>>(
			buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC, 
			static_cast<unsigned int>(M), static_cast<unsigned int>(N), colBlocksPerRow);
		timer.stop();
	});

	// Verify results
	{
		std::vector<float> out_host(static_cast<size_t>(M) * N);
		cudaMemcpy(out_host.data(), buf->gpuC, out_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
		std::vector<float> ref;
		cpu_matmul_ref(buf->matrixA, buf->matrixB, ref);
		const float eps = 1e-2f;
		for (size_t i = 0; i < out_host.size(); ++i) {
			if (std::fabs(out_host[i] - ref[i]) > eps) {
				std::fprintf(stderr, "[Adaptive BCSR 32x16 kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s aspect_ratio=%.2f colBlocksPerRow=%u\n", 
					M, K, N, spars*100.0, pattern.c_str(), aspectRatio, colBlocksPerRow);
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

static void bench_sparseMatrixMulTensorAdaptive64x16(nvbench::state &state) {
	const int M = static_cast<int>(state.get_int64("M"));
	const int K = static_cast<int>(state.get_int64("K"));
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(M, K, N, spars, pattern, 64, 16);

	// Calculate aspect ratio and determine load balancing strategy
	const float aspectRatio = static_cast<float>(N) / static_cast<float>(M);
	unsigned int colBlocksPerRow = 1;
	
	if (aspectRatio >= 3.0f) {
		colBlocksPerRow = 3;
	} else if (aspectRatio >= 2.0f) {
		colBlocksPerRow = 2;
	}

	// Adjusted grid size for 64x16 blocks (v2-style)
	const unsigned int bcsrRows = (M + 63) / 64;
	dim3 gridSize{colBlocksPerRow, bcsrRows, 1};
	dim3 blockSize{64, 1, 1}; // v2-style: 64 threads per block (2 warps)

	state.add_element_count(static_cast<size_t>(M) * N);
	state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer){
		cudaMemsetAsync(buf->gpuC, 0, static_cast<size_t>(M) * N * sizeof(float), launch.get_stream());
		timer.start();
		sparseMatrixMulTensorAdaptive64x16<<<gridSize, blockSize, 0, launch.get_stream()>>>(
			buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC, 
			static_cast<unsigned int>(M), static_cast<unsigned int>(N), colBlocksPerRow);
		timer.stop();
	});

	// Verify results
	{
		std::vector<float> out_host(static_cast<size_t>(M) * N);
		cudaMemcpy(out_host.data(), buf->gpuC, out_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
		std::vector<float> ref;
		cpu_matmul_ref(buf->matrixA, buf->matrixB, ref);
		const float eps = 1e-2f;
		for (size_t i = 0; i < out_host.size(); ++i) {
			if (std::fabs(out_host[i] - ref[i]) > eps) {
				std::fprintf(stderr, "[Adaptive BCSR 64x16 kernel] Mismatch at index %zu: got %f expected %f\n", i, out_host[i], ref[i]);
				std::fprintf(stderr, "Test specs: M=%d K=%d N=%d sparsity=%.1f%% pattern=%s aspect_ratio=%.2f colBlocksPerRow=%u\n", 
					M, K, N, spars*100.0, pattern.c_str(), aspectRatio, colBlocksPerRow);
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

// Fat Matrix Benchmarks - Testing matrices with high aspect ratios for load balancing
NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 65, 70, 75, 80, 85, 90}).add_int64_axis("PAT", {5});
NVBENCH_BENCH(bench_sparseMatrixMulTensor_v2).set_name("sparseMatrixMulTensor_v2").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 65, 70, 75, 80, 85, 90}).add_int64_axis("PAT", {5});
NVBENCH_BENCH(bench_sparseMatrixMulTensor32x16).set_name("sparseMatrixMulTensor32x16").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 65, 70, 75, 80, 85, 90}).add_int64_axis("PAT", {6});
NVBENCH_BENCH(bench_sparseMatrixMulTensor32x16_v2).set_name("sparseMatrixMulTensor32x16_v2").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 65, 70, 75, 80, 85, 90}).add_int64_axis("PAT", {6});
NVBENCH_BENCH(bench_sparseMatrixMulTensorAdaptive).set_name("sparseMatrixMulTensorAdaptive_16x16").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 65, 70, 75, 80, 85, 90}).add_int64_axis("PAT", {5});
NVBENCH_BENCH(bench_sparseMatrixMulTensorAdaptive32x16).set_name("sparseMatrixMulTensorAdaptive_32x16").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 65, 70, 75, 80, 85, 90}).add_int64_axis("PAT", {6});
NVBENCH_BENCH(bench_sparseMatrixMulTensor32x16_shared_vectorized).set_name("sparseMatrixMulTensor32x16_shared_vectorized").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 65, 70, 75, 80, 85, 90}).add_int64_axis("PAT", {6});
NVBENCH_BENCH(bench_sparseMatrixMulTensor32x16_v2_shared_vectorized).set_name("sparseMatrixMulTensor32x16_v2_shared_vectorized").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 65, 70, 75, 80, 85, 90}).add_int64_axis("PAT", {6});
// NVBENCH_BENCH(bench_cute_gemm_tensor).set_name("cute_gemm_tensor").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 65, 70, 75, 80, 85, 90}).add_int64_axis("PAT", {5});
// NVBENCH_BENCH(bench_cute_gemm_simt).set_name("cute_gemm_simt").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {1024}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 65, 70, 75, 80, 85, 90}).add_int64_axis("PAT", {5});

// size 512 512 for same methods
NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
NVBENCH_BENCH(bench_sparseMatrixMulTensor_v2).set_name("sparseMatrixMulTensor_v2").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
NVBENCH_BENCH(bench_sparseMatrixMulTensor32x16).set_name("sparseMatrixMulTensor32x16").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
NVBENCH_BENCH(bench_sparseMatrixMulTensor32x16_v2).set_name("sparseMatrixMulTensor32x16_v2").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
NVBENCH_BENCH(bench_sparseMatrixMulTensorAdaptive).set_name("sparseMatrixMulTensorAdaptive_16x16").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
NVBENCH_BENCH(bench_sparseMatrixMulTensorAdaptive32x16).set_name("sparseMatrixMulTensorAdaptive_32x16").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis	("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
NVBENCH_BENCH(bench_sparseMatrixMulTensor32x16_shared_vectorized).set_name("sparseMatrixMulTensor32x16_shared_vectorized").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
NVBENCH_BENCH(bench_sparseMatrixMulTensor32x16_v2_shared_vectorized).set_name("sparseMatrixMulTensor32x16_v2_shared_vectorized").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
// NVBENCH_BENCH(bench_cute_gemm_tensor).set_name("cute_gemm_tensor").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
// NVBENCH_BENCH(bench_cute_gemm_simt).set_name("cute_gemm_simt").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {512}).add_int64_axis("K", {512}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});


// size 2048 256 for same methods
NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {2048}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2	,3,4,5,6});
NVBENCH_BENCH(bench_sparseMatrixMulTensor_v2).set_name("sparseMatrixMulTensor_v2").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {2048}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
NVBENCH_BENCH(bench_sparseMatrixMulTensor32x16).set_name("sparseMatrixMulTensor32x16").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {2048}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
NVBENCH_BENCH(bench_sparseMatrixMulTensor32x16_v2).set_name("sparseMatrixMulTensor32x16_v2").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {2048}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
NVBENCH_BENCH(bench_sparseMatrixMulTensorAdaptive).set_name("sparseMatrixMulTensorAdaptive_16x16").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {2048}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
NVBENCH_BENCH(bench_sparseMatrixMulTensorAdaptive32x16).set_name("sparseMatrixMulTensorAdaptive_32x16").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {2048}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
NVBENCH_BENCH(bench_sparseMatrixMulTensor32x16_shared_vectorized).set_name("sparseMatrixMulTensor32x16_shared_vectorized").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {2048}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
NVBENCH_BENCH(bench_sparseMatrixMulTensor32x16_v2_shared_vectorized).set_name("sparseMatrixMulTensor32x16_v2_shared_vectorized").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {2048}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
// NVBENCH_BENCH(bench_cute_gemm_tensor).set_name("cute_gemm_tensor").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {2048}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
// NVBENCH_BENCH(bench_cute_gemm_simt).set_name("cute_gemm_simt").add_int64_axis("N", {32, 64, 128}).add_int64_axis("M", {2048}).add_int64_axis("K", {256}).add_int64_axis("SPARS", {60, 70, 80, 90}).add_int64_axis("PAT", {0,1,2,3,4,5,6});
