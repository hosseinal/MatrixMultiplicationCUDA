// nvbench harness to run selected kernels from main.cu across patterns/sizes/sparsities
#include <nvbench/nvbench.cuh>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <vector>
#include <string>
#include <memory>

#include "matrix_generator.h"
// Include implementation so templates are available in this TU (quick solution)
#include "matrix_generator.cpp"

#include "Matrix.cuh"
#include "CSRMatrix.cuh"
#include "BCSRMatrix.cuh"
#include "HCSRMatrix.h"

// The matrix generator provides mg::Matrix<T> as a templated alias.
// We don't need to alias it here; the project's Matrix type is declared
// in "Matrix.cuh" which is included below.

// Forward-declare kernels from main.cu so this translation unit can
// call them as CUDA kernels. Signatures must match the definitions
// in main.cu. Do NOT use extern "C" here — CUDA kernel symbols are
// emitted by nvcc with device linkage and C++ linkage; adding
// extern "C" prevents the correct linkage and causes undefined
// references at link time.
__global__ void denseMatrixMul(const half *d_A, const half *d_B, float *d_C, const unsigned int n);
__global__ void denseMatrixMulTensor(const half *d_A, const half *d_B, float *d_C, const unsigned int n);
__global__ void sparseMatrixMult1(const int *hdr, const int *idx, const half *data, const half *B, float *C, const unsigned int n);
__global__ void sparseMatrixMult1Co(const int *hdr, const int *idx, const half *data, const half *B, float *C, const unsigned int n);
__global__ void sparseMatrixMulTensor(const int *hdr, const int *idx, const half *data, const half *B, float *C, const unsigned int n);
__global__ void sparseMatrixMulTensor1(const int *hdr, const int *idx, const half *data, const half *B, float *C, const unsigned int n);
__global__ void addMatrices(float *C, const float *CPart, const unsigned int n);

// Local constant to match main.cu's thread configuration
constexpr unsigned int N_THREADS = 32;

static const std::vector<std::string> patterns = {
	"random",
	"checkerboard",
	"diagonal",
	"blockdiagonal",
	"blockrandom"
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

static std::unique_ptr<GenDeviceBuffers> prepare_buffers(int N, double sparsity, const std::string &pattern) {
	auto out = std::make_unique<GenDeviceBuffers>();
	// Generate float matrices with generator
	auto genA = mg::generate_matrix<float>(N, N, sparsity, pattern, 16, 123);
	auto genB = mg::generate_matrix<float>(N, N, 0.0, "random", 16, 456);

	out->matrixA = new Matrix(N, N);
	out->matrixB = new Matrix(N, N);
	fill_Matrix_from_generated(*out->matrixA, genA);
	fill_Matrix_from_generated(*out->matrixB, genB);

	// Build sparse representations from matrixA
	out->csrA = new CSRMatrix(*out->matrixA);
	out->bcsrA = new BCSRMatrix(*out->matrixA);

	// Copy CSR/BCSR to device
	out->bcsrA->copyToDevice(&out->gpuBCSRHdr, &out->gpuBCSRIdx, &out->gpuBCSRData);
	out->csrA->copyToDevice(&out->gpuCSRHdr, &out->gpuCSRIdx, &out->gpuCSRData);

	size_t bytes_half = static_cast<size_t>(N) * N * sizeof(half);
	size_t bytes_float = static_cast<size_t>(N) * N * sizeof(float);
	cudaMalloc(reinterpret_cast<void **>(&out->gpuA_half), bytes_half);
	cudaMalloc(reinterpret_cast<void **>(&out->gpuB_half), bytes_half);
	cudaMalloc(reinterpret_cast<void **>(&out->gpuC), bytes_float);
	cudaMalloc(reinterpret_cast<void **>(&out->gpuCPart), bytes_float);
	cudaMemcpy(out->gpuA_half, out->matrixA->data, bytes_half, cudaMemcpyHostToDevice);
	cudaMemcpy(out->gpuB_half, out->matrixB->data, bytes_half, cudaMemcpyHostToDevice);
	cudaMemset(out->gpuC, 0, bytes_float);
	cudaMemset(out->gpuCPart, 0, bytes_float);

	return out;
}

// Benchmark: denseMatrixMul (naive)
static void bench_denseMatrixMul(nvbench::state &state) {
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(N, spars, pattern);

	// grid/block similar to main.cu naive kernel
	dim3 gridSize{static_cast<unsigned int>(N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0)), static_cast<unsigned int>(N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0)), 1};
	dim3 blockSize{N_THREADS, N_THREADS, 1};

	state.add_element_count(static_cast<size_t>(N) * N);
	state.exec([&](nvbench::launch &launch){
		denseMatrixMul<<<gridSize, blockSize, 0, launch.get_stream()>>>(buf->gpuA_half, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(N));
		cudaStreamSynchronize(launch.get_stream());
	});
}

// Benchmark: denseMatrixMulTensor (wmma)
static void bench_denseMatrixMulTensor(nvbench::state &state) {
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(N, spars, pattern);

	dim3 gridSize{static_cast<unsigned int>(N / 16), static_cast<unsigned int>(N / 16), 1};
	dim3 blockSize{32, 1, 1};

	state.add_element_count(static_cast<size_t>(N) * N);
	state.exec([&](nvbench::launch &launch){
		denseMatrixMulTensor<<<gridSize, blockSize, 0, launch.get_stream()>>>(buf->gpuA_half, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(N));
		cudaStreamSynchronize(launch.get_stream());
	});
}

// Benchmark: sparseMatrixMult1
static void bench_sparseMatrixMult1(nvbench::state &state) {
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(N, spars, pattern);

	dim3 gridSize{static_cast<unsigned int>(N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0)), static_cast<unsigned int>(N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0)), 1};
	dim3 blockSize{N_THREADS, N_THREADS, 1};

	state.add_element_count(static_cast<size_t>(N) * N);
	state.exec([&](nvbench::launch &launch){
		sparseMatrixMult1<<<gridSize, blockSize, 0, launch.get_stream()>>>(buf->gpuCSRHdr, buf->gpuCSRIdx, buf->gpuCSRData, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(N));
		cudaStreamSynchronize(launch.get_stream());
	});
}

// Benchmark: sparseMatrixMulTensor (BCSR tensor)
static void bench_sparseMatrixMulTensor(nvbench::state &state) {
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(N, spars, pattern);

	dim3 gridSize{static_cast<unsigned int>(N / 16), static_cast<unsigned int>(N / 16), 1};
	dim3 blockSize{32, 1, 1};

	state.add_element_count(static_cast<size_t>(N) * N);
	state.exec([&](nvbench::launch &launch){
		sparseMatrixMulTensor<<<gridSize, blockSize, 0, launch.get_stream()>>>(buf->gpuBCSRHdr, buf->gpuBCSRIdx, buf->gpuBCSRData, buf->gpuB_half, buf->gpuC, static_cast<unsigned int>(N));
		cudaStreamSynchronize(launch.get_stream());
	});
}

// Benchmark: cuBLAS (GEMM) - no tensor ops
static void bench_cuBLAS(nvbench::state &state) {
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(N, spars, pattern);

	cublasHandle_t handle;
	cublasCreate(&handle);
	constexpr float alpha = 1.0f;
	constexpr float beta = 0.0f;
	state.add_element_count(static_cast<size_t>(N) * N);
	state.exec([&](nvbench::launch &launch){
		cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, buf->gpuB_half, CUDA_R_16F, N, buf->gpuA_half, CUDA_R_16F, N, &beta, buf->gpuC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
	});
	cublasDestroy(handle);
}

// Benchmark: cuBLAS with Tensor Cores
static void bench_cuBLAS_Tensor(nvbench::state &state) {
	const int N = static_cast<int>(state.get_int64("N"));
	const int sparsP = static_cast<int>(state.get_int64("SPARS"));
	const int patIdx = static_cast<int>(state.get_int64("PAT"));
	const double spars = sparsP / 100.0;
	const std::string pattern = patterns.at(patIdx % patterns.size());

	auto buf = prepare_buffers(N, spars, pattern);

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
	constexpr float alpha = 1.0f;
	constexpr float beta = 0.0f;
	// warm up
	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, buf->gpuB_half, CUDA_R_16F, N, buf->gpuA_half, CUDA_R_16F, N, &beta, buf->gpuC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

	state.add_element_count(static_cast<size_t>(N) * N);
	state.exec([&](nvbench::launch &launch){
		cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, buf->gpuB_half, CUDA_R_16F, N, buf->gpuA_half, CUDA_R_16F, N, &beta, buf->gpuC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	});

	cublasDestroy(handle);
}

// Register benches and axes
NVBENCH_BENCH(bench_denseMatrixMul).set_name("denseMatrixMul").add_int64_axis("N", {128, 256, 512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_denseMatrixMulTensor).set_name("denseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_sparseMatrixMult1).set_name("sparseMatrixMult1").add_int64_axis("N", {128, 256, 512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_sparseMatrixMulTensor).set_name("sparseMatrixMulTensor").add_int64_axis("N", {128, 256, 512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_cuBLAS).set_name("cuBLAS_GEMM").add_int64_axis("N", {128, 256, 512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});
NVBENCH_BENCH(bench_cuBLAS_Tensor).set_name("cuBLAS_GEMM_TENSOR").add_int64_axis("N", {128, 256, 512}).add_int64_axis("SPARS", {50,60,70,80,90}).add_int64_axis("PAT", {0,1,2,3,4});


