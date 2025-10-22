#include <nvbench/nvbench.cuh>

#include <cuda/std/chrono>

#include <cuda_runtime.h>
#include <vector>
#include <random>


__global__ void sleep_kernel(nvbench::int64_t microseconds) {
  const auto start = cuda::std::chrono::high_resolution_clock::now();
  const auto target_duration = cuda::std::chrono::microseconds(microseconds);
  const auto finish = start + target_duration;

  while (cuda::std::chrono::high_resolution_clock::now() < finish) {
    // busy wait
  }
}

void sleep_benchmark(nvbench::state &state) {
  const auto duration_us = state.get_int64("Duration (us)");
  state.exec([&duration_us](nvbench::launch &launch) {
    sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(duration_us);
  });
}
NVBENCH_BENCH(sleep_benchmark)
    .add_int64_axis("Duration (us)", nvbench::range(0, 100, 5))
    .set_timeout(1); // Limit to one second per measurement.

__global__ void vec_add_kernel(const float *a, const float *b, float *c, std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

void vector_add_benchmark(nvbench::state &state) {
  const auto N = static_cast<std::size_t>(state.get_int64("N"));
  const auto block = static_cast<int>(state.get_int64("Block"));

  // prepare host data
  std::vector<float> h_a(N), h_b(N);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < N; ++i) { h_a[i] = dist(rng); h_b[i] = dist(rng); }

  // allocate device buffers once per benchmark invocation
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_b, N * sizeof(float));
  cudaMalloc(&d_c, N * sizeof(float));
  cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

  state.add_element_count(N);

  const int grid = static_cast<int>((N + block - 1) / block);

  state.exec([=](nvbench::launch &launch) {
    vec_add_kernel<<<grid, block, 0, launch.get_stream()>>>(d_a, d_b, d_c, N);
  });

  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

NVBENCH_BENCH(vector_add_benchmark)
    .add_int64_axis("N", {1 << 10, 1 << 16, 1 << 20})
    .add_int64_axis("Block", {128, 256, 512});
