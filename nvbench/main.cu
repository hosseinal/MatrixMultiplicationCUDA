#include <nvbench/nvbench.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cassert>

__global__ void vec_add(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float* __restrict__ c,
                        std::size_t n)
{
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

static void add_benchmark(nvbench::state& state)
{
  // Read axes (set below in NVBENCH_BENCH)
  const std::size_t N = static_cast<std::size_t>(state.get_int64("N"));
  const int BLOCK = static_cast<int>(state.get_int64("BLOCK_SIZE"));

  // Host data (just to have non-trivial inputs)
  std::vector<float> h_a(N), h_b(N);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (std::size_t i = 0; i < N; ++i) {
    h_a[i] = dist(rng);
    h_b[i] = dist(rng);
  }

  // Device buffers
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_b, N * sizeof(float));
  cudaMalloc(&d_c, N * sizeof(float));
  cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

  // Tell NVBench about the work (helps with derived metrics)
  state.add_element_count(N);                         // elements processed
  state.add_global_memory_reads<float>(2 * N);        // a + b
  state.add_global_memory_writes<float>(N);           // c
  state.add_flops(1.0 * N);                           // one add per element

  const int grid = static_cast<int>((N + BLOCK - 1) / BLOCK);

  state.exec([&](nvbench::launch& launch) {
    // Run on NVBench-provided stream
    vec_add<<<grid, BLOCK, 0, launch.get_stream()>>>(d_a, d_b, d_c, N);
  });

  // Clean up
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

// Register the benchmark and define axes.
NVBENCH_BENCH(add_benchmark)
  .set_name("vector_add")
  .add_int64_axis("N", {1 << 16, 1 << 18, 1 << 20})       // try a few sizes
  .add_int64_axis("BLOCK_SIZE", {128, 256, 512});         // try a few blocks
