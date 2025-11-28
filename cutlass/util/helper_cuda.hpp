// Minimal helper_cuda.hpp stub for local builds
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

inline void checkCudaError(cudaError_t err, const char* msg = "CUDA Error") {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s (%d)\n", msg, cudaGetErrorString(err), static_cast<int>(err));
        std::fflush(stderr);
        std::abort();
    }
}

// Convenience macro similar to CUTLASS helper
#define CHECK_CUDA_ERROR(call) checkCudaError((call), #call)
