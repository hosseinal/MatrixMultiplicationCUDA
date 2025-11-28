// Minimal GPU_Clock implementation using CUDA events
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

struct GPU_Clock {
  cudaEvent_t startEvent{nullptr}, stopEvent{nullptr};
  GPU_Clock() {
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
  }
  ~GPU_Clock() {
    if (startEvent) cudaEventDestroy(startEvent);
    if (stopEvent) cudaEventDestroy(stopEvent);
  }
  void start() {
    cudaEventRecord(startEvent, 0);
  }
  void stop() {
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
  }
  // Returns elapsed seconds between last start/stop
  double seconds() const {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    return static_cast<double>(ms) * 1e-3;
  }
};
