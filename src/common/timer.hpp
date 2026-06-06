#pragma once

#include <chrono>
#include <cuda_runtime.h>
#include <string>

class ChronoTimer {
public:
  void start() { start_ = std::chrono::steady_clock::now(); }

  double elapsed_ms() const
  {
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
  }

private:
  std::chrono::steady_clock::time_point start_;
};

class CudaEventTimer {
public:
  CudaEventTimer()
  {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~CudaEventTimer()
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() { cudaEventRecord(start_); }

  double elapsed_ms()
  {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start_, stop_);
    return static_cast<double>(ms);
  }

private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};
