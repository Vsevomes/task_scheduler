#pragma once

#include <cstddef>

struct IndependentArgs {
  std::size_t n;
};

extern "C" void cuda_independent_codelet(void *buffers[], void *cl_arg);

void independent_native_gpu(const double *input, double *output, std::size_t total_elements);
