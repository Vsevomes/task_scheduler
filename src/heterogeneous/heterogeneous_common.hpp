#pragma once

#include <cstddef>

enum class HeteroTaskKind { Light, Medium, Heavy };

struct HeteroTaskArgs {
  HeteroTaskKind kind;
  std::size_t n;
};

extern "C" void cuda_hetero_codelet(void *buffers[], void *cl_arg);

void heterogeneous_native_gpu(const double *input, double *output, const std::size_t *offsets,
                              const std::size_t *sizes, const int *kinds, unsigned task_count,
                              std::size_t total_elements);
