#pragma once

#include <cstddef>

enum class HeteroTaskKind { LightMatvec, HeavyMatmul };

struct HeteroTaskArgs {
  HeteroTaskKind kind;
  std::size_t n;
};

extern "C" void cuda_hetero_codelet(void *buffers[], void *cl_arg);
