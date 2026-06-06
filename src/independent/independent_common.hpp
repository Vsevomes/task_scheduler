#pragma once

#include <cstddef>

struct MatvecArgs {
  std::size_t n;
};

extern "C" void cuda_matvec_codelet(void *buffers[], void *cl_arg);
