#pragma once

#include <cstddef>

struct MatmulTileArgs {
  std::size_t n;
  std::size_t row;
  std::size_t col;
  std::size_t tile_rows;
  std::size_t tile_cols;
};

extern "C" void cuda_matmul_tile_codelet(void *buffers[], void *cl_arg);

void matmul_native_gpu(const double *a, const double *b, double *c, std::size_t n);
