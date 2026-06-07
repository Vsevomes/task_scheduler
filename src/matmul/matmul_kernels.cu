#include "matmul_common.hpp"
#include "starpu_ptr.hpp"

#include <cuda_runtime.h>
#include <starpu.h>

__global__ void matmul_kernel(const double *a, const double *b, double *c, std::size_t n)
{
  const std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n || col >= n)
    return;

  double sum = 0.0;
  for (std::size_t k = 0; k < n; ++k)
    sum += a[row * n + k] * b[k * n + col];
  c[row * n + col] = sum;
}

__global__ void matmul_tile_kernel(const double *a, const double *b, double *tile,
                                   MatmulTileArgs args)
{
  const std::size_t local_row = blockIdx.y * blockDim.y + threadIdx.y;
  const std::size_t local_col = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_row >= args.tile_rows || local_col >= args.tile_cols)
    return;

  const std::size_t row = args.row + local_row;
  const std::size_t col = args.col + local_col;
  double sum = 0.0;
  for (std::size_t k = 0; k < args.n; ++k)
    sum += a[row * args.n + k] * b[k * args.n + col];
  tile[local_row * args.tile_cols + local_col] = sum;
}

void matmul_native_gpu(const double *a_host, const double *b_host, double *c_host, std::size_t n)
{
  double *a = nullptr;
  double *b = nullptr;
  double *c = nullptr;
  const std::size_t bytes = n * n * sizeof(double);
  cudaMalloc(&a, bytes);
  cudaMalloc(&b, bytes);
  cudaMalloc(&c, bytes);
  cudaMemcpy(a, a_host, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(b, b_host, bytes, cudaMemcpyHostToDevice);

  const dim3 threads(16, 16);
  const dim3 blocks(static_cast<unsigned>((n + threads.x - 1) / threads.x),
                    static_cast<unsigned>((n + threads.y - 1) / threads.y));
  matmul_kernel<<<blocks, threads>>>(a, b, c, n);
  cudaDeviceSynchronize();

  cudaMemcpy(c_host, c, bytes, cudaMemcpyDeviceToHost);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

extern "C" void cuda_matmul_tile_codelet(void *buffers[], void *cl_arg)
{
  const auto *args = static_cast<const MatmulTileArgs *>(cl_arg);
  const double *a = starpu_vector_ptr<const double>(buffers[0]);
  const double *b = starpu_vector_ptr<const double>(buffers[1]);
  double *tile = starpu_vector_ptr<double>(buffers[2]);

  const dim3 threads(16, 16);
  const dim3 blocks(static_cast<unsigned>((args->tile_cols + threads.x - 1) / threads.x),
                    static_cast<unsigned>((args->tile_rows + threads.y - 1) / threads.y));
  matmul_tile_kernel<<<blocks, threads>>>(a, b, tile, *args);
  cudaDeviceSynchronize();
}
