#include "heterogeneous_common.hpp"
#include "starpu_ptr.hpp"

#include <cuda_runtime.h>
#include <math.h>
#include <starpu.h>
#include <starpu_cuda.h>

__device__ __host__ static inline double hetero_op(double x, int kind)
{
  double y = x;
  const int iterations = kind == 0 ? 4 : (kind == 1 ? 16 : 64);
  for (int i = 0; i < iterations; ++i)
    y = sin(y) + sqrt(fabs(y) + 1.0) + 0.0001 * y * y;
  return y;
}

__global__ void hetero_flat_kernel(const double *input, double *output,
                                   const std::size_t *offsets, const std::size_t *sizes,
                                   const int *kinds, unsigned task_count)
{
  const unsigned task = blockIdx.y;
  if (task >= task_count)
    return;

  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= sizes[task])
    return;

  const std::size_t offset = offsets[task];
  output[offset + i] = hetero_op(input[offset + i], kinds[task]);
}

__global__ void hetero_task_kernel(const double *input, double *output, std::size_t n, int kind)
{
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    output[i] = hetero_op(input[i], kind);
}

static int kind_to_int(HeteroTaskKind kind)
{
  return kind == HeteroTaskKind::Light ? 0 : (kind == HeteroTaskKind::Medium ? 1 : 2);
}

void heterogeneous_native_gpu(const double *input_host, double *output_host,
                              const std::size_t *offsets_host, const std::size_t *sizes_host,
                              const int *kinds_host, unsigned task_count,
                              std::size_t total_elements)
{
  double *input = nullptr;
  double *output = nullptr;
  std::size_t *offsets = nullptr;
  std::size_t *sizes = nullptr;
  int *kinds = nullptr;

  cudaMalloc(&input, total_elements * sizeof(double));
  cudaMalloc(&output, total_elements * sizeof(double));
  cudaMalloc(&offsets, task_count * sizeof(std::size_t));
  cudaMalloc(&sizes, task_count * sizeof(std::size_t));
  cudaMalloc(&kinds, task_count * sizeof(int));
  cudaMemcpy(input, input_host, total_elements * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(offsets, offsets_host, task_count * sizeof(std::size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(sizes, sizes_host, task_count * sizeof(std::size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(kinds, kinds_host, task_count * sizeof(int), cudaMemcpyHostToDevice);

  std::size_t max_size = 0;
  for (unsigned t = 0; t < task_count; ++t)
    max_size = max_size < sizes_host[t] ? sizes_host[t] : max_size;

  const unsigned threads = 256;
  const dim3 blocks(static_cast<unsigned>((max_size + threads - 1) / threads), task_count);
  hetero_flat_kernel<<<blocks, threads>>>(input, output, offsets, sizes, kinds, task_count);
  cudaDeviceSynchronize();

  cudaMemcpy(output_host, output, total_elements * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(input);
  cudaFree(output);
  cudaFree(offsets);
  cudaFree(sizes);
  cudaFree(kinds);
}

extern "C" void cuda_hetero_codelet(void *buffers[], void *cl_arg)
{
  const auto *args = static_cast<const HeteroTaskArgs *>(cl_arg);
  const double *input = starpu_vector_ptr<const double>(buffers[0]);
  double *output = starpu_vector_ptr<double>(buffers[1]);

  const unsigned threads = 256;
  const unsigned blocks = static_cast<unsigned>((args->n + threads - 1) / threads);
  cudaStream_t stream = starpu_cuda_get_local_stream();
  hetero_task_kernel<<<blocks, threads, 0, stream>>>(input, output, args->n, kind_to_int(args->kind));
  cudaStreamSynchronize(stream);
}
