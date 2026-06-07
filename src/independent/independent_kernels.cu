#include "independent_common.hpp"
#include "starpu_ptr.hpp"

#include <cuda_runtime.h>
#include <math.h>
#include <starpu.h>

__device__ __host__ static inline double independent_op(double x)
{
  return sin(x) + sqrt(x) + x * x;
}

__global__ void independent_kernel(const double *input, double *output, std::size_t n)
{
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    output[i] = independent_op(input[i]);
}

void independent_native_gpu(const double *input_host, double *output_host,
                            std::size_t total_elements)
{
  double *input = nullptr;
  double *output = nullptr;
  const std::size_t bytes = total_elements * sizeof(double);
  cudaMalloc(&input, bytes);
  cudaMalloc(&output, bytes);
  cudaMemcpy(input, input_host, bytes, cudaMemcpyHostToDevice);

  const unsigned threads = 256;
  const unsigned blocks = static_cast<unsigned>((total_elements + threads - 1) / threads);
  independent_kernel<<<blocks, threads>>>(input, output, total_elements);
  cudaDeviceSynchronize();

  cudaMemcpy(output_host, output, bytes, cudaMemcpyDeviceToHost);
  cudaFree(input);
  cudaFree(output);
}

extern "C" void cuda_independent_codelet(void *buffers[], void *cl_arg)
{
  const auto *args = static_cast<const IndependentArgs *>(cl_arg);
  const double *input = starpu_vector_ptr<const double>(buffers[0]);
  double *output = starpu_vector_ptr<double>(buffers[1]);

  const unsigned threads = 256;
  const unsigned blocks = static_cast<unsigned>((args->n + threads - 1) / threads);
  independent_kernel<<<blocks, threads>>>(input, output, args->n);
  cudaDeviceSynchronize();
}
