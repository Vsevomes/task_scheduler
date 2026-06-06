#include "starpu_ptr.hpp"

#include <starpu.h>

#include <cuda_runtime.h>

__global__ void add_one_kernel(float *vec, unsigned n)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    vec[i] += 1.f;
}

extern "C" void cuda_add_one(void *buffers[], void * /*cl_arg*/)
{
  const unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
  float *vec = starpu_vector_ptr<float>(buffers[0]);
  const unsigned blocks = (n + 255) / 256;
  add_one_kernel<<<blocks, 256>>>(vec, n);
  cudaDeviceSynchronize();
}
