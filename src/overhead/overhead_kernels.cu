#include "starpu_ptr.hpp"

#include <cuda_runtime.h>
#include <starpu.h>

extern "C" void cuda_noop(void * /*buffers*/[], void * /*cl_arg*/) {}

extern "C" void cuda_memcpy_task(void *buffers[], void * /*cl_arg*/)
{
  const unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
  char *src = starpu_vector_ptr<char>(buffers[0]);
  char *dst = starpu_vector_ptr<char>(buffers[1]);
  cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice);
}
