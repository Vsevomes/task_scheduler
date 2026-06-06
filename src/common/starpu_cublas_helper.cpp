#include "starpu_cublas_helper.hpp"

#include <starpu_cublas_v2.h>

extern "C" cublasHandle_t task_scheduler_cublas_handle()
{
  return starpu_cublas_get_local_handle();
}
