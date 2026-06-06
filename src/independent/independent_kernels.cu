#include "independent_common.hpp"
#include "starpu_cublas_helper.hpp"
#include "starpu_ptr.hpp"

#include <cublas_v2.h>
#include <starpu.h>

extern "C" void cuda_matvec_codelet(void *buffers[], void *cl_arg)
{
  const auto *args = static_cast<const MatvecArgs *>(cl_arg);
  const std::size_t n = args->n;
  double *a = starpu_vector_ptr<double>(buffers[0]);
  double *x = starpu_vector_ptr<double>(buffers[1]);
  double *y = starpu_vector_ptr<double>(buffers[2]);

  cublasHandle_t handle = task_scheduler_cublas_handle();
  const double alpha = 1.0;
  const double beta = 0.0;
  cublasDgemv(handle, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(n), &alpha, a,
              static_cast<int>(n), x, 1, &beta, y, 1);
}
