#include "heterogeneous_common.hpp"
#include "starpu_cublas_helper.hpp"
#include "starpu_ptr.hpp"

#include <cublas_v2.h>
#include <starpu.h>

extern "C" void cuda_hetero_codelet(void *buffers[], void *cl_arg)
{
  const auto *args = static_cast<const HeteroTaskArgs *>(cl_arg);
  cublasHandle_t handle = task_scheduler_cublas_handle();
  const double alpha = 1.0;
  const double beta = 0.0;

  if (args->kind == HeteroTaskKind::LightMatvec) {
    const std::size_t n = args->n;
    double *a = starpu_vector_ptr<double>(buffers[0]);
    double *x = starpu_vector_ptr<double>(buffers[1]);
    double *y = starpu_vector_ptr<double>(buffers[2]);
    cublasDgemv(handle, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(n), &alpha, a,
                static_cast<int>(n), x, 1, &beta, y, 1);
  } else {
    const std::size_t n = args->n;
    const unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);
    double *a = starpu_matrix_ptr<double>(buffers[0]);
    double *b = starpu_matrix_ptr<double>(buffers[1]);
    double *c = starpu_matrix_ptr<double>(buffers[2]);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(n),
                static_cast<int>(n), &alpha, a, static_cast<int>(ld), b, static_cast<int>(ld),
                &beta, c, static_cast<int>(ld));
  }
}
