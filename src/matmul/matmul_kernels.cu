#include "starpu_cublas_helper.hpp"
#include "starpu_ptr.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <starpu.h>

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

  cublasHandle_t handle;
  cublasCreate(&handle);
  const double alpha = 1.0;
  const double beta = 0.0;
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(n),
              static_cast<int>(n), &alpha, a, static_cast<int>(n), b, static_cast<int>(n), &beta, c,
              static_cast<int>(n));
  cudaDeviceSynchronize();

  cudaMemcpy(c_host, c, bytes, cudaMemcpyDeviceToHost);
  cublasDestroy(handle);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

extern "C" void cuda_matmul_codelet(void *buffers[], void * /*cl_arg*/)
{
  const std::size_t n = STARPU_MATRIX_GET_NX(buffers[0]);
  const unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);
  double *a = starpu_matrix_ptr<double>(buffers[0]);
  double *b = starpu_matrix_ptr<double>(buffers[1]);
  double *c = starpu_matrix_ptr<double>(buffers[2]);

  cublasHandle_t handle = task_scheduler_cublas_handle();
  const double alpha = 1.0;
  const double beta = 0.0;
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(n),
              static_cast<int>(n), &alpha, a, static_cast<int>(ld), b, static_cast<int>(ld), &beta,
              c, static_cast<int>(ld));
}
