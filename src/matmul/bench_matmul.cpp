#include "starpu_ptr.hpp"

#include "execution_mode.hpp"
#include "metrics.hpp"
#include "starpu_runtime.hpp"
#include "timer.hpp"

#include <algorithm>
#include <cblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <starpu.h>
#include <vector>

void matmul_native_gpu(const double *a, const double *b, double *c, std::size_t n);

void fill_random(double *matrix, std::size_t n)
{
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (std::size_t i = 0; i < n * n; ++i)
    matrix[i] = dist(rng);
}

void matmul_native_cpu(const double *a, const double *b, double *c, std::size_t n)
{
  const double alpha = 1.0;
  const double beta = 0.0;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n),
              static_cast<int>(n), static_cast<int>(n), alpha, a, static_cast<int>(n), b,
              static_cast<int>(n), beta, c, static_cast<int>(n));
}

static void cpu_matmul(void *buffers[], void * /*cl_arg*/)
{
  const std::size_t n = STARPU_MATRIX_GET_NX(buffers[0]);
  const unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);
  double *a = starpu_matrix_ptr<double>(buffers[0]);
  double *b = starpu_matrix_ptr<double>(buffers[1]);
  double *c = starpu_matrix_ptr<double>(buffers[2]);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n),
              static_cast<int>(n), static_cast<int>(n), 1.0, a, static_cast<int>(ld), b,
              static_cast<int>(ld), 0.0, c, static_cast<int>(ld));
}

extern "C" void cuda_matmul_codelet(void *buffers[], void *cl_arg);

static struct starpu_codelet matmul_cl = {
    .where = STARPU_CPU | STARPU_CUDA,
    .cpu_funcs = {cpu_matmul},
    .cuda_funcs = {cuda_matmul_codelet},
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_W},
    .name = "matmul",
};

struct starpu_codelet *matmul_codelet() { return &matmul_cl; }

int matmul_starpu(double *a, double *b, double *c, std::size_t n, int /*starpu_ncuda*/,
                  int /*starpu_ncpu*/)
{
  starpu_data_handle_t ha, hb, hc;
  starpu_matrix_data_register(&ha, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(a), n, n, n,
                              sizeof(double));
  starpu_matrix_data_register(&hb, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(b), n, n, n,
                              sizeof(double));
  starpu_matrix_data_register(&hc, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(c), n, n, n,
                              sizeof(double));

  struct starpu_task *task = starpu_task_create();
  task->cl = &matmul_cl;
  task->handles[0] = ha;
  task->handles[1] = hb;
  task->handles[2] = hc;
  task->destroy = 0;
  const int ret = starpu_task_submit(task);
  starpu_task_wait_for_all();

  starpu_data_unregister(ha);
  starpu_data_unregister(hb);
  starpu_data_unregister(hc);
  return ret;
}

static void print_usage(const char *prog)
{
  std::fprintf(stderr,
               "Usage: %s --mode MODE --size N [--output PATH]\n"
               "Modes: native_cpu native_gpu starpu_cpu starpu_gpu starpu_hybrid\n",
               prog);
}

int main(int argc, char **argv)
{
  ExecutionMode mode = ExecutionMode::StarpuHybrid;
  std::size_t n = 1024;
  std::string output;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
      mode = parse_execution_mode(argv[++i]);
    else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc)
      n = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
      output = argv[++i];
    else if (std::strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }

  const std::size_t elems = n * n;
  std::vector<double> a(elems), b(elems), c(elems);
  fill_random(a.data(), n);
  fill_random(b.data(), n);

  ChronoTimer timer;
  double gpu_kernel_ms = 0.0;

  if (mode == ExecutionMode::NativeCpu) {
    timer.start();
    matmul_native_cpu(a.data(), b.data(), c.data(), n);
    gpu_kernel_ms = 0.0;
  } else if (mode == ExecutionMode::NativeGpu) {
    timer.start();
    matmul_native_gpu(a.data(), b.data(), c.data(), n);
    gpu_kernel_ms = timer.elapsed_ms();
  } else {
    if (init_starpu_for_mode(mode) != 0) {
      std::fprintf(stderr, "starpu_init failed\n");
      return 1;
    }
    timer.start();
    matmul_starpu(a.data(), b.data(), c.data(), n, 0, 0);
    shutdown_starpu();
  }

  const double total_ms = timer.elapsed_ms();

  MetricsWriter writer;
  writer.set_scenario("matmul");
  writer.set_mode(execution_mode_name(mode));
  writer.set_param("size", std::to_string(n));
  writer.set_metric("total_time_ms", total_ms);
  writer.set_metric("gpu_kernel_ms", gpu_kernel_ms);
  writer.set_metric("gflops", (2.0 * n * n * n) / (total_ms * 1e6));
  if (output.empty())
    output = default_result_path("matmul", execution_mode_name(mode));
  writer.write_json(output);

  std::printf("bench_matmul: n=%zu mode=%s time=%.3f ms gflops=%.2f -> %s\n", n,
              execution_mode_name(mode), total_ms, (2.0 * n * n * n) / (total_ms * 1e6),
              output.c_str());
  return 0;
}
