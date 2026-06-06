#include "independent_common.hpp"
#include "starpu_ptr.hpp"

#include "execution_mode.hpp"
#include "metrics.hpp"
#include "starpu_runtime.hpp"
#include "timer.hpp"

#include <cblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <starpu.h>
#include <vector>

static void cpu_matvec(void *buffers[], void *cl_arg)
{
  const auto *args = static_cast<const MatvecArgs *>(cl_arg);
  const std::size_t n = args->n;
  const double *a = starpu_vector_ptr<const double>(buffers[0]);
  const double *x = starpu_vector_ptr<const double>(buffers[1]);
  double *y = starpu_vector_ptr<double>(buffers[2]);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(n), static_cast<int>(n), 1.0, a,
              static_cast<int>(n), x, 1, 0.0, y, 1);
}

static struct starpu_codelet matvec_cl = {
    .where = STARPU_CPU | STARPU_CUDA,
    .cpu_funcs = {cpu_matvec},
    .cuda_funcs = {cuda_matvec_codelet},
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_W},
    .name = "matvec",
};

static void run_starpu_batch(std::size_t n, unsigned task_count, ExecutionMode mode)
{
  if (init_starpu_for_mode(mode) != 0)
    std::exit(1);

  std::mt19937 rng(7);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::vector<double> a(n * n);
  std::vector<double> x(n);
  for (auto &v : a)
    v = dist(rng);
  for (auto &v : x)
    v = dist(rng);

  starpu_data_handle_t ha, hx, hy;
  starpu_vector_data_register(&ha, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(a.data()), n * n,
                              sizeof(double));
  starpu_vector_data_register(&hx, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(x.data()), n,
                              sizeof(double));

  std::vector<MatvecArgs> args(task_count);
  std::vector<std::vector<double>> outputs(task_count, std::vector<double>(n));
  std::vector<starpu_data_handle_t> hy_handles(task_count);

  for (unsigned t = 0; t < task_count; ++t) {
    args[t].n = n;
    starpu_vector_data_register(&hy_handles[t], STARPU_MAIN_RAM,
                                reinterpret_cast<uintptr_t>(outputs[t].data()), n, sizeof(double));
    struct starpu_task *task = starpu_task_create();
    task->cl = &matvec_cl;
    task->handles[0] = ha;
    task->handles[1] = hx;
    task->handles[2] = hy_handles[t];
    task->cl_arg = &args[t];
    task->cl_arg_size = sizeof(MatvecArgs);
    task->destroy = 0;
    starpu_task_submit(task);
  }

  starpu_task_wait_for_all();
  starpu_data_unregister(ha);
  starpu_data_unregister(hx);
  for (unsigned t = 0; t < task_count; ++t)
    starpu_data_unregister(hy_handles[t]);
  shutdown_starpu();
}

static void print_usage(const char *prog)
{
  std::fprintf(stderr,
               "Usage: %s --mode MODE --tasks N --size N [--output PATH]\n"
               "Modes: starpu_cpu starpu_gpu starpu_hybrid\n",
               prog);
}

int main(int argc, char **argv)
{
  ExecutionMode mode = ExecutionMode::StarpuHybrid;
  unsigned task_count = 1000;
  std::size_t n = 512;
  std::string output;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
      mode = parse_execution_mode(argv[++i]);
    else if (std::strcmp(argv[i], "--tasks") == 0 && i + 1 < argc)
      task_count = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc)
      n = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
      output = argv[++i];
    else if (std::strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }

  if (!uses_starpu(mode)) {
    std::fprintf(stderr, "bench_independent requires a StarPU mode\n");
    return 1;
  }

  ChronoTimer timer;
  timer.start();
  run_starpu_batch(n, task_count, mode);
  const double total_ms = timer.elapsed_ms();

  MetricsWriter writer;
  writer.set_scenario("independent");
  writer.set_mode(execution_mode_name(mode));
  writer.set_param("size", std::to_string(n));
  writer.set_param("tasks", std::to_string(task_count));
  writer.set_metric("total_time_ms", total_ms);
  writer.set_metric("task_count", static_cast<double>(task_count));
  writer.set_metric("avg_task_ms", total_ms / task_count);
  if (output.empty())
    output = default_result_path("independent", execution_mode_name(mode));
  writer.write_json(output);

  std::printf("bench_independent: tasks=%u n=%zu mode=%s time=%.3f ms -> %s\n", task_count, n,
              execution_mode_name(mode), total_ms, output.c_str());
  return 0;
}
