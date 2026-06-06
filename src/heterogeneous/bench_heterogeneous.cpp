#include "heterogeneous_common.hpp"
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

static void cpu_hetero(void *buffers[], void *cl_arg)
{
  const auto *args = static_cast<const HeteroTaskArgs *>(cl_arg);
  if (args->kind == HeteroTaskKind::LightMatvec) {
    const std::size_t n = args->n;
    const double *a = starpu_vector_ptr<const double>(buffers[0]);
    const double *x = starpu_vector_ptr<const double>(buffers[1]);
    double *y = starpu_vector_ptr<double>(buffers[2]);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(n), static_cast<int>(n), 1.0, a,
                static_cast<int>(n), x, 1, 0.0, y, 1);
  } else {
    const std::size_t n = args->n;
    const unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);
    double *a = starpu_matrix_ptr<double>(buffers[0]);
    double *b = starpu_matrix_ptr<double>(buffers[1]);
    double *c = starpu_matrix_ptr<double>(buffers[2]);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n),
                static_cast<int>(n), static_cast<int>(n), 1.0, a, static_cast<int>(ld), b,
                static_cast<int>(ld), 0.0, c, static_cast<int>(ld));
  }
}

static struct starpu_codelet hetero_cl = {
    .where = STARPU_CPU | STARPU_CUDA,
    .cpu_funcs = {cpu_hetero},
    .cuda_funcs = {cuda_hetero_codelet},
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_W},
    .name = "hetero",
};

static void print_usage(const char *prog)
{
  std::fprintf(stderr,
               "Usage: %s --mode MODE --tasks N --light-ratio R --light-size N --heavy-size N\n",
               prog);
}

int main(int argc, char **argv)
{
  ExecutionMode mode = ExecutionMode::StarpuHybrid;
  unsigned task_count = 200;
  double light_ratio = 0.7;
  std::size_t light_n = 256;
  std::size_t heavy_n = 512;
  std::string output;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
      mode = parse_execution_mode(argv[++i]);
    else if (std::strcmp(argv[i], "--tasks") == 0 && i + 1 < argc)
      task_count = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--light-ratio") == 0 && i + 1 < argc)
      light_ratio = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--light-size") == 0 && i + 1 < argc)
      light_n = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--heavy-size") == 0 && i + 1 < argc)
      heavy_n = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
      output = argv[++i];
    else if (std::strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }

  if (!uses_starpu(mode)) {
    std::fprintf(stderr, "bench_heterogeneous requires a StarPU mode\n");
    return 1;
  }

  if (init_starpu_for_mode(mode) != 0)
    return 1;

  std::mt19937 rng(11);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::vector<double> light_a(light_n * light_n), light_x(light_n);
  std::vector<std::vector<double>> light_y(task_count, std::vector<double>(light_n));
  std::vector<double> heavy_a(heavy_n * heavy_n), heavy_b(heavy_n * heavy_n);
  std::vector<std::vector<double>> heavy_c(task_count, std::vector<double>(heavy_n * heavy_n));
  for (auto &v : light_a)
    v = dist(rng);
  for (auto &v : light_x)
    v = dist(rng);
  for (auto &v : heavy_a)
    v = dist(rng);
  for (auto &v : heavy_b)
    v = dist(rng);

  starpu_data_handle_t h_la, h_lx, h_ha, h_hb;
  std::vector<starpu_data_handle_t> h_ly(task_count), h_hc(task_count);
  starpu_vector_data_register(&h_la, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(light_a.data()),
                              light_n * light_n, sizeof(double));
  starpu_vector_data_register(&h_lx, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(light_x.data()),
                              light_n, sizeof(double));
  starpu_matrix_data_register(&h_ha, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(heavy_a.data()),
                              heavy_n, heavy_n, heavy_n, sizeof(double));
  starpu_matrix_data_register(&h_hb, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(heavy_b.data()),
                              heavy_n, heavy_n, heavy_n, sizeof(double));

  for (unsigned t = 0; t < task_count; ++t) {
    starpu_vector_data_register(&h_ly[t], STARPU_MAIN_RAM,
                                reinterpret_cast<uintptr_t>(light_y[t].data()), light_n,
                                sizeof(double));
    starpu_matrix_data_register(&h_hc[t], STARPU_MAIN_RAM,
                                reinterpret_cast<uintptr_t>(heavy_c[t].data()), heavy_n, heavy_n,
                                heavy_n, sizeof(double));
  }

  const unsigned light_tasks = static_cast<unsigned>(task_count * light_ratio);
  std::vector<HeteroTaskArgs> args(task_count);

  ChronoTimer timer;
  timer.start();

  for (unsigned t = 0; t < task_count; ++t) {
    const bool light = t < light_tasks;
    args[t].kind = light ? HeteroTaskKind::LightMatvec : HeteroTaskKind::HeavyMatmul;
    args[t].n = light ? light_n : heavy_n;

    struct starpu_task *task = starpu_task_create();
    task->cl = &hetero_cl;
    if (light) {
      task->handles[0] = h_la;
      task->handles[1] = h_lx;
      task->handles[2] = h_ly[t];
    } else {
      task->handles[0] = h_ha;
      task->handles[1] = h_hb;
      task->handles[2] = h_hc[t];
    }
    task->cl_arg = &args[t];
    task->cl_arg_size = sizeof(HeteroTaskArgs);
    task->destroy = 0;
    starpu_task_submit(task);
  }

  starpu_task_wait_for_all();
  const double total_ms = timer.elapsed_ms();

  starpu_data_unregister(h_la);
  starpu_data_unregister(h_lx);
  starpu_data_unregister(h_ha);
  starpu_data_unregister(h_hb);
  for (unsigned t = 0; t < task_count; ++t) {
    starpu_data_unregister(h_ly[t]);
    starpu_data_unregister(h_hc[t]);
  }
  shutdown_starpu();

  MetricsWriter writer;
  writer.set_scenario("heterogeneous");
  writer.set_mode(execution_mode_name(mode));
  writer.set_param("tasks", std::to_string(task_count));
  writer.set_param("light_ratio", std::to_string(light_ratio));
  writer.set_param("light_size", std::to_string(light_n));
  writer.set_param("heavy_size", std::to_string(heavy_n));
  writer.set_metric("total_time_ms", total_ms);
  writer.set_metric("light_tasks", static_cast<double>(light_tasks));
  writer.set_metric("heavy_tasks", static_cast<double>(task_count - light_tasks));
  if (output.empty())
    output = default_result_path("heterogeneous", execution_mode_name(mode));
  writer.write_json(output);

  std::printf("bench_heterogeneous: tasks=%u light=%.0f%% time=%.3f ms -> %s\n", task_count,
              light_ratio * 100.0, total_ms, output.c_str());
  return 0;
}
