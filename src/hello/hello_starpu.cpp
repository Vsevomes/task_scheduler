#include "execution_mode.hpp"
#include "metrics.hpp"
#include "starpu_ptr.hpp"
#include "starpu_runtime.hpp"
#include "timer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <starpu.h>
#include <vector>

static void cpu_add_one(void *buffers[], void * /*cl_arg*/)
{
  const unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
  float *vec = starpu_vector_ptr<float>(buffers[0]);
  for (unsigned i = 0; i < n; ++i)
    vec[i] += 1.f;
}

extern "C" void cuda_add_one(void *buffers[], void *cl_arg);

static struct starpu_codelet add_one_cl = {
    .where = STARPU_CPU | STARPU_CUDA,
    .cpu_funcs = {cpu_add_one},
    .cuda_funcs = {cuda_add_one},
    .nbuffers = 1,
    .modes = {STARPU_RW},
    .name = "add_one",
};

static void print_usage(const char *prog)
{
  std::fprintf(stderr, "Usage: %s [--mode MODE] [--size N] [--output PATH]\n", prog);
  std::fprintf(stderr, "Modes: starpu_cpu, starpu_gpu, starpu_hybrid\n");
}

int main(int argc, char **argv)
{
  ExecutionMode mode = ExecutionMode::StarpuHybrid;
  unsigned n = 1 << 20;
  std::string output;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
      mode = parse_execution_mode(argv[++i]);
    else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc)
      n = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
      output = argv[++i];
    else if (std::strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }

  if (!uses_starpu(mode)) {
    std::fprintf(stderr, "hello_starpu requires a StarPU mode\n");
    return 1;
  }

  if (init_starpu_for_mode(mode) != 0) {
    std::fprintf(stderr, "starpu_init failed\n");
    return 1;
  }

  std::vector<float> data(n, 0.f);
  starpu_data_handle_t handle;
  starpu_vector_data_register(&handle, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(data.data()), n,
                              sizeof(float));

  ChronoTimer timer;
  timer.start();

  struct starpu_task *task = starpu_task_create();
  task->cl = &add_one_cl;
  task->handles[0] = handle;
  task->destroy = 0;
  starpu_task_submit(task);
  starpu_task_wait_for_all();

  const double elapsed_ms = timer.elapsed_ms();

  starpu_data_acquire(handle, STARPU_R);
  starpu_data_release(handle);
  starpu_data_unregister(handle);
  shutdown_starpu();

  MetricsWriter writer;
  writer.set_scenario("hello");
  writer.set_mode(execution_mode_name(mode));
  writer.set_param("size", std::to_string(n));
  writer.set_metric("total_time_ms", elapsed_ms);
  writer.set_metric("task_count", 1.0);
  if (output.empty())
    output = default_result_path("hello", execution_mode_name(mode));
  writer.write_json(output);

  std::printf("hello_starpu ok: n=%u mode=%s time=%.3f ms -> %s\n", n,
              execution_mode_name(mode), elapsed_ms, output.c_str());
  return 0;
}
