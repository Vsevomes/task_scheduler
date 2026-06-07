#include "independent_common.hpp"
#include "starpu_ptr.hpp"

#include "execution_mode.hpp"
#include "metrics.hpp"
#include "starpu_runtime.hpp"
#include "timer.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <starpu.h>
#include <vector>

static double independent_op(double x) { return std::sin(x) + std::sqrt(x) + x * x; }

struct StarpuTimings {
  double init_ms = 0.0;
  double data_registration_ms = 0.0;
  double task_submission_ms = 0.0;
  double wait_ms = 0.0;
  double data_unregister_ms = 0.0;
};

static void independent_native_cpu(const double *input, double *output, std::size_t total_elements)
{
  for (std::size_t i = 0; i < total_elements; ++i)
    output[i] = independent_op(input[i]);
}

static void cpu_independent(void *buffers[], void *cl_arg)
{
  const auto *args = static_cast<const IndependentArgs *>(cl_arg);
  const double *input = starpu_vector_ptr<const double>(buffers[0]);
  double *output = starpu_vector_ptr<double>(buffers[1]);
  for (std::size_t i = 0; i < args->n; ++i)
    output[i] = independent_op(input[i]);
}

static struct starpu_codelet independent_cl = {
    .where = STARPU_CPU | STARPU_CUDA,
    .cpu_funcs = {cpu_independent},
    .cuda_funcs = {cuda_independent_codelet},
    .nbuffers = 2,
    .modes = {STARPU_R, STARPU_W},
    .name = "independent_array",
};

static void run_starpu_batch(const double *input, double *output, std::size_t block_size,
                             unsigned task_count, ExecutionMode mode, StarpuTimings *timings)
{
  ChronoTimer phase;
  phase.start();
  if (init_starpu_for_mode(mode) != 0)
    std::exit(1);
  timings->init_ms = phase.elapsed_ms();

  std::vector<IndependentArgs> args(task_count);
  std::vector<starpu_data_handle_t> input_handles(task_count);
  std::vector<starpu_data_handle_t> output_handles(task_count);

  phase.start();
  for (unsigned t = 0; t < task_count; ++t) {
    const std::size_t offset = static_cast<std::size_t>(t) * block_size;
    args[t].n = block_size;
    starpu_vector_data_register(&input_handles[t], STARPU_MAIN_RAM,
                                reinterpret_cast<uintptr_t>(input + offset), block_size,
                                sizeof(double));
    starpu_vector_data_register(&output_handles[t], STARPU_MAIN_RAM,
                                reinterpret_cast<uintptr_t>(output + offset), block_size,
                                sizeof(double));
  }
  timings->data_registration_ms = phase.elapsed_ms();

  phase.start();
  for (unsigned t = 0; t < task_count; ++t) {
    struct starpu_task *task = starpu_task_create();
    task->cl = &independent_cl;
    task->handles[0] = input_handles[t];
    task->handles[1] = output_handles[t];
    task->cl_arg = &args[t];
    task->cl_arg_size = sizeof(IndependentArgs);
    task->destroy = 0;
    starpu_task_submit(task);
  }
  timings->task_submission_ms = phase.elapsed_ms();

  phase.start();
  starpu_task_wait_for_all();
  timings->wait_ms = phase.elapsed_ms();

  phase.start();
  for (unsigned t = 0; t < task_count; ++t) {
    starpu_data_unregister(input_handles[t]);
    starpu_data_unregister(output_handles[t]);
  }
  shutdown_starpu();
  timings->data_unregister_ms = phase.elapsed_ms();
}

static void print_usage(const char *prog)
{
  std::fprintf(stderr,
               "Usage: %s --mode MODE --tasks N --size N [--output PATH]\n"
               "Modes: native_cpu native_gpu starpu_hybrid\n",
               prog);
}

int main(int argc, char **argv)
{
  ExecutionMode mode = ExecutionMode::StarpuHybrid;
  unsigned task_count = 1000;
  std::size_t block_size = 4096;
  std::string output_path;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
      mode = parse_execution_mode(argv[++i]);
    else if (std::strcmp(argv[i], "--tasks") == 0 && i + 1 < argc)
      task_count = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc)
      block_size = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
      output_path = argv[++i];
    else if (std::strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }

  if (task_count == 0 || block_size == 0) {
    std::fprintf(stderr, "tasks and size must be greater than zero\n");
    return 1;
  }

  const std::size_t total_elements = static_cast<std::size_t>(task_count) * block_size;
  std::vector<double> input(total_elements);
  std::vector<double> output(total_elements);

  std::mt19937 rng(7);
  std::uniform_real_distribution<double> dist(0.0, 10.0);
  for (auto &v : input)
    v = dist(rng);

  ChronoTimer timer;
  StarpuTimings starpu_timings;
  timer.start();
  if (mode == ExecutionMode::NativeCpu)
    independent_native_cpu(input.data(), output.data(), total_elements);
  else if (mode == ExecutionMode::NativeGpu)
    independent_native_gpu(input.data(), output.data(), total_elements);
  else
    run_starpu_batch(input.data(), output.data(), block_size, task_count, mode, &starpu_timings);
  const double total_ms = timer.elapsed_ms();

  MetricsWriter writer;
  writer.set_scenario("independent");
  writer.set_mode(execution_mode_name(mode));
  writer.set_param("block_size", std::to_string(block_size));
  writer.set_param("tasks", std::to_string(task_count));
  writer.set_param("total_elements", std::to_string(total_elements));
  writer.set_metric("total_time_ms", total_ms);
  writer.set_metric("task_count", static_cast<double>(task_count));
  writer.set_metric("avg_task_ms", total_ms / task_count);
  if (uses_starpu(mode)) {
    writer.set_metric("starpu_init_ms", starpu_timings.init_ms);
    writer.set_metric("starpu_data_registration_ms", starpu_timings.data_registration_ms);
    writer.set_metric("starpu_task_submission_ms", starpu_timings.task_submission_ms);
    writer.set_metric("starpu_wait_ms", starpu_timings.wait_ms);
    writer.set_metric("starpu_data_unregister_ms", starpu_timings.data_unregister_ms);
  }
  if (output_path.empty())
    output_path = default_result_path("independent", execution_mode_name(mode));
  writer.write_json(output_path);

  std::printf("bench_independent: tasks=%u block=%zu mode=%s time=%.3f ms -> %s\n",
              task_count, block_size, execution_mode_name(mode), total_ms, output_path.c_str());
  return 0;
}
