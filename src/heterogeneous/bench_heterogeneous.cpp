#include "heterogeneous_common.hpp"
#include "starpu_ptr.hpp"

#include "execution_mode.hpp"
#include "metrics.hpp"
#include "starpu_runtime.hpp"
#include "timer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <starpu.h>
#include <vector>

static int kind_to_int(HeteroTaskKind kind)
{
  return kind == HeteroTaskKind::Light ? 0 : (kind == HeteroTaskKind::Medium ? 1 : 2);
}

struct StarpuTimings {
  double init_ms = 0.0;
  double data_registration_ms = 0.0;
  double task_submission_ms = 0.0;
  double wait_ms = 0.0;
  double data_unregister_ms = 0.0;
};

static double hetero_op(double x, HeteroTaskKind kind)
{
  double y = x;
  const int iterations = kind == HeteroTaskKind::Light ? 4 : (kind == HeteroTaskKind::Medium ? 16 : 64);
  for (int i = 0; i < iterations; ++i)
    y = std::sin(y) + std::sqrt(std::fabs(y) + 1.0) + 0.0001 * y * y;
  return y;
}

static void run_cpu_task(const double *input, double *output, std::size_t n, HeteroTaskKind kind)
{
  for (std::size_t i = 0; i < n; ++i)
    output[i] = hetero_op(input[i], kind);
}

static void cpu_hetero(void *buffers[], void *cl_arg)
{
  const auto *args = static_cast<const HeteroTaskArgs *>(cl_arg);
  const double *input = starpu_vector_ptr<const double>(buffers[0]);
  double *output = starpu_vector_ptr<double>(buffers[1]);
  run_cpu_task(input, output, args->n, args->kind);
}

static struct starpu_perfmodel hetero_perfmodel = {
    .type = STARPU_HISTORY_BASED,
    .symbol = "heterogeneous_array",
};

static struct starpu_codelet hetero_cl = {
    .where = STARPU_CPU | STARPU_CUDA,
    .cpu_funcs = {cpu_hetero},
    .cuda_funcs = {cuda_hetero_codelet},
    .nbuffers = 2,
    .modes = {STARPU_R, STARPU_W},
    .model = &hetero_perfmodel,
    .name = "heterogeneous_array",
};

static void print_usage(const char *prog)
{
  std::fprintf(stderr,
               "Usage: %s --mode MODE --tasks N --light-size N --medium-size N --heavy-size N "
               "[--light-ratio R --medium-ratio R --output PATH]\n"
               "Modes: native_cpu native_gpu starpu_hybrid\n",
               prog);
}

int main(int argc, char **argv)
{
  ExecutionMode mode = ExecutionMode::StarpuHybrid;
  unsigned task_count = 900;
  double light_ratio = 0.5;
  double medium_ratio = 0.3;
  std::size_t light_n = 1024;
  std::size_t medium_n = 4096;
  std::size_t heavy_n = 16384;
  std::string output_path;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
      mode = parse_execution_mode(argv[++i]);
    else if (std::strcmp(argv[i], "--tasks") == 0 && i + 1 < argc)
      task_count = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--light-ratio") == 0 && i + 1 < argc)
      light_ratio = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--medium-ratio") == 0 && i + 1 < argc)
      medium_ratio = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--light-size") == 0 && i + 1 < argc)
      light_n = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--medium-size") == 0 && i + 1 < argc)
      medium_n = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--heavy-size") == 0 && i + 1 < argc)
      heavy_n = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
      output_path = argv[++i];
    else if (std::strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }

  if (light_ratio < 0.0)
    light_ratio = 0.0;
  if (medium_ratio < 0.0)
    medium_ratio = 0.0;
  if (light_ratio + medium_ratio > 1.0)
    medium_ratio = 1.0 - light_ratio;

  if (task_count == 0 || light_n == 0 || medium_n == 0 || heavy_n == 0) {
    std::fprintf(stderr, "tasks and task sizes must be greater than zero\n");
    return 1;
  }

  const unsigned light_tasks = static_cast<unsigned>(task_count * light_ratio);
  const unsigned medium_tasks = static_cast<unsigned>(task_count * medium_ratio);
  const unsigned heavy_tasks = task_count - light_tasks - medium_tasks;

  std::vector<HeteroTaskArgs> args(task_count);
  std::vector<std::size_t> offsets(task_count);
  std::vector<std::size_t> sizes(task_count);
  std::vector<int> kinds(task_count);
  std::vector<HeteroTaskKind> task_kinds;
  task_kinds.reserve(task_count);
  task_kinds.insert(task_kinds.end(), light_tasks, HeteroTaskKind::Light);
  task_kinds.insert(task_kinds.end(), medium_tasks, HeteroTaskKind::Medium);
  task_kinds.insert(task_kinds.end(), heavy_tasks, HeteroTaskKind::Heavy);

  std::mt19937 rng(11);
  std::shuffle(task_kinds.begin(), task_kinds.end(), rng);

  std::size_t total_elements = 0;
  for (unsigned t = 0; t < task_count; ++t) {
    const HeteroTaskKind kind = task_kinds[t];

    const std::size_t n =
        kind == HeteroTaskKind::Light ? light_n : (kind == HeteroTaskKind::Medium ? medium_n : heavy_n);
    args[t] = HeteroTaskArgs{kind, n};
    offsets[t] = total_elements;
    sizes[t] = n;
    kinds[t] = kind_to_int(kind);
    total_elements += n;
  }

  std::vector<double> input(total_elements);
  std::vector<double> output(total_elements);
  std::uniform_real_distribution<double> dist(0.0, 10.0);
  for (auto &v : input)
    v = dist(rng);

  ChronoTimer timer;
  StarpuTimings starpu_timings;
  timer.start();

  if (mode == ExecutionMode::NativeCpu) {
    for (unsigned t = 0; t < task_count; ++t)
      run_cpu_task(input.data() + offsets[t], output.data() + offsets[t], sizes[t], args[t].kind);
  } else if (mode == ExecutionMode::NativeGpu) {
    heterogeneous_native_gpu(input.data(), output.data(), offsets.data(), sizes.data(), kinds.data(),
                             task_count, total_elements);
  } else {
    ChronoTimer phase;
    phase.start();
    if (init_starpu_for_mode(mode) != 0)
      return 1;
    starpu_timings.init_ms = phase.elapsed_ms();

    std::vector<starpu_data_handle_t> input_handles(task_count);
    std::vector<starpu_data_handle_t> output_handles(task_count);

    phase.start();
    for (unsigned t = 0; t < task_count; ++t) {
      starpu_vector_data_register(&input_handles[t], STARPU_MAIN_RAM,
                                  reinterpret_cast<uintptr_t>(input.data() + offsets[t]), sizes[t],
                                  sizeof(double));
      starpu_vector_data_register(&output_handles[t], STARPU_MAIN_RAM,
                                  reinterpret_cast<uintptr_t>(output.data() + offsets[t]), sizes[t],
                                  sizeof(double));
    }
    starpu_timings.data_registration_ms = phase.elapsed_ms();

    phase.start();
    for (unsigned t = 0; t < task_count; ++t) {
      struct starpu_task *task = starpu_task_create();
      task->cl = &hetero_cl;
      task->handles[0] = input_handles[t];
      task->handles[1] = output_handles[t];
      task->cl_arg = &args[t];
      task->cl_arg_size = sizeof(HeteroTaskArgs);
      task->destroy = 0;
      starpu_task_submit(task);
    }
    starpu_timings.task_submission_ms = phase.elapsed_ms();

    phase.start();
    starpu_task_wait_for_all();
    starpu_timings.wait_ms = phase.elapsed_ms();

    phase.start();
    for (unsigned t = 0; t < task_count; ++t) {
      starpu_data_unregister(input_handles[t]);
      starpu_data_unregister(output_handles[t]);
    }
    shutdown_starpu();
    starpu_timings.data_unregister_ms = phase.elapsed_ms();
  }

  const double total_ms = timer.elapsed_ms();

  MetricsWriter writer;
  writer.set_scenario("heterogeneous");
  writer.set_mode(execution_mode_name(mode));
  writer.set_param("tasks", std::to_string(task_count));
  writer.set_param("light_size", std::to_string(light_n));
  writer.set_param("medium_size", std::to_string(medium_n));
  writer.set_param("heavy_size", std::to_string(heavy_n));
  writer.set_param("total_elements", std::to_string(total_elements));
  writer.set_metric("total_time_ms", total_ms);
  writer.set_metric("light_tasks", static_cast<double>(light_tasks));
  writer.set_metric("medium_tasks", static_cast<double>(medium_tasks));
  writer.set_metric("heavy_tasks", static_cast<double>(heavy_tasks));
  if (uses_starpu(mode)) {
    const char *sched = std::getenv("STARPU_SCHED");
    writer.set_param("starpu_sched", (sched != nullptr && sched[0] != '\0') ? sched : "dmda");
    writer.set_metric("starpu_init_ms", starpu_timings.init_ms);
    writer.set_metric("starpu_data_registration_ms", starpu_timings.data_registration_ms);
    writer.set_metric("starpu_task_submission_ms", starpu_timings.task_submission_ms);
    writer.set_metric("starpu_wait_ms", starpu_timings.wait_ms);
    writer.set_metric("starpu_data_unregister_ms", starpu_timings.data_unregister_ms);
  }
  if (output_path.empty())
    output_path = default_result_path("heterogeneous", execution_mode_name(mode));
  writer.write_json(output_path);

  std::printf("bench_heterogeneous: tasks=%u light=%u medium=%u heavy=%u mode=%s time=%.3f ms -> %s\n",
              task_count, light_tasks, medium_tasks, heavy_tasks, execution_mode_name(mode),
              total_ms, output_path.c_str());
  return 0;
}
