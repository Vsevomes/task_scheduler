#include "starpu_ptr.hpp"

#include "execution_mode.hpp"
#include "metrics.hpp"
#include "starpu_runtime.hpp"
#include "timer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <starpu.h>
#include <vector>

enum class OverheadKind { Noop, Memcpy };

static void cpu_noop(void * /*buffers*/[], void * /*cl_arg*/) {}

extern "C" void cuda_noop(void * /*buffers*/[], void * /*cl_arg*/);

static void cpu_memcpy_task(void *buffers[], void * /*cl_arg*/)
{
  const unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
  char *src = starpu_vector_ptr<char>(buffers[0]);
  char *dst = starpu_vector_ptr<char>(buffers[1]);
  std::memcpy(dst, src, n);
}

extern "C" void cuda_memcpy_task(void *buffers[], void *cl_arg);

static struct starpu_codelet noop_cl = {
    .where = STARPU_CPU | STARPU_CUDA,
    .cpu_funcs = {cpu_noop},
    .cuda_funcs = {cuda_noop},
    .nbuffers = 0,
    .name = "noop",
};

static struct starpu_codelet memcpy_cl = {
    .where = STARPU_CPU | STARPU_CUDA,
    .cpu_funcs = {cpu_memcpy_task},
    .cuda_funcs = {cuda_memcpy_task},
    .nbuffers = 2,
    .modes = {STARPU_R, STARPU_W},
    .name = "memcpy",
};

static void run_overhead(OverheadKind kind, unsigned task_count, ExecutionMode mode,
                         std::size_t bytes)
{
  if (init_starpu_for_mode(mode) != 0)
    std::exit(1);

  std::vector<char> src(bytes, 1);
  std::vector<char> dst(bytes, 0);
  starpu_data_handle_t h_src, h_dst;

  if (kind == OverheadKind::Memcpy) {
    starpu_vector_data_register(&h_src, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(src.data()),
                                bytes, 1);
    starpu_vector_data_register(&h_dst, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(dst.data()),
                                bytes, 1);
  }

  for (unsigned t = 0; t < task_count; ++t) {
    struct starpu_task *task = starpu_task_create();
    if (kind == OverheadKind::Noop) {
      task->cl = &noop_cl;
    } else {
      task->cl = &memcpy_cl;
      task->handles[0] = h_src;
      task->handles[1] = h_dst;
    }
    task->destroy = 0;
    starpu_task_submit(task);
  }

  starpu_task_wait_for_all();

  if (kind == OverheadKind::Memcpy) {
    starpu_data_unregister(h_src);
    starpu_data_unregister(h_dst);
  }
  shutdown_starpu();
}

static void print_usage(const char *prog)
{
  std::fprintf(stderr,
               "Usage: %s --mode MODE --kind noop|memcpy --tasks N [--bytes N]\n",
               prog);
}

int main(int argc, char **argv)
{
  ExecutionMode mode = ExecutionMode::StarpuHybrid;
  OverheadKind kind = OverheadKind::Noop;
  unsigned task_count = 10000;
  std::size_t bytes = 1 << 20;
  std::string output;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
      mode = parse_execution_mode(argv[++i]);
    else if (std::strcmp(argv[i], "--kind") == 0 && i + 1 < argc) {
      kind = std::strcmp(argv[++i], "memcpy") == 0 ? OverheadKind::Memcpy : OverheadKind::Noop;
    } else if (std::strcmp(argv[i], "--tasks") == 0 && i + 1 < argc)
      task_count = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc)
      bytes = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
      output = argv[++i];
    else if (std::strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }

  if (!uses_starpu(mode)) {
    std::fprintf(stderr, "bench_overhead requires a StarPU mode\n");
    return 1;
  }

  ChronoTimer timer;
  timer.start();
  run_overhead(kind, task_count, mode, bytes);
  const double total_ms = timer.elapsed_ms();

  MetricsWriter writer;
  writer.set_scenario("overhead");
  writer.set_mode(execution_mode_name(mode));
  writer.set_param("kind", kind == OverheadKind::Noop ? "noop" : "memcpy");
  writer.set_param("tasks", std::to_string(task_count));
  writer.set_param("bytes", std::to_string(bytes));
  writer.set_metric("total_time_ms", total_ms);
  writer.set_metric("avg_task_us", (total_ms * 1000.0) / task_count);
  if (output.empty())
    output = default_result_path("overhead", execution_mode_name(mode));
  writer.write_json(output);

  std::printf("bench_overhead: kind=%s tasks=%u mode=%s time=%.3f ms -> %s\n",
              kind == OverheadKind::Noop ? "noop" : "memcpy", task_count,
              execution_mode_name(mode), total_ms, output.c_str());
  return 0;
}
