#include "matmul_common.hpp"
#include "starpu_ptr.hpp"

#include "execution_mode.hpp"
#include "metrics.hpp"
#include "starpu_runtime.hpp"
#include "timer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <starpu.h>
#include <vector>

static void fill_random(double *matrix, std::size_t n)
{
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (std::size_t i = 0; i < n * n; ++i)
    matrix[i] = dist(rng);
}

struct StarpuTimings {
  double init_ms = 0.0;
  double data_registration_ms = 0.0;
  double task_submission_ms = 0.0;
  double wait_ms = 0.0;
  double data_unregister_ms = 0.0;
};

static void matmul_native_cpu(const double *a, const double *b, double *c, std::size_t n)
{
  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      double sum = 0.0;
      for (std::size_t k = 0; k < n; ++k)
        sum += a[row * n + k] * b[k * n + col];
      c[row * n + col] = sum;
    }
  }
}

static void cpu_matmul_tile(void *buffers[], void *cl_arg)
{
  const auto *args = static_cast<const MatmulTileArgs *>(cl_arg);
  const double *a = starpu_vector_ptr<const double>(buffers[0]);
  const double *b = starpu_vector_ptr<const double>(buffers[1]);
  double *tile = starpu_vector_ptr<double>(buffers[2]);

  for (std::size_t local_row = 0; local_row < args->tile_rows; ++local_row) {
    const std::size_t row = args->row + local_row;
    for (std::size_t local_col = 0; local_col < args->tile_cols; ++local_col) {
      const std::size_t col = args->col + local_col;
      double sum = 0.0;
      for (std::size_t k = 0; k < args->n; ++k)
        sum += a[row * args->n + k] * b[k * args->n + col];
      tile[local_row * args->tile_cols + local_col] = sum;
    }
  }
}

static struct starpu_perfmodel matmul_tile_perfmodel = {
    .type = STARPU_HISTORY_BASED,
    .symbol = "matmul_tile",
};

static struct starpu_codelet matmul_tile_cl = {
    .where = STARPU_CPU | STARPU_CUDA,
    .cpu_funcs = {cpu_matmul_tile},
    .cuda_funcs = {cuda_matmul_tile_codelet},
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_W},
    .model = &matmul_tile_perfmodel,
    .name = "matmul_tile",
};

static unsigned matmul_starpu(const double *a, const double *b, double *c, std::size_t n,
                              std::size_t tile_size, ExecutionMode mode, StarpuTimings *timings)
{
  ChronoTimer phase;
  phase.start();
  if (init_starpu_for_mode(mode) != 0)
    std::exit(1);
  timings->init_ms = phase.elapsed_ms();

  starpu_data_handle_t ha, hb;
  phase.start();
  starpu_vector_data_register(&ha, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(a), n * n,
                              sizeof(double));
  starpu_vector_data_register(&hb, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(b), n * n,
                              sizeof(double));

  const unsigned tiles_x = static_cast<unsigned>((n + tile_size - 1) / tile_size);
  const unsigned tiles_y = static_cast<unsigned>((n + tile_size - 1) / tile_size);
  const unsigned task_count = tiles_x * tiles_y;
  std::vector<MatmulTileArgs> args(task_count);
  std::vector<std::vector<double>> tiles(task_count);
  std::vector<starpu_data_handle_t> tile_handles(task_count);

  unsigned idx = 0;
  for (unsigned ty = 0; ty < tiles_y; ++ty) {
    for (unsigned tx = 0; tx < tiles_x; ++tx) {
      const std::size_t row = static_cast<std::size_t>(ty) * tile_size;
      const std::size_t col = static_cast<std::size_t>(tx) * tile_size;
      const std::size_t tile_rows = std::min(tile_size, n - row);
      const std::size_t tile_cols = std::min(tile_size, n - col);
      args[idx] = MatmulTileArgs{n, row, col, tile_rows, tile_cols};
      tiles[idx].resize(tile_rows * tile_cols);

      starpu_vector_data_register(&tile_handles[idx], STARPU_MAIN_RAM,
                                  reinterpret_cast<uintptr_t>(tiles[idx].data()), tiles[idx].size(),
                                  sizeof(double));
      ++idx;
    }
  }
  timings->data_registration_ms = phase.elapsed_ms();

  phase.start();
  idx = 0;
  for (unsigned ty = 0; ty < tiles_y; ++ty) {
    for (unsigned tx = 0; tx < tiles_x; ++tx) {
      struct starpu_task *task = starpu_task_create();
      task->cl = &matmul_tile_cl;
      task->handles[0] = ha;
      task->handles[1] = hb;
      task->handles[2] = tile_handles[idx];
      task->cl_arg = &args[idx];
      task->cl_arg_size = sizeof(MatmulTileArgs);
      task->destroy = 0;
      starpu_task_submit(task);
      ++idx;
    }
  }
  timings->task_submission_ms = phase.elapsed_ms();

  phase.start();
  starpu_task_wait_for_all();
  timings->wait_ms = phase.elapsed_ms();

  phase.start();
  for (unsigned t = 0; t < task_count; ++t) {
    starpu_data_unregister(tile_handles[t]);
    const auto &arg = args[t];
    for (std::size_t local_row = 0; local_row < arg.tile_rows; ++local_row) {
      std::memcpy(c + (arg.row + local_row) * n + arg.col,
                  tiles[t].data() + local_row * arg.tile_cols, arg.tile_cols * sizeof(double));
    }
  }
  starpu_data_unregister(ha);
  starpu_data_unregister(hb);
  shutdown_starpu();
  timings->data_unregister_ms = phase.elapsed_ms();
  return task_count;
}

static void print_usage(const char *prog)
{
  std::fprintf(stderr,
               "Usage: %s --mode MODE --size N [--tile-size N] [--output PATH]\n"
               "Modes: native_cpu native_gpu starpu_hybrid\n",
               prog);
}

int main(int argc, char **argv)
{
  ExecutionMode mode = ExecutionMode::StarpuHybrid;
  std::size_t n = 1024;
  std::size_t tile_size = 256;
  std::string output_path;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
      mode = parse_execution_mode(argv[++i]);
    else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc)
      n = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--tile-size") == 0 && i + 1 < argc)
      tile_size = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
      output_path = argv[++i];
    else if (std::strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }

  if (n == 0 || tile_size == 0) {
    std::fprintf(stderr, "size and tile-size must be greater than zero\n");
    return 1;
  }

  const std::size_t elems = n * n;
  std::vector<double> a(elems), b(elems), c(elems);
  fill_random(a.data(), n);
  fill_random(b.data(), n);

  ChronoTimer timer;
  StarpuTimings starpu_timings;
  unsigned task_count = 1;
  timer.start();

  if (mode == ExecutionMode::NativeCpu) {
    matmul_native_cpu(a.data(), b.data(), c.data(), n);
  } else if (mode == ExecutionMode::NativeGpu) {
    matmul_native_gpu(a.data(), b.data(), c.data(), n);
  } else {
    task_count = matmul_starpu(a.data(), b.data(), c.data(), n, tile_size, mode, &starpu_timings);
  }

  const double total_ms = timer.elapsed_ms();

  MetricsWriter writer;
  writer.set_scenario("matmul");
  writer.set_mode(execution_mode_name(mode));
  writer.set_param("size", std::to_string(n));
  writer.set_param("tile_size", std::to_string(tile_size));
  writer.set_param("submitted_tasks", std::to_string(task_count));
  writer.set_metric("total_time_ms", total_ms);
  writer.set_metric("task_count", static_cast<double>(task_count));
  writer.set_metric("gflops", (2.0 * n * n * n) / (total_ms * 1e6));
  if (uses_starpu(mode)) {
    const char *sched = std::getenv("STARPU_SCHED");
    writer.set_param("starpu_sched", (sched != nullptr && sched[0] != '\0') ? sched : "dmda");
    writer.set_metric("starpu_init_ms", starpu_timings.init_ms);
    writer.set_metric("starpu_data_registration_ms", starpu_timings.data_registration_ms);
    writer.set_metric("starpu_task_submission_ms", starpu_timings.task_submission_ms);
    writer.set_metric("starpu_wait_ms", starpu_timings.wait_ms);
    writer.set_metric("starpu_data_unregister_copyback_ms", starpu_timings.data_unregister_ms);
  }
  if (output_path.empty())
    output_path = default_result_path("matmul", execution_mode_name(mode));
  writer.write_json(output_path);

  std::printf("bench_matmul: n=%zu tile=%zu tasks=%u mode=%s time=%.3f ms gflops=%.2f -> %s\n",
              n, tile_size, task_count, execution_mode_name(mode), total_ms,
              (2.0 * n * n * n) / (total_ms * 1e6), output_path.c_str());
  return 0;
}
