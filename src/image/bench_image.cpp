#include "image_common.hpp"
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
#include <string>
#include <vector>

static std::uint8_t clamp_u8(int value)
{
  return static_cast<std::uint8_t>(value < 0 ? 0 : (value > 255 ? 255 : value));
}

struct StarpuTimings {
  double init_ms = 0.0;
  double data_registration_ms = 0.0;
  double task_submission_ms = 0.0;
  double wait_ms = 0.0;
  double data_unregister_ms = 0.0;
};

static void process_image_cpu(const std::uint8_t *input, std::uint8_t *output, unsigned width,
                              unsigned height, ImageOp op)
{
  for (unsigned y = 0; y < height; ++y) {
    for (unsigned x = 0; x < width; ++x) {
      const unsigned idx = (y * width + x) * 3;
      if (op == ImageOp::Grayscale) {
        const std::uint8_t gray =
            static_cast<std::uint8_t>((input[idx] + input[idx + 1] + input[idx + 2]) / 3);
        output[idx] = output[idx + 1] = output[idx + 2] = gray;
        continue;
      }

      if (x == 0 || y == 0 || x + 1 >= width || y + 1 >= height) {
        output[idx] = input[idx];
        output[idx + 1] = input[idx + 1];
        output[idx + 2] = input[idx + 2];
        continue;
      }

      for (unsigned c = 0; c < 3; ++c) {
        if (op == ImageOp::Blur) {
          int sum = 0;
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
              sum += input[((y + dy) * width + (x + dx)) * 3 + c];
          output[idx + c] = static_cast<std::uint8_t>(sum / 9);
        } else if (op == ImageOp::Edge) {
          const int center = input[idx + c];
          const int left = input[(y * width + x - 1) * 3 + c];
          const int right = input[(y * width + x + 1) * 3 + c];
          const int up = input[((y - 1) * width + x) * 3 + c];
          const int down = input[((y + 1) * width + x) * 3 + c];
          output[idx + c] = clamp_u8(4 * center - left - right - up - down);
        } else if (op == ImageOp::Convolution) {
          const int center = input[idx + c];
          const int left = input[(y * width + x - 1) * 3 + c];
          const int right = input[(y * width + x + 1) * 3 + c];
          const int up = input[((y - 1) * width + x) * 3 + c];
          const int down = input[((y + 1) * width + x) * 3 + c];
          output[idx + c] = clamp_u8(5 * center - left - right - up - down);
        } else {
          int sum = 0;
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
              sum += input[((y + dy) * width + (x + dx)) * 3 + c];
          output[idx + c] = clamp_u8((input[idx + c] + sum / 9) / 2);
        }
      }
    }
  }
}

static void cpu_image_tile(void *buffers[], void *cl_arg)
{
  const auto *args = static_cast<const ImageArgs *>(cl_arg);
  auto *tile = starpu_vector_ptr<std::uint8_t>(buffers[0]);
  std::vector<std::uint8_t> input(static_cast<std::size_t>(args->width) * args->height * 3);
  std::memcpy(input.data(), tile, input.size());
  process_image_cpu(input.data(), tile, args->width, args->height, args->op);
}

static struct starpu_perfmodel image_tile_perfmodel = {
    .type = STARPU_HISTORY_BASED,
    .symbol = "image_tile",
};

static struct starpu_codelet image_tile_cl = {
    .where = STARPU_CPU | STARPU_CUDA,
    .cpu_funcs = {cpu_image_tile},
    .cuda_funcs = {cuda_image_tile_codelet},
    .nbuffers = 1,
    .modes = {STARPU_RW},
    .model = &image_tile_perfmodel,
    .name = "image_tile",
};

static ImageOp parse_image_op(const char *value)
{
  if (std::strcmp(value, "grayscale") == 0)
    return ImageOp::Grayscale;
  if (std::strcmp(value, "blur") == 0)
    return ImageOp::Blur;
  if (std::strcmp(value, "edge") == 0)
    return ImageOp::Edge;
  if (std::strcmp(value, "convolution") == 0)
    return ImageOp::Convolution;
  return ImageOp::Filter;
}

static const char *image_op_name(ImageOp op)
{
  switch (op) {
  case ImageOp::Grayscale:
    return "grayscale";
  case ImageOp::Blur:
    return "blur";
  case ImageOp::Edge:
    return "edge";
  case ImageOp::Convolution:
    return "convolution";
  case ImageOp::Filter:
    return "filter";
  }
  return "unknown";
}

static ImageOp tile_mixed_op(unsigned tile_index)
{
  switch (tile_index % 5) {
  case 0:
    return ImageOp::Grayscale;
  case 1:
    return ImageOp::Blur;
  case 2:
    return ImageOp::Edge;
  case 3:
    return ImageOp::Convolution;
  default:
    return ImageOp::Filter;
  }
}

static unsigned process_image_tiles_native_cpu(std::vector<std::uint8_t> &pixels, unsigned width,
                                               unsigned height, unsigned tile_size, ImageOp op,
                                               bool mixed_ops)
{
  const unsigned tiles_x = (width + tile_size - 1) / tile_size;
  const unsigned tiles_y = (height + tile_size - 1) / tile_size;
  const unsigned task_count = tiles_x * tiles_y;
  std::vector<std::uint8_t> result(pixels.size());

  unsigned idx = 0;
  for (unsigned ty = 0; ty < tiles_y; ++ty) {
    for (unsigned tx = 0; tx < tiles_x; ++tx) {
      const unsigned tw = std::min(tile_size, width - tx * tile_size);
      const unsigned th = std::min(tile_size, height - ty * tile_size);
      std::vector<std::uint8_t> input(static_cast<std::size_t>(tw) * th * 3);
      std::vector<std::uint8_t> output(input.size());

      for (unsigned y = 0; y < th; ++y) {
        const unsigned src_y = ty * tile_size + y;
        const unsigned src_off = (src_y * width + tx * tile_size) * 3;
        const unsigned dst_off = y * tw * 3;
        std::memcpy(input.data() + dst_off, pixels.data() + src_off, tw * 3);
      }

      process_image_cpu(input.data(), output.data(), tw, th, mixed_ops ? tile_mixed_op(idx) : op);

      for (unsigned y = 0; y < th; ++y) {
        const unsigned dst_y = ty * tile_size + y;
        const unsigned dst_off = (dst_y * width + tx * tile_size) * 3;
        const unsigned src_off = y * tw * 3;
        std::memcpy(result.data() + dst_off, output.data() + src_off, tw * 3);
      }
      ++idx;
    }
  }

  pixels.swap(result);
  return task_count;
}

static unsigned process_image_tiles_native_gpu(std::vector<std::uint8_t> &pixels, unsigned width,
                                               unsigned height, unsigned tile_size, ImageOp op,
                                               bool mixed_ops)
{
  const unsigned tiles_x = (width + tile_size - 1) / tile_size;
  const unsigned tiles_y = (height + tile_size - 1) / tile_size;
  const unsigned task_count = tiles_x * tiles_y;
  std::vector<std::uint8_t> result(pixels.size());

  unsigned idx = 0;
  for (unsigned ty = 0; ty < tiles_y; ++ty) {
    for (unsigned tx = 0; tx < tiles_x; ++tx) {
      const unsigned tw = std::min(tile_size, width - tx * tile_size);
      const unsigned th = std::min(tile_size, height - ty * tile_size);
      std::vector<std::uint8_t> input(static_cast<std::size_t>(tw) * th * 3);
      std::vector<std::uint8_t> output(input.size());

      for (unsigned y = 0; y < th; ++y) {
        const unsigned src_y = ty * tile_size + y;
        const unsigned src_off = (src_y * width + tx * tile_size) * 3;
        const unsigned dst_off = y * tw * 3;
        std::memcpy(input.data() + dst_off, pixels.data() + src_off, tw * 3);
      }

      image_native_gpu(input.data(), output.data(), tw, th, mixed_ops ? tile_mixed_op(idx) : op);

      for (unsigned y = 0; y < th; ++y) {
        const unsigned dst_y = ty * tile_size + y;
        const unsigned dst_off = (dst_y * width + tx * tile_size) * 3;
        const unsigned src_off = y * tw * 3;
        std::memcpy(result.data() + dst_off, output.data() + src_off, tw * 3);
      }
      ++idx;
    }
  }

  pixels.swap(result);
  return task_count;
}

static unsigned run_starpu_image(std::vector<std::uint8_t> &pixels, unsigned width, unsigned height,
                                 ImageOp op, unsigned tile_size, ExecutionMode mode,
                                 bool mixed_ops, StarpuTimings *timings)
{
  ChronoTimer phase;
  phase.start();
  if (init_starpu_for_mode(mode) != 0)
    std::exit(1);
  timings->init_ms = phase.elapsed_ms();

  const unsigned tiles_x = (width + tile_size - 1) / tile_size;
  const unsigned tiles_y = (height + tile_size - 1) / tile_size;
  const unsigned task_count = tiles_x * tiles_y;
  std::vector<ImageArgs> args(task_count);
  std::vector<starpu_data_handle_t> handles(task_count);
  std::vector<std::vector<std::uint8_t>> tile_storage(task_count);

  phase.start();
  unsigned idx = 0;
  for (unsigned ty = 0; ty < tiles_y; ++ty) {
    for (unsigned tx = 0; tx < tiles_x; ++tx) {
      const unsigned tw = std::min(tile_size, width - tx * tile_size);
      const unsigned th = std::min(tile_size, height - ty * tile_size);
      tile_storage[idx].resize(static_cast<std::size_t>(tw) * th * 3);
      auto &local = tile_storage[idx];

      for (unsigned y = 0; y < th; ++y) {
        const unsigned src_y = ty * tile_size + y;
        const unsigned dst_off = y * tw * 3;
        const unsigned src_off = (src_y * width + tx * tile_size) * 3;
        std::memcpy(local.data() + dst_off, pixels.data() + src_off, tw * 3);
      }

      args[idx] = ImageArgs{tw, th, tx, ty, mixed_ops ? tile_mixed_op(idx) : op};
      starpu_vector_data_register(&handles[idx], STARPU_MAIN_RAM,
                                  reinterpret_cast<uintptr_t>(local.data()), local.size(),
                                  sizeof(std::uint8_t));
      ++idx;
    }
  }
  timings->data_registration_ms = phase.elapsed_ms();

  phase.start();
  idx = 0;
  for (unsigned ty = 0; ty < tiles_y; ++ty) {
    for (unsigned tx = 0; tx < tiles_x; ++tx) {
      struct starpu_task *task = starpu_task_create();
      task->cl = &image_tile_cl;
      task->handles[0] = handles[idx];
      task->cl_arg = &args[idx];
      task->cl_arg_size = sizeof(ImageArgs);
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
  idx = 0;
  for (unsigned ty = 0; ty < tiles_y; ++ty) {
    for (unsigned tx = 0; tx < tiles_x; ++tx) {
      const unsigned tw = args[idx].width;
      const unsigned th = args[idx].height;
      const auto &local = tile_storage[idx];
      for (unsigned y = 0; y < th; ++y) {
        const unsigned dst_y = ty * tile_size + y;
        const unsigned dst_off = (dst_y * width + tx * tile_size) * 3;
        const unsigned src_off = y * tw * 3;
        std::memcpy(pixels.data() + dst_off, local.data() + src_off, tw * 3);
      }
      starpu_data_unregister(handles[idx]);
      ++idx;
    }
  }

  shutdown_starpu();
  timings->data_unregister_ms = phase.elapsed_ms();
  return task_count;
}

static void print_usage(const char *prog)
{
  std::fprintf(stderr,
               "Usage: %s --mode MODE --width W --height H --op OP [--tile-size N] "
               "[--mixed-ops] [--output PATH]\n"
               "Ops: grayscale blur edge convolution filter\n"
               "Modes: native_cpu native_gpu starpu_hybrid\n",
               prog);
}

int main(int argc, char **argv)
{
  ExecutionMode mode = ExecutionMode::StarpuHybrid;
  unsigned width = 1280;
  unsigned height = 720;
  unsigned tile_size = 64;
  ImageOp op = ImageOp::Grayscale;
  bool mixed_ops = false;
  std::string output_path;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
      mode = parse_execution_mode(argv[++i]);
    else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc)
      width = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc)
      height = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--tile-size") == 0 && i + 1 < argc)
      tile_size = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--op") == 0 && i + 1 < argc)
      op = parse_image_op(argv[++i]);
    else if (std::strcmp(argv[i], "--mixed-ops") == 0)
      mixed_ops = true;
    else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
      output_path = argv[++i];
    else if (std::strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }

  if (width == 0 || height == 0 || tile_size == 0) {
    std::fprintf(stderr, "width, height, and tile-size must be greater than zero\n");
    return 1;
  }

  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * 3);
  std::mt19937 rng(99);
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto &p : pixels)
    p = static_cast<std::uint8_t>(dist(rng));

  ChronoTimer timer;
  StarpuTimings starpu_timings;
  unsigned task_count = 1;
  timer.start();

  if (mode == ExecutionMode::NativeCpu) {
    task_count = process_image_tiles_native_cpu(pixels, width, height, tile_size, op, mixed_ops);
  } else if (mode == ExecutionMode::NativeGpu) {
    task_count = process_image_tiles_native_gpu(pixels, width, height, tile_size, op, mixed_ops);
  } else {
    task_count = run_starpu_image(pixels, width, height, op, tile_size, mode, mixed_ops,
                                  &starpu_timings);
  }

  const double total_ms = timer.elapsed_ms();

  MetricsWriter writer;
  writer.set_scenario("image");
  writer.set_mode(execution_mode_name(mode));
  writer.set_param("width", std::to_string(width));
  writer.set_param("height", std::to_string(height));
  writer.set_param("operation", image_op_name(op));
  writer.set_param("mixed_ops", mixed_ops ? "true" : "false");
  writer.set_param("tile_size", std::to_string(tile_size));
  writer.set_param("tile_border_policy", "per_tile_simplified_borders_no_halo");
  writer.set_metric("total_time_ms", total_ms);
  writer.set_metric("megapixels", (width * height) / 1e6);
  writer.set_metric("task_count", static_cast<double>(task_count));
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
    output_path = default_result_path("image", execution_mode_name(mode));
  writer.write_json(output_path);

  std::printf("bench_image: %ux%u tile=%u op=%s mode=%s time=%.3f ms -> %s\n", width, height,
              tile_size, image_op_name(op), execution_mode_name(mode), total_ms,
              output_path.c_str());
  return 0;
}
