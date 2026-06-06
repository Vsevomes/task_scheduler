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

static void cpu_image_tile(void *buffers[], void *cl_arg)
{
  const auto *args = static_cast<const ImageArgs *>(cl_arg);
  auto *tile = starpu_vector_ptr<std::uint8_t>(buffers[0]);
  const unsigned tile_w = args->width;
  const unsigned tile_h = args->height;

  if (args->op == ImageOp::Grayscale) {
    for (unsigned y = 0; y < tile_h; ++y) {
      for (unsigned x = 0; x < tile_w; ++x) {
        const unsigned idx = (y * tile_w + x) * 3;
        const std::uint8_t gray =
            static_cast<std::uint8_t>((tile[idx] + tile[idx + 1] + tile[idx + 2]) / 3);
        tile[idx] = tile[idx + 1] = tile[idx + 2] = gray;
      }
    }
  } else if (args->op == ImageOp::Threshold) {
    for (unsigned y = 0; y < tile_h; ++y) {
      for (unsigned x = 0; x < tile_w; ++x) {
        const unsigned idx = (y * tile_w + x) * 3;
        const std::uint8_t gray =
            static_cast<std::uint8_t>((tile[idx] + tile[idx + 1] + tile[idx + 2]) / 3);
        const std::uint8_t v = gray > 128 ? 255 : 0;
        tile[idx] = tile[idx + 1] = tile[idx + 2] = v;
      }
    }
  } else {
    std::vector<std::uint8_t> copy(tile_w * tile_h * 3);
    std::memcpy(copy.data(), tile, copy.size());
    for (unsigned y = 1; y + 1 < tile_h; ++y) {
      for (unsigned x = 1; x + 1 < tile_w; ++x) {
        for (unsigned c = 0; c < 3; ++c) {
          unsigned sum = 0;
          for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
              const unsigned idx = ((y + dy) * tile_w + (x + dx)) * 3 + c;
              sum += copy[idx];
            }
          }
          tile[(y * tile_w + x) * 3 + c] = static_cast<std::uint8_t>(sum / 9);
        }
      }
    }
  }
}

static struct starpu_codelet image_tile_cl = {
    .where = STARPU_CPU | STARPU_CUDA,
    .cpu_funcs = {cpu_image_tile},
    .cuda_funcs = {cuda_image_tile_codelet},
    .nbuffers = 1,
    .modes = {STARPU_RW},
    .name = "image_tile",
};

static ImageOp parse_image_op(const char *value)
{
  if (std::strcmp(value, "grayscale") == 0)
    return ImageOp::Grayscale;
  if (std::strcmp(value, "blur") == 0)
    return ImageOp::Blur;
  return ImageOp::Threshold;
}

static const char *image_op_name(ImageOp op)
{
  switch (op) {
  case ImageOp::Grayscale:
    return "grayscale";
  case ImageOp::Blur:
    return "blur";
  case ImageOp::Threshold:
    return "threshold";
  }
  return "unknown";
}

static void run_starpu_image(std::vector<std::uint8_t> &pixels, unsigned width, unsigned height,
                             ImageOp op, ExecutionMode mode)
{
  if (init_starpu_for_mode(mode) != 0)
    std::exit(1);

  const unsigned tile = 64;
  const unsigned tiles_x = (width + tile - 1) / tile;
  const unsigned tiles_y = (height + tile - 1) / tile;
  std::vector<ImageArgs> args;
  std::vector<starpu_data_handle_t> handles;
  std::vector<std::vector<std::uint8_t>> tile_storage;

  for (unsigned ty = 0; ty < tiles_y; ++ty) {
    for (unsigned tx = 0; tx < tiles_x; ++tx) {
      const unsigned tw = std::min(tile, width - tx * tile);
      const unsigned th = std::min(tile, height - ty * tile);
      tile_storage.emplace_back(tw * th * 3);
      auto &local = tile_storage.back();

      for (unsigned y = 0; y < th; ++y) {
        const unsigned src_y = ty * tile + y;
        const unsigned dst_off = y * tw * 3;
        const unsigned src_off = (src_y * width + tx * tile) * 3;
        std::memcpy(local.data() + dst_off, pixels.data() + src_off, tw * 3);
      }

      ImageArgs arg{tw, th, tx, ty, op};
      args.push_back(arg);

      starpu_data_handle_t handle;
      starpu_vector_data_register(&handle, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(local.data()),
                                  local.size(), sizeof(std::uint8_t));
      handles.push_back(handle);

      struct starpu_task *task = starpu_task_create();
      task->cl = &image_tile_cl;
      task->handles[0] = handle;
      task->cl_arg = &args.back();
      task->cl_arg_size = sizeof(ImageArgs);
      task->destroy = 0;
      starpu_task_submit(task);
    }
  }

  starpu_task_wait_for_all();

  size_t idx = 0;
  for (unsigned ty = 0; ty < tiles_y; ++ty) {
    for (unsigned tx = 0; tx < tiles_x; ++tx) {
      const unsigned tw = args[idx].width;
      const unsigned th = args[idx].height;
      const auto &local = tile_storage[idx];
      for (unsigned y = 0; y < th; ++y) {
        const unsigned dst_y = ty * tile + y;
        const unsigned dst_off = (dst_y * width + tx * tile) * 3;
        const unsigned src_off = y * tw * 3;
        std::memcpy(pixels.data() + dst_off, local.data() + src_off, tw * 3);
      }
      starpu_data_unregister(handles[idx]);
      ++idx;
    }
  }

  shutdown_starpu();
}

static void print_usage(const char *prog)
{
  std::fprintf(stderr,
               "Usage: %s --mode MODE --width W --height H --op OP [--output PATH]\n"
               "Ops: grayscale blur threshold\n",
               prog);
}

int main(int argc, char **argv)
{
  ExecutionMode mode = ExecutionMode::StarpuHybrid;
  unsigned width = 1280;
  unsigned height = 720;
  ImageOp op = ImageOp::Grayscale;
  std::string output;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
      mode = parse_execution_mode(argv[++i]);
    else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc)
      width = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc)
      height = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
    else if (std::strcmp(argv[i], "--op") == 0 && i + 1 < argc)
      op = parse_image_op(argv[++i]);
    else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
      output = argv[++i];
    else if (std::strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }

  std::vector<std::uint8_t> pixels(width * height * 3);
  std::mt19937 rng(99);
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto &p : pixels)
    p = static_cast<std::uint8_t>(dist(rng));

  ChronoTimer timer;
  timer.start();

  if (mode == ExecutionMode::NativeCpu) {
    ImageArgs args{width, height, 0, 0, op};
    void *buffers[1] = {pixels.data()};
    cpu_image_tile(buffers, &args);
  } else if (mode == ExecutionMode::NativeGpu) {
    std::vector<std::uint8_t> out(pixels.size());
    image_native_gpu(pixels.data(), out.data(), width, height, op);
    pixels.swap(out);
  } else {
    run_starpu_image(pixels, width, height, op, mode);
  }

  const double total_ms = timer.elapsed_ms();

  MetricsWriter writer;
  writer.set_scenario("image");
  writer.set_mode(execution_mode_name(mode));
  writer.set_param("width", std::to_string(width));
  writer.set_param("height", std::to_string(height));
  writer.set_param("operation", image_op_name(op));
  writer.set_metric("total_time_ms", total_ms);
  writer.set_metric("megapixels", (width * height) / 1e6);
  if (output.empty())
    output = default_result_path("image", execution_mode_name(mode));
  writer.write_json(output);

  std::printf("bench_image: %ux%u op=%s mode=%s time=%.3f ms -> %s\n", width, height,
              image_op_name(op), execution_mode_name(mode), total_ms, output.c_str());
  return 0;
}
