#pragma once

#include <cstddef>
#include <cstdint>

enum class ImageOp { Grayscale, Blur, Edge, Convolution, Filter };

struct ImageArgs {
  unsigned width;
  unsigned height;
  unsigned tile_x;
  unsigned tile_y;
  ImageOp op;
};

extern "C" void cuda_image_tile_codelet(void *buffers[], void *cl_arg);

void image_native_gpu(const std::uint8_t *input, std::uint8_t *output, unsigned width,
                      unsigned height, ImageOp op);
