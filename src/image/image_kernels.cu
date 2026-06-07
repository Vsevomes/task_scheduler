#include "image_common.hpp"
#include "starpu_ptr.hpp"

#include <cuda_runtime.h>
#include <starpu.h>

__device__ static inline std::uint8_t clamp_u8(int value)
{
  return static_cast<std::uint8_t>(value < 0 ? 0 : (value > 255 ? 255 : value));
}

__global__ void image_kernel(const std::uint8_t *input, std::uint8_t *output, unsigned width,
                             unsigned height, ImageOp op)
{
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  const unsigned idx = (y * width + x) * 3;
  if (op == ImageOp::Grayscale) {
    const std::uint8_t gray =
        static_cast<std::uint8_t>((input[idx] + input[idx + 1] + input[idx + 2]) / 3);
    output[idx] = output[idx + 1] = output[idx + 2] = gray;
    return;
  }

  if (x == 0 || y == 0 || x + 1 >= width || y + 1 >= height) {
    output[idx] = input[idx];
    output[idx + 1] = input[idx + 1];
    output[idx + 2] = input[idx + 2];
    return;
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

void image_native_gpu(const std::uint8_t *input_host, std::uint8_t *output_host, unsigned width,
                      unsigned height, ImageOp op)
{
  const std::size_t bytes = static_cast<std::size_t>(width) * height * 3;
  std::uint8_t *input = nullptr;
  std::uint8_t *output = nullptr;
  cudaMalloc(&input, bytes);
  cudaMalloc(&output, bytes);
  cudaMemcpy(input, input_host, bytes, cudaMemcpyHostToDevice);

  const dim3 threads(16, 16);
  const dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  image_kernel<<<blocks, threads>>>(input, output, width, height, op);
  cudaDeviceSynchronize();

  cudaMemcpy(output_host, output, bytes, cudaMemcpyDeviceToHost);
  cudaFree(input);
  cudaFree(output);
}

extern "C" void cuda_image_tile_codelet(void *buffers[], void *cl_arg)
{
  const auto *args = static_cast<const ImageArgs *>(cl_arg);
  std::uint8_t *tile = starpu_vector_ptr<std::uint8_t>(buffers[0]);
  std::uint8_t *scratch = nullptr;
  const std::size_t bytes = static_cast<std::size_t>(args->width) * args->height * 3;
  cudaMalloc(&scratch, bytes);
  cudaMemcpy(scratch, tile, bytes, cudaMemcpyDeviceToDevice);

  const dim3 threads(16, 16);
  const dim3 blocks((args->width + threads.x - 1) / threads.x,
                    (args->height + threads.y - 1) / threads.y);
  image_kernel<<<blocks, threads>>>(scratch, tile, args->width, args->height, args->op);
  cudaDeviceSynchronize();
  cudaFree(scratch);
}
