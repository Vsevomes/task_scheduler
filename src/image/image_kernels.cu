#include "image_common.hpp"
#include "starpu_ptr.hpp"

#include <cuda_runtime.h>
#include <starpu.h>

__global__ void grayscale_kernel(std::uint8_t *pixels, unsigned count)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const unsigned idx = i * 3;
  const std::uint8_t gray =
      static_cast<std::uint8_t>((pixels[idx] + pixels[idx + 1] + pixels[idx + 2]) / 3);
  pixels[idx] = pixels[idx + 1] = pixels[idx + 2] = gray;
}

__global__ void threshold_kernel(std::uint8_t *pixels, unsigned count)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const unsigned idx = i * 3;
  const std::uint8_t gray =
      static_cast<std::uint8_t>((pixels[idx] + pixels[idx + 1] + pixels[idx + 2]) / 3);
  const std::uint8_t v = gray > 128 ? 255 : 0;
  pixels[idx] = pixels[idx + 1] = pixels[idx + 2] = v;
}

void image_native_gpu(const std::uint8_t *input, std::uint8_t *output, unsigned width,
                      unsigned height, ImageOp op)
{
  const unsigned count = width * height;
  const std::size_t bytes = count * 3;
  std::uint8_t *d_in = nullptr;
  std::uint8_t *d_out = nullptr;
  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_in, input, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, d_in, bytes, cudaMemcpyDeviceToDevice);

  const unsigned blocks = (count + 255) / 256;
  if (op == ImageOp::Grayscale)
    grayscale_kernel<<<blocks, 256>>>(d_out, count);
  else if (op == ImageOp::Threshold)
    threshold_kernel<<<blocks, 256>>>(d_out, count);
  else
    grayscale_kernel<<<blocks, 256>>>(d_out, count);

  cudaDeviceSynchronize();
  cudaMemcpy(output, d_out, bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
}

extern "C" void cuda_image_tile_codelet(void *buffers[], void *cl_arg)
{
  const auto *args = static_cast<const ImageArgs *>(cl_arg);
  std::uint8_t *tile = starpu_vector_ptr<std::uint8_t>(buffers[0]);
  const unsigned count = args->width * args->height;
  const unsigned blocks = (count + 255) / 256;
  if (args->op == ImageOp::Grayscale)
    grayscale_kernel<<<blocks, 256>>>(tile, count);
  else if (args->op == ImageOp::Threshold)
    threshold_kernel<<<blocks, 256>>>(tile, count);
  else
    grayscale_kernel<<<blocks, 256>>>(tile, count);
  cudaDeviceSynchronize();
}
