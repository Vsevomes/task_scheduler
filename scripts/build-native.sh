#!/usr/bin/env bash
# Native build on Ubuntu 22.04+ (RTX 4060 host) when Docker is unavailable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Installing build dependencies (requires sudo)..."
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  pkg-config \
  libopenblas-dev \
  libhwloc-dev \
  libstarpu-contrib-dev \
  cuda-nvcc-12-4 \
  cuda-cudart-dev-12-4 \
  cuda-nvml-dev-12-4 \
  libcublas-dev-12-4 \
  libcusparse-dev-12-4 \
  sysstat

export PATH="/usr/local/cuda/bin:${PATH}"
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
export PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"

echo "nvcc: $(nvcc --version | tail -1)"
echo "starpu: $(pkg-config --modversion starpu-1.3)"

mkdir -p "$ROOT/build"
cd "$ROOT/build"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER="$CUDACXX"
cmake --build . -j"$(nproc)"

echo ""
echo "Build complete:"
ls -1 "$ROOT/build"/bench_* "$ROOT/build"/hello_starpu 2>/dev/null
