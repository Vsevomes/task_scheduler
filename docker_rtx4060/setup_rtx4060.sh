#!/usr/bin/env bash
set -e

STARPU_ARCHIVE="starpu-runtime-rtx4060.tar.gz"
STARPU_INSTALL_DIR="/opt"

if [ ! -f "$STARPU_ARCHIVE" ]; then
    echo "Ошибка: рядом со скриптом нет $STARPU_ARCHIVE"
    echo "Сначала перенеси архив StarPU runtime с build-хоста."
    exit 1
fi

echo "Updating packages..."
sudo apt update

echo "Installing basic runtime dependencies..."
sudo apt install -y \
    wget \
    ca-certificates \
    libhwloc15 \
    libopenblas0 \
    libnuma1 \
    libgfortran5

echo "Installing CUDA runtime repository..."
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    -O /tmp/cuda-keyring.deb

sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt update

echo "Installing CUDA runtime..."
sudo apt install -y cuda-runtime-12-4

echo "Installing StarPU runtime from build Docker image..."
sudo tar -xzf "$STARPU_ARCHIVE" -C "$STARPU_INSTALL_DIR"

echo "Registering StarPU library path..."
echo "/opt/starpu/lib" | sudo tee /etc/ld.so.conf.d/starpu-rtx4060.conf > /dev/null
sudo ldconfig

echo ""
echo "Checking libraries..."
ldconfig -p | grep libstarpu || true
ldconfig -p | grep libcudart || true

echo ""
echo "Done."
echo "Now try:"
echo "  ldd ./starpu_cuda_test | grep starpu"
echo "  ./starpu_cuda_test"
