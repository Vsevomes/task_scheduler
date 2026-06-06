#!/usr/bin/env bash
set -e

echo "Updating packages..."

sudo apt update

echo "Installing NVIDIA driver..."

sudo apt install -y \
    ubuntu-drivers-common

sudo ubuntu-drivers autoinstall

echo "Installing CUDA runtime..."

wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    -O /tmp/cuda-keyring.deb

sudo dpkg -i /tmp/cuda-keyring.deb

sudo apt update

sudo apt install -y \
    cuda-runtime-12-4

echo "Installing StarPU runtime..."

sudo apt install -y \
    libstarpu-1.4-9 \
    libhwloc15 \
    libopenblas0

echo ""
echo "Done."
echo ""

echo "Checking NVIDIA driver:"
nvidia-smi || true

echo ""
echo "Checking CUDA runtime:"
ldconfig -p | grep libcudart || true

echo ""
echo "Checking StarPU:"
ldconfig -p | grep libstarpu || true
