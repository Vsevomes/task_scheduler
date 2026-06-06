# Uses locally cached nvidia/cuda:12.4.0-base-ubuntu22.04 when devel pull fails.
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    libopenblas-dev \
    libhwloc-dev \
    libnuma-dev \
    sysstat \
    cuda-nvcc-12-4 \
    cuda-cudart-dev-12-4 \
    cuda-nvml-dev-12-4 \
    libcublas-dev-12-4 \
    libcusparse-dev-12-4 \
    libstarpu-contrib-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]
