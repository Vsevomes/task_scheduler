FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ARG STARPU_VERSION=1.4.9
ARG STARPU_PREFIX=/opt/starpu

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV STARPU_HOME=${STARPU_PREFIX}

ENV PATH=${CUDA_HOME}/bin:${STARPU_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${STARPU_HOME}/lib
ENV PKG_CONFIG_PATH=${STARPU_HOME}/lib/pkgconfig

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    git \
    wget \
    pkg-config \
    autoconf \
    automake \
    libtool \
    hwloc \
    libhwloc-dev \
    libopenblas-dev \
    libnuma-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

RUN wget https://files.inria.fr/starpu/starpu-${STARPU_VERSION}/starpu-${STARPU_VERSION}.tar.gz && \
    tar -xzf starpu-${STARPU_VERSION}.tar.gz && \
    cd starpu-${STARPU_VERSION} && \
    ./configure \
        --prefix=${STARPU_PREFIX} \
        --enable-cuda \
        --disable-opencl \
        --disable-build-doc \
        CUDA_HOME=${CUDA_HOME} && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/starpu-${STARPU_VERSION}*

WORKDIR /workspace

CMD ["/bin/bash"]