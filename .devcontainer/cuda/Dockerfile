FROM mcr.microsoft.com/devcontainers/python:1-3.10-bookworm as devcontainer-cuda

# Install missing dependencies
RUN apt-get update
RUN apt-get install -y \
    git \
    pkg-config \
    libhdf5-dev \
    python3-pip \
    python3-setuptools \
    python3-venv
RUN python3 -m pip install -U pip

# Set up poetry
RUN pip install poetry
ENV PYTHONPATH="$PYTHONPATH:$PWD"

# ADDITIONAL: Set up NVIDA GPU support
# (Adapted from https://gitlab.com/nvidia/container-images/cuda/-/blob/a819e795/dist/12.2.2/ubuntu2204/base/Dockerfile)

# !!! IMPORATANT !!!
# UNCOMMENT THE ENVIRONMENT BELOW APPROPRIATELY!
#   For amd64
ENV NVARCH x86_64
ENV NVIDIA_REQUIRE_CUDA "cuda>=12.2 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 brand=titan,driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526"
ENV NV_CUDA_CUDART_VERSION 12.2.140-1
ENV NV_CUDA_COMPAT_PACKAGE cuda-compat-12-2

#   For arm64
# ENV NVARCH sbsa
# ENV NVIDIA_REQUIRE_CUDA "cuda>=12.2"
# ENV NV_CUDA_CUDART_VERSION 12.2.140-1

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${NVARCH}/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 12.2.2 

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-12-2=${NV_CUDA_CUDART_VERSION} \
    ${NV_CUDA_COMPAT_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
