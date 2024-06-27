FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye as devcontainer

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
