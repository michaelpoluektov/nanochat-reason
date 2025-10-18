FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Base system deps, Python toolchain, and useful utilities for training
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    python-is-python3 \
    build-essential \
    git \
    curl \
    unzip \
    screen \
    rsync \
    pkg-config \
    libssl-dev \
    libffi-dev \
    ca-certificates \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install uv (Python package/dependency manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install Rust toolchain ahead of tokenizer build
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . "$HOME/.cargo/env" && \
    rustup component add rustfmt
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace

# Copy project files into the image
COPY . .

# Create project venv, install Python deps, and pre-build the Rust tokenizer
RUN set -eux; \
    uv venv --python=python3; \
    uv sync --frozen; \
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Make project virtualenv active by default
ENV PATH="/workspace/.venv/bin:${PATH}"
ENV PYTHONPATH="/workspace"
ENV UV_PROJECT_ENVIRONMENT="/workspace/.venv"

CMD ["/bin/bash"]
