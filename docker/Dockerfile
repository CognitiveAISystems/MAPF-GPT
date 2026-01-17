FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-venv build-essential curl cmake g++ wget \
    ffmpeg libsm6 libxext6 libboost-all-dev vim && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# copy uv binaries
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PATH="/root/.local/bin:$PATH"
ENV UV_SYSTEM_PYTHON=1

# install python dependencies
COPY requirements.txt /tmp/
RUN uv pip install -U pip && uv pip install -r /tmp/requirements.txt
