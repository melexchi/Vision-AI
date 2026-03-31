# Ditto Avatar GPU Inference Server — RunPod Deployment
#
# Base: NVIDIA CUDA 12.8 + cuDNN + Ubuntu 22.04
# Includes: Python 3.10, TensorRT 8.6, PyTorch 2.8, FastAPI
#
# Build:
#   docker build -t ditto-avatar .
#
# Run:
#   docker run --gpus all -p 8181:8181 \
#     -v /workspace/checkpoints:/workspace/ditto-talkinghead/checkpoints \
#     ditto-avatar

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv \
    ffmpeg libsndfile1 libgl1-mesa-glx libglib2.0-0 \
    build-essential git wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/bin/uv

WORKDIR /app

# Install Python deps first (cache-friendly layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy Ditto engine + API code
COPY ditto-talkinghead/ ./ditto-talkinghead/
COPY ditto_api.py .
COPY prerender_clips.py .

# Build Cython blend extension
RUN cd ditto-talkinghead/core/utils/blend && \
    uv run python -c "import pyximport; pyximport.install(); import blend" 2>/dev/null || true

# Environment defaults — override at runtime via docker run -e or RunPod env vars
ENV DITTO_PATH=/app/ditto-talkinghead
ENV AVATAR_CACHE_DIR=/workspace/avatar_cache
ENV AVATAR_CLIPS_DIR=/workspace/avatar_clips
ENV AVATAR_IMAGES_DIR=/workspace/avatar_images
ENV SKYREELS_PATH=/workspace/SkyReels-A1
ENV PRERENDER_SCRIPT=/app/prerender_clips.py
ENV TTS_URL=http://localhost:8282/tts/stream
ENV DITTO_BACKEND=trt

EXPOSE 8181

CMD ["uv", "run", "ditto_api.py"]
