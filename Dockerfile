# ============================================================
# Stage 1: base — shared foundation for dev and prod
# ============================================================
FROM python:3.11-slim AS base

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Work around CDN hash mismatch issues with apt retries
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::http::Pipeline-Depth "0";' >> /etc/apt/apt.conf.d/80-retries

# Install system dependencies
RUN rm -rf /var/lib/apt/lists/* \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    # System libs for numpy, pandas, matplotlib
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    fontconfig \
    # For yfinance / requests (SSL)
    ca-certificates \
    # For font download
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Iansui (芫荽) Traditional Chinese font
# https://github.com/ButTaiwan/iansui
RUN mkdir -p /usr/share/fonts/truetype/iansui \
    && curl -fsSL -o /tmp/iansui.zip \
       https://github.com/ButTaiwan/iansui/releases/download/v1.020/iansui.zip \
    && unzip -o /tmp/iansui.zip -d /tmp/iansui \
    && find /tmp/iansui -name '*.ttf' -exec cp {} /usr/share/fonts/truetype/iansui/ \; \
    && fc-cache -fv \
    && rm -rf /tmp/iansui /tmp/iansui.zip

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /uvx /bin/

WORKDIR /app

# Environment variables
# - torch PyPI wheel bundles CUDA 12.x (nvidia-cublas-cu12 etc.)
# - NVIDIA Container Toolkit injects libcuda.so at runtime
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=python3.11 \
    HOME=/app \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/hub \
    TORCH_HOME=/app/.cache/torch \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1

# Create necessary directories (world-writable for non-root runtime)
RUN mkdir -p data models results logs .cache/huggingface .cache/matplotlib \
    && chmod -R 777 data models results logs .cache

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock .python-version ./

# ============================================================
# Stage 2: prod — minimal production image
# ============================================================
FROM base AS prod

# Install production dependencies (including optional extras for all model backends)
RUN uv sync --frozen --no-dev --all-extras --no-install-project

# Copy application source code
COPY src/ ./src/
COPY main.py ./

# Install the project itself
RUN uv sync --frozen --no-dev --all-extras

# Runtime: skip uv sync (env is fully set up at build time) and use writable cache dir
ENV UV_NO_SYNC=true \
    UV_CACHE_DIR=/tmp/uv-cache

# Copy config as fallback
COPY config.json ./config.json

CMD ["uv", "run", "main.py"]

# ============================================================
# Stage 3: dev — full development image
# ============================================================
FROM base AS dev

# Install ALL dependencies including dev groups and optional extras
RUN uv sync --frozen --all-extras --all-groups --no-install-project

# Copy application source code
COPY src/ ./src/
COPY main.py ./

# Install the project itself
RUN uv sync --frozen --all-extras --all-groups

# Make project's editable install writable for non-root uv run
RUN chmod 777 /app/.venv/lib/python3.11/site-packages/ \
    && chmod -R 777 /app/.venv/lib/python3.11/site-packages/__editable__* \
                     /app/.venv/lib/python3.11/site-packages/currency_predict_attempt* \
                     /app/src/currency_predict_attempt.egg-info 2>/dev/null || true

# Redirect uv cache to /tmp at runtime (build cache stays in image, runtime uses writable /tmp)
ENV UV_CACHE_DIR=/tmp/uv-cache

# Copy additional dev files
COPY tests/ ./tests/
COPY examples/ ./examples/
COPY notebooks/ ./notebooks/
COPY config.json config_example.json ./

EXPOSE 8888

CMD ["uv", "run", "pytest", "-v"]
