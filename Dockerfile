# ObserveGuard Docker Image
# Reproducible environment for experiments
# Build: docker build -t observeguard:latest .
# Run: docker run -it --rm -v $(pwd)/results:/workspace/results observeguard:latest

FROM ubuntu:22.04

LABEL maintainer="ObserveGuard Team"
LABEL description="ObserveGuard: Observation-Centric Secure Multimodal Agents"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3.10 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    wget \
    curl \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libopenblas-dev \
    libomp-dev \
    libhdf5-dev \
    libharfbuzz0b \
    libwebp6 \
    libtiff5 \
    libopenjp2-7 \
    libjasper1 \
    libjbig2dec0 \
    libz3-4 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU for reproducibility across platforms)
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install -r requirements.txt \
    --no-cache-dir

# Create necessary directories
RUN mkdir -p data/{osworld,ssv2,probes} \
    && mkdir -p results logs models/.cache \
    && chmod -R 755 /workspace

# Verify installation
RUN python3 -c "import torch; print(f'PyTorch {torch.__version__}')" && \
    python3 -c "import agents; print('Agents module OK')" && \
    python3 -c "import evaluation; print('Evaluation module OK')"

# Set default command
ENTRYPOINT ["python3"]
CMD ["--version"]

# Usage examples in comments:
# Interactive shell:
#   docker run -it --rm -v $(pwd)/results:/workspace/results observeguard:latest /bin/bash
#
# Run OSWorld evaluation:
#   docker run -it --rm -v $(pwd)/results:/workspace/results \
#     observeguard:latest evaluation/run_osworld.py --agent observe_guard
#
# Run with GPU (if available):
#   docker run -it --rm --gpus all -v $(pwd)/results:/workspace/results \
#     observeguard:latest evaluation/run_osworld.py --agent observe_guard
