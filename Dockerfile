# =============================================
# Dockerfile — LocalLLM (Qwen2.5-7B Instruct server)
# =============================================
# GPU-enabled image using CUDA 12.1 runtime on Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System deps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

WORKDIR /app

# Copy only requirements first (for better Docker layer caching)
COPY requirements.txt /app/requirements.txt

# Install Python deps (torch CUDA wheels via index URL)
# If you need to pin versions, edit requirements.txt accordingly.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && python3 -c "import torch; print('Torch CUDA:', torch.cuda.is_available())"

# Copy server code
COPY qwen_server.py /app/qwen_server.py

# Create a models directory inside the container (we'll mount host models here)
RUN mkdir -p /models
ENV MODEL_PATH=/models/qwen2.5-7b-instruct
ENV LOAD_4BIT=1

EXPOSE 8000
USER ${USERNAME}

# Default command (overridable): start FastAPI server
CMD ["python3", "-m", "uvicorn", "qwen_server:app", "--host", "0.0.0.0", "--port", "8000"]