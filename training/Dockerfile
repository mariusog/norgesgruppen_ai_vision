# Training container for NM i AI 2026 — NorgesGruppen grocery detection
# Pushed to: europe-west4-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest
#
# Build and push:
#   gcloud builds submit --tag europe-west4-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest training/
#
# Run locally (requires GPU):
#   docker run --gpus all -v $(pwd):/workspace \
#     europe-west4-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest \
#     python training/train.py --data training/data.yaml --epochs 10 --no-upload

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /workspace

# Copy project files
COPY pyproject.toml ./
COPY src/ src/
COPY training/ training/

# Install project and all dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Default command: run training with standard config
CMD ["python", "training/train.py", "--data", "training/data.yaml"]
