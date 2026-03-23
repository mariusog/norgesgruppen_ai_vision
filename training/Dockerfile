# Training container for NM i AI 2026 — NorgesGruppen grocery detection
# Pushed to: YOUR_DOCKER_REGISTRY/trainer:latest
#
# Build and push:
#   gcloud builds submit --config=cloudbuild.yaml --project=YOUR_GCP_PROJECT_ID
#
# Run locally (requires GPU):
#   docker run --gpus all -v $(pwd):/workspace \
#     YOUR_DOCKER_REGISTRY/trainer:latest \
#     python training/train.py --data training/data.yaml --epochs 10 --no-upload

# Ultralytics base image — includes ultralytics 8.1.0, PyTorch, CUDA, Python 3.11
FROM ultralytics/ultralytics:8.1.0

WORKDIR /workspace

# Copy project files
COPY pyproject.toml ./
COPY run.py ./
COPY src/ src/
COPY training/ training/
COPY scripts/ scripts/

# Fix numba/coverage conflict: uninstall numba entirely (not needed for our code)
RUN pip uninstall -y numba llvmlite 2>/dev/null; true

# Install what's NOT in the base image
RUN pip install --no-cache-dir \
    "google-cloud-storage==2.14.0" \
    "timm==0.9.12" \
    "ensemble-boxes==1.0.9"

# Default command: run training with standard config
CMD ["python", "training/train.py", "--data", "training/data.yaml"]
