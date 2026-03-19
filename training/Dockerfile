# Training container for NM i AI 2026 — NorgesGruppen grocery detection
# Pushed to: us-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest
#
# Build and push:
#   gcloud builds submit --tag us-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest training/
#
# Run locally (requires GPU):
#   docker run --gpus all -v $(pwd):/workspace \
#     us-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest \
#     python training/train.py --data training/data.yaml --epochs 10 --no-upload

FROM ultralytics/ultralytics:8.1.0

WORKDIR /workspace

# Install GCS client for dataset download and weight upload
RUN pip install --no-cache-dir google-cloud-storage==2.14.0

# Copy project files
COPY pyproject.toml ./
COPY src/ src/
COPY training/ training/

# Install project in editable mode (for src.constants imports)
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir -e .

# Default command: run training with standard config
CMD ["python", "training/train.py", "--data", "training/data.yaml"]
