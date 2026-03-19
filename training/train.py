"""
Vertex AI training job for YOLOv8m fine-tuning on NorgesGruppen grocery dataset.

Runs inside a Docker container on Vertex AI Custom Training.
Dataset is pulled from GCS at job start.

Usage (local test):
    python training/train.py --data training/data.yaml --epochs 50 --imgsz 640

Usage (Vertex AI):
    Set via gcloud ai custom-jobs create worker pool spec.
    See model-agent.md for the full gcloud command.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

GCS_BUCKET = "ai-nm26osl-1792-nmiai"
WEIGHTS_PREFIX = "weights"
DEFAULT_BASE_MODEL = "yolov8m.pt"
DEFAULT_DATA = "training/data.yaml"
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
OUTPUT_DIR = Path("/workspace/runs")
RESULTS_DIR = Path("docs")


def download_dataset(data_yaml: Path) -> None:
    """Pull dataset from GCS if running on Vertex AI (indicated by /workspace mount)."""
    workspace = Path("/workspace")
    if not workspace.exists():
        print("Skipping GCS download — not running in Vertex AI container")
        return
    data_dir = workspace / "data"
    if data_dir.exists() and any(data_dir.rglob("*.jpg")):
        print(f"Dataset already present at {data_dir}")
        return
    print(f"Pulling dataset from gs://{GCS_BUCKET}/datasets/ → {data_dir}")
    # Use google-cloud-storage SDK (no subprocess allowed in run.py but training runs in container)
    from google.cloud import storage  # type: ignore[import]

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blobs = list(bucket.list_blobs(prefix="datasets/"))
    data_dir.mkdir(parents=True, exist_ok=True)
    for blob in blobs:
        dest = data_dir / Path(blob.name).relative_to("datasets")
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
    print(f"Downloaded {len(blobs)} files to {data_dir}")


def upload_weights(run_dir: Path, run_id: str) -> None:
    """Upload best weights to GCS after training."""
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        print(f"WARNING: best.pt not found at {best_weights}")
        return
    size_mb = best_weights.stat().st_size / 1024 / 1024
    print(f"Uploading {best_weights} ({size_mb:.1f} MB) to GCS...")
    from google.cloud import storage  # type: ignore[import]

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    dest_path = f"{WEIGHTS_PREFIX}/yolov8m_run{run_id}.pt"
    blob = bucket.blob(dest_path)
    blob.upload_from_filename(str(best_weights))
    print(f"Uploaded to gs://{GCS_BUCKET}/{dest_path}")


def log_results(results: object, run_id: str, elapsed: float) -> None:
    """Write training summary to docs/benchmark_results.md."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "benchmark_results.md"

    # Extract mAP from ultralytics results object
    try:
        map50 = float(results.results_dict.get("metrics/mAP50(B)", 0.0))  # type: ignore[union-attr]
        map5095 = float(results.results_dict.get("metrics/mAP50-95(B)", 0.0))  # type: ignore[union-attr]
    except (AttributeError, KeyError):
        map50 = 0.0
        map5095 = 0.0

    header = "| Run | Date | Model | Epochs | mAP@50 | mAP@50:95 | Time (min) | Notes |\n|-----|------|-------|--------|--------|-----------|------------|-------|\n"
    row = f"| {run_id} | {time.strftime('%Y-%m-%d')} | yolov8m | - | {map50:.3f} | {map5095:.3f} | {elapsed/60:.1f} | |\n"

    if not results_file.exists():
        results_file.write_text(f"# Training Benchmark Results\n\n{header}{row}")
    else:
        content = results_file.read_text()
        if "| Run |" not in content:
            results_file.write_text(content + f"\n## Training Runs\n\n{header}{row}")
        else:
            results_file.write_text(content + row)

    print(f"Results logged to {results_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8m training for NorgesGruppen grocery detection")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--run-id", type=str, default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--no-upload", action="store_true", help="Skip GCS upload after training")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    download_dataset(data_yaml)

    model = YOLO(args.model)

    t_start = time.perf_counter()
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        project=str(OUTPUT_DIR),
        name=f"run_{args.run_id}",
        exist_ok=True,
        plots=True,
        save=True,
        save_period=10,
        patience=20,
        cos_lr=True,
        label_smoothing=0.1,
    )
    elapsed = time.perf_counter() - t_start

    run_dir = OUTPUT_DIR / f"run_{args.run_id}"
    log_results(results, args.run_id, elapsed)

    if not args.no_upload:
        upload_weights(run_dir, args.run_id)

    print(f"Training complete in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
