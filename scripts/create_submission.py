#!/usr/bin/env python3
"""Build a ready-to-upload submission zip for NM i AI 2026.

This is a build/prep script -- NOT included in the submission itself.
It may freely use os, shutil, zipfile, etc.

Usage:
    python scripts/create_submission.py

Creates submission.zip at the project root containing:
    run.py                          (at zip root)
    src/__init__.py
    src/constants.py
    weights/yolov8l-1280-aug.pt     (from ENSEMBLE_WEIGHTS)
    weights/yolov8l-640-aug.pt
    weights/yolov8m-640-aug.pt
    weights/classifier.pt           (if present)

Idempotent: overwrites any existing submission.zip.
"""

import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = REPO_ROOT / "submission.zip"

# Files to include at the root of the zip (relative to repo root)
REQUIRED_FILES: list[str] = [
    "run.py",
    "src/__init__.py",
    "src/constants.py",
    "src/prototype_matcher.py",
]

# Read ensemble weights dynamically from constants.py
sys.path.insert(0, str(REPO_ROOT))
from src.constants import ENSEMBLE_WEIGHTS as ENSEMBLE_WEIGHT_FILES  # noqa: E402

# Optional classifier weight (only include if classifier is enabled and not bundled)
from src.constants import BUNDLE_WEIGHT_PATH, USE_CLASSIFIER  # noqa: E402

CLASSIFIER_PATH = ""
if USE_CLASSIFIER and not BUNDLE_WEIGHT_PATH:
    CLASSIFIER_PATH = "weights/classifier.pt"

# Constraints
MAX_ZIP_SIZE_MB = 420
MAX_WEIGHT_FILES = 3  # ensemble only; classifier is a special case

# Explicitly excluded files (even if they match weight patterns)
EXCLUDED_WEIGHTS = {
    "weights/model.pt",  # duplicate of ensemble member
    "weights/yolov8x-640-aug.pt",  # 4th model, would exceed limit
}


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    included: list[tuple[str, float]] = []  # (arcname, size_mb)

    # ------------------------------------------------------------------
    # 1. Validate required files exist
    # ------------------------------------------------------------------
    all_files: list[tuple[Path, str]] = []  # (abs_path, arcname_in_zip)

    for rel in REQUIRED_FILES:
        full = REPO_ROOT / rel
        if not full.is_file():
            errors.append(f"Required file missing: {rel}")
        else:
            all_files.append((full, rel))

    # ------------------------------------------------------------------
    # 2. Collect weight files
    # ------------------------------------------------------------------
    weight_files_added = 0

    for rel in ENSEMBLE_WEIGHT_FILES:
        full = REPO_ROOT / rel
        if not full.is_file():
            errors.append(f"Ensemble weight missing: {rel}")
        else:
            all_files.append((full, rel))
            weight_files_added += 1

    # Optional classifier
    classifier_full = REPO_ROOT / CLASSIFIER_PATH
    has_classifier = classifier_full.is_file()
    if has_classifier:
        all_files.append((classifier_full, CLASSIFIER_PATH))

    # ------------------------------------------------------------------
    # 3. Pre-flight checks
    # ------------------------------------------------------------------
    if weight_files_added > MAX_WEIGHT_FILES:
        errors.append(f"Too many ensemble weight files: {weight_files_added} > {MAX_WEIGHT_FILES}")

    if has_classifier:
        total_weight_count = weight_files_added + 1
        if total_weight_count > MAX_WEIGHT_FILES + 1:
            warnings.append(
                f"Classifier brings total weight count to {total_weight_count}. "
                "Verify competition allows classifier as a special case."
            )

    if errors:
        print("ERRORS -- cannot build zip:")
        for e in errors:
            print(f"  - {e}")
        return 1

    # ------------------------------------------------------------------
    # 4. Build zip
    # ------------------------------------------------------------------
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    total_size_mb = 0.0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for abs_path, arcname in all_files:
            zf.write(abs_path, arcname)
            size = _size_mb(abs_path)
            total_size_mb += size
            included.append((arcname, size))

    zip_size_mb = _size_mb(OUTPUT_ZIP)

    # ------------------------------------------------------------------
    # 5. Verify zip structure
    # ------------------------------------------------------------------
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as zf:
        names = zf.namelist()

    # run.py must be at zip root (not in a subfolder)
    if "run.py" not in names:
        run_matches = [n for n in names if n.endswith("run.py")]
        errors.append(
            f"run.py is NOT at the zip root! Found: {run_matches or 'nowhere'}. "
            "This will cause submission failure."
        )

    # Size check (uncompressed total of weights)
    if total_size_mb > MAX_ZIP_SIZE_MB:
        errors.append(
            f"Total uncompressed size {total_size_mb:.1f} MB exceeds {MAX_ZIP_SIZE_MB} MB limit"
        )

    # ------------------------------------------------------------------
    # 6. Print summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  NM i AI 2026 — Submission Zip Builder")
    print("=" * 60)
    print()
    print("Files included:")
    for arcname, size in included:
        tag = ""
        if arcname.endswith(".pt"):
            tag = "  [weight]"
        print(f"  {arcname:45s} {size:8.2f} MB{tag}")
    print()
    print(f"  Total uncompressed:  {total_size_mb:8.2f} MB")
    print(f"  Zip file size:       {zip_size_mb:8.2f} MB")
    print(f"  Output:              {OUTPUT_ZIP}")
    print()

    # Zip contents verification
    print("Zip contents (zipfile.namelist):")
    for name in sorted(names):
        print(f"  {name}")
    print()

    if "run.py" in names:
        print("  [OK] run.py is at the zip root")
    else:
        print("  [FAIL] run.py is NOT at the zip root!")

    weight_count = sum(1 for n in names if n.endswith(".pt"))
    print(f"  [OK] Weight files in zip: {weight_count}")

    if total_size_mb <= MAX_ZIP_SIZE_MB:
        print(f"  [OK] Size {total_size_mb:.1f} MB <= {MAX_ZIP_SIZE_MB} MB limit")
    else:
        print(f"  [FAIL] Size {total_size_mb:.1f} MB > {MAX_ZIP_SIZE_MB} MB limit")

    if warnings:
        print()
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print()
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print()
    print("Submission zip is ready for upload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
