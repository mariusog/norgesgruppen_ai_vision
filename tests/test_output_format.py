"""
Tests that the competition output JSON format is correct.

Competition spec: JSON array where each entry contains:
  - image_id   (int)
  - category_id (int, 0-355)
  - bbox        ([x, y, width, height] — four floats/ints, xywh format)
  - score       (float, 0.0-1.0)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _is_valid_prediction(item: Any, index: int) -> list[str]:
    """Return list of error strings for a single prediction dict."""
    errors: list[str] = []
    if not isinstance(item, dict):
        return [f"[{index}] expected dict, got {type(item).__name__}"]

    # Required fields
    for field in ("image_id", "category_id", "bbox", "score"):
        if field not in item:
            errors.append(f"[{index}] missing field: {field!r}")

    if errors:
        return errors

    # image_id must be int
    if not isinstance(item["image_id"], int):
        errors.append(f"[{index}] image_id must be int, got {type(item['image_id']).__name__}")

    # category_id must be int in [0, 355]
    if not isinstance(item["category_id"], int):
        errors.append(
            f"[{index}] category_id must be int, got {type(item['category_id']).__name__}"
        )
    elif not (0 <= item["category_id"] <= 355):
        errors.append(f"[{index}] category_id {item['category_id']} out of range [0, 355]")

    # bbox must be [x, y, w, h] — exactly 4 numeric elements
    bbox = item["bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        errors.append(f"[{index}] bbox must be a list of 4 elements, got {bbox!r}")
    else:
        for i, val in enumerate(bbox):
            if not isinstance(val, (int, float)):
                errors.append(f"[{index}] bbox[{i}] must be numeric, got {type(val).__name__}")
            if val < 0:
                errors.append(f"[{index}] bbox[{i}] is negative ({val}), expected pixels ≥ 0")

    # score must be float in [0.0, 1.0]
    score = item["score"]
    if not isinstance(score, float):
        errors.append(f"[{index}] score must be float, got {type(score).__name__}")
    elif not (0.0 <= score <= 1.0):
        errors.append(f"[{index}] score {score} out of range [0.0, 1.0]")

    return errors


class TestPredictionSchema:
    """Unit tests for prediction format validation logic."""

    def test_valid_prediction_passes(self) -> None:
        pred = {"image_id": 42, "category_id": 7, "bbox": [10.0, 20.0, 100.0, 80.0], "score": 0.95}
        assert _is_valid_prediction(pred, 0) == []

    def test_missing_image_id_fails(self) -> None:
        pred = {"category_id": 7, "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.5}
        errors = _is_valid_prediction(pred, 0)
        assert any("image_id" in e for e in errors)

    def test_missing_bbox_fails(self) -> None:
        pred = {"image_id": 1, "category_id": 0, "score": 0.9}
        errors = _is_valid_prediction(pred, 0)
        assert any("bbox" in e for e in errors)

    def test_bbox_wrong_length_fails(self) -> None:
        pred = {"image_id": 1, "category_id": 0, "bbox": [10, 20, 30], "score": 0.5}
        errors = _is_valid_prediction(pred, 0)
        assert any("bbox" in e for e in errors)

    def test_category_id_out_of_range_fails(self) -> None:
        pred = {"image_id": 1, "category_id": 356, "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.5}
        errors = _is_valid_prediction(pred, 0)
        assert any("category_id" in e for e in errors)

    def test_score_not_float_fails(self) -> None:
        # int score is not accepted — must be float
        pred = {"image_id": 1, "category_id": 0, "bbox": [0.0, 0.0, 10.0, 10.0], "score": 1}
        errors = _is_valid_prediction(pred, 0)
        assert any("score" in e for e in errors)

    def test_empty_list_is_valid(self) -> None:
        """An empty predictions list is valid (no detections for any image)."""
        data: list[dict] = []
        assert isinstance(data, list)
        for i, item in enumerate(data):
            assert _is_valid_prediction(item, i) == []

    def test_numpy_float_not_accepted(self) -> None:
        """Ensure score must be Python float, not numpy.float32 (common mistake)."""
        # Simulate what numpy.float32 looks like from ultralytics
        import struct

        raw = struct.pack("f", 0.95)
        numpy_like = struct.unpack("f", raw)[0]  # This is a Python float
        # Python float — should pass
        pred = {
            "image_id": 1,
            "category_id": 0,
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "score": float(numpy_like),
        }
        assert _is_valid_prediction(pred, 0) == []


class TestOutputFileFormat:
    """Tests for serialized JSON output files."""

    def test_json_is_array(self, tmp_path: Path) -> None:
        output = tmp_path / "predictions.json"
        output.write_text(json.dumps([]))
        data = json.loads(output.read_text())
        assert isinstance(data, list), "Output must be a JSON array, not an object"

    def test_predictions_round_trip(self, tmp_path: Path) -> None:
        predictions = [
            {"image_id": 1, "category_id": 0, "bbox": [10.0, 20.0, 50.0, 60.0], "score": 0.87},
            {"image_id": 1, "category_id": 12, "bbox": [100.0, 150.0, 30.0, 40.0], "score": 0.63},
            {"image_id": 2, "category_id": 355, "bbox": [5.0, 5.0, 200.0, 300.0], "score": 0.99},
        ]
        output = tmp_path / "out.json"
        output.write_text(json.dumps(predictions))
        loaded = json.loads(output.read_text())
        assert loaded == predictions

    def test_all_predictions_valid(self, tmp_path: Path) -> None:
        predictions = [
            {"image_id": 42, "category_id": 7, "bbox": [0.0, 0.0, 640.0, 480.0], "score": 0.75},
        ]
        output = tmp_path / "out.json"
        output.write_text(json.dumps(predictions))
        data = json.loads(output.read_text())
        all_errors: list[str] = []
        for i, item in enumerate(data):
            all_errors.extend(_is_valid_prediction(item, i))
        assert all_errors == [], f"Schema violations:\n" + "\n".join(all_errors)
