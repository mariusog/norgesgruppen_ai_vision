"""Tests for run.py helper functions (no model weights needed)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Stub cv2 before importing run.py (requires libGL which isn't in CI)
if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

from run import collect_images, load_model, run_inference


class TestCollectImages:
    def test_finds_jpg_files(self, tmp_path: Path) -> None:
        (tmp_path / "img_00001.jpg").touch()
        (tmp_path / "img_00002.jpeg").touch()
        (tmp_path / "img_00003.png").touch()
        result = collect_images(tmp_path)
        assert len(result) == 3

    def test_deduplicates_case_variants(self, tmp_path: Path) -> None:
        (tmp_path / "img_00001.JPG").touch()
        result = collect_images(tmp_path)
        assert len(result) == 1

    def test_returns_sorted(self, tmp_path: Path) -> None:
        (tmp_path / "img_00003.jpg").touch()
        (tmp_path / "img_00001.jpg").touch()
        (tmp_path / "img_00002.jpg").touch()
        result = collect_images(tmp_path)
        assert result == sorted(result)

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert collect_images(tmp_path) == []

    def test_ignores_non_image_files(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").touch()
        (tmp_path / "data.json").touch()
        assert collect_images(tmp_path) == []


class TestLoadModel:
    def test_load_model_file_not_found(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError when weights file does not exist."""
        with (
            patch("run.MODEL_PATH", str(tmp_path / "nonexistent.pt")),
            patch("run.MODEL_ENGINE_PATH", str(tmp_path / "nonexistent.engine")),
            pytest.raises(FileNotFoundError, match="No model found at"),
        ):
            load_model()


def _make_mock_boxes(xyxy_list: list, cls_list: list, conf_list: list) -> MagicMock:
    """Create a mock Boxes object with .xyxy, .cls, .conf tensors."""
    boxes = []
    for xyxy, cls_val, conf_val in zip(xyxy_list, cls_list, conf_list, strict=True):
        box = MagicMock()
        box.xyxy = torch.tensor([xyxy])
        box.cls = torch.tensor([cls_val])
        box.conf = torch.tensor([conf_val])
        boxes.append(box)

    mock_boxes = MagicMock()
    mock_boxes.__len__ = lambda self: len(boxes)
    mock_boxes.__iter__ = lambda self: iter(boxes)
    mock_boxes.__bool__ = lambda self: len(boxes) > 0
    return mock_boxes


def _make_mock_result(boxes_mock: MagicMock | None) -> MagicMock:
    """Create a mock Result object with a .boxes attribute."""
    result = MagicMock()
    result.boxes = boxes_mock
    return result


class TestRunInference:
    def test_run_inference_with_mock_model(self, tmp_path: Path) -> None:
        """Mock YOLO model returning known boxes, verify output dict format."""
        # Create dummy image files
        img1 = tmp_path / "img_00042.jpg"
        img1.touch()

        # Build mock boxes: one detection at (10, 20, 110, 220), class 7, conf 0.95
        boxes = _make_mock_boxes(
            xyxy_list=[[10.0, 20.0, 110.0, 220.0]],
            cls_list=[7],
            conf_list=[0.95],
        )
        mock_result = _make_mock_result(boxes)

        model = MagicMock()
        model.predict.return_value = [mock_result]

        preds = run_inference(model, [img1])

        assert len(preds) == 1
        p = preds[0]

        # Verify required keys exist
        assert set(p.keys()) == {"image_id", "category_id", "bbox", "score"}

        # Verify types
        assert isinstance(p["image_id"], int)
        assert isinstance(p["category_id"], int)
        assert isinstance(p["score"], float)
        assert isinstance(p["bbox"], list)
        assert len(p["bbox"]) == 4

        # Verify values
        assert p["image_id"] == 42
        assert p["category_id"] == 7
        assert p["score"] == pytest.approx(0.95, abs=1e-5)

        # Verify bbox is [x, y, w, h] converted from xyxy
        x, y, w, h = p["bbox"]
        assert x == pytest.approx(10.0)
        assert y == pytest.approx(20.0)
        assert w == pytest.approx(100.0)  # 110 - 10
        assert h == pytest.approx(200.0)  # 220 - 20

    def test_run_inference_empty_results(self, tmp_path: Path) -> None:
        """Mock model returning no boxes, verify empty list."""
        img1 = tmp_path / "img_00001.jpg"
        img1.touch()

        # Result with no boxes
        mock_result = _make_mock_result(None)

        model = MagicMock()
        model.predict.return_value = [mock_result]

        preds = run_inference(model, [img1])
        assert preds == []

    def test_run_inference_empty_boxes(self, tmp_path: Path) -> None:
        """Mock model returning empty boxes list, verify empty list."""
        img1 = tmp_path / "img_00001.jpg"
        img1.touch()

        empty_boxes = MagicMock()
        empty_boxes.__len__ = lambda self: 0
        mock_result = _make_mock_result(empty_boxes)

        model = MagicMock()
        model.predict.return_value = [mock_result]

        preds = run_inference(model, [img1])
        assert preds == []


class TestImageIdExtraction:
    """Verify image_id extraction from filenames via run_inference."""

    def _run_with_filename(self, tmp_path: Path, filename: str) -> int:
        """Helper: create a file, run inference with one mock detection, return image_id."""
        img = tmp_path / filename
        img.touch()

        boxes = _make_mock_boxes(
            xyxy_list=[[0.0, 0.0, 1.0, 1.0]],
            cls_list=[0],
            conf_list=[0.5],
        )
        mock_result = _make_mock_result(boxes)

        model = MagicMock()
        model.predict.return_value = [mock_result]

        preds = run_inference(model, [img])
        assert len(preds) == 1
        return preds[0]["image_id"]

    def test_image_id_from_img_00042(self, tmp_path: Path) -> None:
        assert self._run_with_filename(tmp_path, "img_00042.jpg") == 42

    def test_image_id_from_img_00001(self, tmp_path: Path) -> None:
        assert self._run_with_filename(tmp_path, "img_00001.jpg") == 1

    def test_image_id_from_img_00100(self, tmp_path: Path) -> None:
        assert self._run_with_filename(tmp_path, "img_00100.jpg") == 100

    def test_image_id_from_img_00000(self, tmp_path: Path) -> None:
        assert self._run_with_filename(tmp_path, "img_00000.jpg") == 0
