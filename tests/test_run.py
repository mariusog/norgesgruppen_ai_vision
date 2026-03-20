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

from run import collect_images, load_model, main, run_inference


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

    def test_does_not_recurse_into_subdirs(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "img_00001.jpg").touch()
        (tmp_path / "img_00002.jpg").touch()
        result = collect_images(tmp_path)
        assert len(result) == 1
        assert result[0].name == "img_00002.jpg"


class TestLoadModel:
    def test_load_model_file_not_found(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError when weights file does not exist."""
        with (
            patch("run.MODEL_PATH", str(tmp_path / "nonexistent.pt")),
            patch("run.MODEL_ENGINE_PATH", str(tmp_path / "nonexistent.engine")),
            pytest.raises(FileNotFoundError, match="No model found at"),
        ):
            load_model()

    def test_load_model_prefers_engine_over_pt(self, tmp_path: Path) -> None:
        """When both .engine and .pt exist, load_model should pick .engine."""
        engine_file = tmp_path / "model.engine"
        pt_file = tmp_path / "model.pt"
        engine_file.touch()
        pt_file.touch()

        with (
            patch("run.MODEL_PATH", str(pt_file)),
            patch("run.MODEL_ENGINE_PATH", str(engine_file)),
            patch("run.YOLO") as mock_yolo,
        ):
            mock_yolo.return_value.to = MagicMock()
            load_model()
            mock_yolo.assert_called_once_with(str(engine_file))

    def test_load_model_falls_back_to_pt(self, tmp_path: Path) -> None:
        """When only .pt exists, load_model should use it."""
        pt_file = tmp_path / "model.pt"
        pt_file.touch()

        with (
            patch("run.MODEL_PATH", str(pt_file)),
            patch("run.MODEL_ENGINE_PATH", str(tmp_path / "missing.engine")),
            patch("run.YOLO") as mock_yolo,
        ):
            mock_yolo.return_value.to = MagicMock()
            load_model()
            mock_yolo.assert_called_once_with(str(pt_file))


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

    def test_run_inference_multiple_detections(self, tmp_path: Path) -> None:
        """Multiple detections in one image all appear in output."""
        img = tmp_path / "img_00010.jpg"
        img.touch()

        boxes = _make_mock_boxes(
            xyxy_list=[[0.0, 0.0, 50.0, 50.0], [100.0, 100.0, 200.0, 200.0]],
            cls_list=[3, 7],
            conf_list=[0.9, 0.8],
        )
        mock_result = _make_mock_result(boxes)
        model = MagicMock()
        model.predict.return_value = [mock_result]

        preds = run_inference(model, [img])
        assert len(preds) == 2
        assert preds[0]["category_id"] == 3
        assert preds[1]["category_id"] == 7
        assert all(p["image_id"] == 10 for p in preds)

    def test_run_inference_multiple_images(self, tmp_path: Path) -> None:
        """Batch of images each produce independent predictions."""
        img1 = tmp_path / "img_00001.jpg"
        img2 = tmp_path / "img_00002.jpg"
        img1.touch()
        img2.touch()

        boxes1 = _make_mock_boxes([[10.0, 10.0, 20.0, 20.0]], [0], [0.7])
        boxes2 = _make_mock_boxes([[30.0, 30.0, 40.0, 40.0]], [5], [0.6])

        model = MagicMock()
        model.predict.return_value = [
            _make_mock_result(boxes1),
            _make_mock_result(boxes2),
        ]

        preds = run_inference(model, [img1, img2])
        assert len(preds) == 2
        assert preds[0]["image_id"] == 1
        assert preds[1]["image_id"] == 2

    def test_run_inference_no_images(self) -> None:
        """Empty image list produces empty predictions."""
        model = MagicMock()
        preds = run_inference(model, [])
        assert preds == []
        model.predict.assert_not_called()

    def test_run_inference_bbox_is_xywh(self, tmp_path: Path) -> None:
        """Verify bbox is [x, y, width, height], not [x1, y1, x2, y2]."""
        img = tmp_path / "img_00005.jpg"
        img.touch()
        boxes = _make_mock_boxes([[100.0, 200.0, 350.0, 450.0]], [0], [0.5])
        model = MagicMock()
        model.predict.return_value = [_make_mock_result(boxes)]

        preds = run_inference(model, [img])
        bbox = preds[0]["bbox"]
        assert bbox[0] == pytest.approx(100.0)  # x
        assert bbox[1] == pytest.approx(200.0)  # y
        assert bbox[2] == pytest.approx(250.0)  # width = 350 - 100
        assert bbox[3] == pytest.approx(250.0)  # height = 450 - 200

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


class TestMain:
    def test_main_rejects_nonexistent_input(self, tmp_path: Path) -> None:
        """main() raises NotADirectoryError for missing input dir."""
        fake_input = str(tmp_path / "nonexistent")
        fake_output = str(tmp_path / "out.json")
        with (
            patch("sys.argv", ["run.py", "--input", fake_input, "--output", fake_output]),
            pytest.raises(NotADirectoryError, match="not a directory"),
        ):
            main()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_main_end_to_end(self, tmp_path: Path) -> None:
        """main() writes valid JSON output with mocked model."""
        import json

        input_dir = tmp_path / "images"
        input_dir.mkdir()
        (input_dir / "img_00001.jpg").touch()
        output_file = tmp_path / "predictions.json"

        boxes = _make_mock_boxes([[5.0, 5.0, 15.0, 15.0]], [2], [0.85])
        mock_result = _make_mock_result(boxes)
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]

        with (
            patch("sys.argv", ["run.py", "--input", str(input_dir), "--output", str(output_file)]),
            patch("run.load_model", return_value=mock_model),
        ):
            main()

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["image_id"] == 1
        assert data[0]["category_id"] == 2
