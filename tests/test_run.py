"""Tests for run.py helper functions (no model weights needed)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# Stub cv2 before importing run.py (requires libGL which isn't in CI)
if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

from run import (
    classify_crops,
    collect_images,
    load_model,
    main,
    run_ensemble_inference,
    run_inference,
)


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


def _make_mock_ensemble_result(
    boxes_mock: MagicMock | None, orig_shape: tuple[int, int] = (480, 640)
) -> MagicMock:
    """Create a mock Result with .boxes and .orig_shape for ensemble inference."""
    result = MagicMock()
    result.boxes = boxes_mock
    result.orig_shape = orig_shape  # (height, width)
    return result


class TestRunEnsembleInference:
    """Tests for run_ensemble_inference with mocked YOLO models and WBF."""

    def test_correct_output_format(self, tmp_path: Path) -> None:
        """Ensemble produces predictions with correct keys and types."""
        img = tmp_path / "img_00042.jpg"
        img.touch()

        # Mock two models each returning one detection
        boxes1 = _make_mock_boxes([[10.0, 20.0, 110.0, 220.0]], [7], [0.95])
        boxes2 = _make_mock_boxes([[12.0, 22.0, 112.0, 222.0]], [7], [0.90])

        model1 = MagicMock()
        model1.predict.return_value = [_make_mock_ensemble_result(boxes1, (480, 640))]
        model2 = MagicMock()
        model2.predict.return_value = [_make_mock_ensemble_result(boxes2, (480, 640))]

        # WBF returns fused boxes in normalized [0,1] coords
        fused_boxes = np.array([[11.0 / 640, 21.0 / 480, 111.0 / 640, 221.0 / 480]])
        fused_scores = np.array([0.925])
        fused_labels = np.array([7])

        wbf_rv = (fused_boxes, fused_scores, fused_labels)
        with (
            patch("ensemble_boxes.weighted_boxes_fusion", return_value=wbf_rv),
            patch("run.ENSEMBLE_IMAGE_SIZES", [1280, 640]),
        ):
            preds = run_ensemble_inference([model1, model2], [img])

        assert len(preds) == 1
        p = preds[0]
        assert set(p.keys()) == {"image_id", "category_id", "bbox", "score"}
        assert isinstance(p["image_id"], int)
        assert isinstance(p["category_id"], int)
        assert isinstance(p["score"], float)
        assert isinstance(p["bbox"], list)
        assert len(p["bbox"]) == 4
        assert p["image_id"] == 42
        assert p["category_id"] == 7

        # bbox should be xywh (denormalized from WBF output)
        _x, _y, w, h = p["bbox"]
        assert w > 0
        assert h > 0

    def test_empty_detections_from_all_models(self, tmp_path: Path) -> None:
        """When all models return no detections, output is empty."""
        img = tmp_path / "img_00001.jpg"
        img.touch()

        empty_boxes = MagicMock()
        empty_boxes.__len__ = lambda self: 0

        model1 = MagicMock()
        model1.predict.return_value = [_make_mock_ensemble_result(empty_boxes, (480, 640))]
        model2 = MagicMock()
        model2.predict.return_value = [_make_mock_ensemble_result(empty_boxes, (480, 640))]

        with patch("run.ENSEMBLE_IMAGE_SIZES", [640, 640]):
            preds = run_ensemble_inference([model1, model2], [img])

        assert preds == []

    def test_per_model_image_sizes_used(self, tmp_path: Path) -> None:
        """Each model is called with its corresponding ENSEMBLE_IMAGE_SIZES entry."""
        img = tmp_path / "img_00005.jpg"
        img.touch()

        empty_boxes = MagicMock()
        empty_boxes.__len__ = lambda self: 0

        model1 = MagicMock()
        model1.predict.return_value = [_make_mock_ensemble_result(empty_boxes, (480, 640))]
        model2 = MagicMock()
        model2.predict.return_value = [_make_mock_ensemble_result(empty_boxes, (480, 640))]
        model3 = MagicMock()
        model3.predict.return_value = [_make_mock_ensemble_result(empty_boxes, (480, 640))]

        with patch("run.ENSEMBLE_IMAGE_SIZES", [1280, 640, 960]):
            run_ensemble_inference([model1, model2, model3], [img])

        # Check that each model.predict was called with its specific imgsz
        _, kwargs1 = model1.predict.call_args
        assert kwargs1["imgsz"] == 1280
        _, kwargs2 = model2.predict.call_args
        assert kwargs2["imgsz"] == 640
        _, kwargs3 = model3.predict.call_args
        assert kwargs3["imgsz"] == 960

    def test_bbox_is_xywh_format(self, tmp_path: Path) -> None:
        """Ensemble output bbox is [x, y, width, height], not xyxy."""
        img = tmp_path / "img_00010.jpg"
        img.touch()

        boxes = _make_mock_boxes([[100.0, 200.0, 300.0, 400.0]], [5], [0.8])
        model = MagicMock()
        model.predict.return_value = [_make_mock_ensemble_result(boxes, (800, 600))]

        # WBF returns normalized xyxy coords
        fused_boxes = np.array([[100.0 / 600, 200.0 / 800, 300.0 / 600, 400.0 / 800]])
        fused_scores = np.array([0.8])
        fused_labels = np.array([5])

        wbf_rv = (fused_boxes, fused_scores, fused_labels)
        with (
            patch("ensemble_boxes.weighted_boxes_fusion", return_value=wbf_rv),
            patch("run.ENSEMBLE_IMAGE_SIZES", [640]),
        ):
            preds = run_ensemble_inference([model], [img])

        assert len(preds) == 1
        _x, _y, w, h = preds[0]["bbox"]
        # Width and height should be positive differences, not absolute coords
        assert w == pytest.approx(300.0 - 100.0, abs=1e-3)
        assert h == pytest.approx(400.0 - 200.0, abs=1e-3)


class TestClassifyCrops:
    """Tests for classify_crops with mocked classifier model."""

    @staticmethod
    def _make_test_image(tmp_path: Path, filename: str, size: tuple[int, int] = (640, 480)) -> Path:
        """Create a real small test image file."""
        img_path = tmp_path / filename
        img = Image.new("RGB", size, color=(128, 128, 128))
        img.save(str(img_path))
        return img_path

    def test_overrides_category_when_high_confidence(self, tmp_path: Path) -> None:
        """Classifier overrides category_id when confidence > CLASSIFIER_CONFIDENCE_GATE."""
        img_path = self._make_test_image(tmp_path, "img_00001.jpg")

        predictions = [
            {"image_id": 1, "category_id": 5, "bbox": [10.0, 20.0, 100.0, 100.0], "score": 0.9},
        ]

        # Classifier returns class 42 with high confidence
        logits = torch.zeros(1, 356)
        logits[0, 42] = 10.0  # Very high logit -> high softmax prob

        classifier = MagicMock()
        classifier.return_value = logits
        classifier.to = MagicMock()

        with (
            patch("run.CLASSIFIER_CONFIDENCE_GATE", 0.15),
            patch("run.USE_CLASSIFIER_TTA", False),
            patch("run.USE_PROTOTYPE_MATCHING", False),
            patch("run.SCORE_FUSION_ALPHA", 1.0),
            patch.object(torch.Tensor, "to", lambda self, *a, **kw: self),
        ):
            result = classify_crops([img_path], predictions, [classifier])

        assert len(result) == 1
        assert result[0]["category_id"] == 42  # Overridden by classifier

    def test_does_not_override_when_low_confidence(self, tmp_path: Path) -> None:
        """Classifier does NOT override category_id when confidence < CLASSIFIER_CONFIDENCE_GATE."""
        img_path = self._make_test_image(tmp_path, "img_00001.jpg")

        predictions = [
            {"image_id": 1, "category_id": 5, "bbox": [10.0, 20.0, 100.0, 100.0], "score": 0.9},
        ]

        # Classifier returns roughly uniform logits -> low confidence for any class
        logits = torch.zeros(1, 356)
        # All near zero -> softmax ~ 1/356 ~ 0.0028, well below any reasonable gate

        classifier = MagicMock()
        classifier.return_value = logits
        classifier.to = MagicMock()

        with (
            patch("run.CLASSIFIER_CONFIDENCE_GATE", 0.15),
            patch("run.USE_CLASSIFIER_TTA", False),
            patch("run.USE_PROTOTYPE_MATCHING", False),
            patch("run.SCORE_FUSION_ALPHA", 1.0),
            patch.object(torch.Tensor, "to", lambda self, *a, **kw: self),
        ):
            result = classify_crops([img_path], predictions, [classifier])

        assert len(result) == 1
        assert result[0]["category_id"] == 5  # NOT overridden

    def test_empty_predictions_list(self, tmp_path: Path) -> None:
        """Empty predictions list returns immediately without calling classifier."""
        img_path = self._make_test_image(tmp_path, "img_00001.jpg")

        classifier = MagicMock()
        result = classify_crops([img_path], [], [classifier])

        assert result == []
        classifier.assert_not_called()

    def test_multiple_predictions_mixed_confidence(self, tmp_path: Path) -> None:
        """Only high-confidence classifier predictions override, low ones are kept."""
        img_path = self._make_test_image(tmp_path, "img_00002.jpg")

        predictions = [
            {"image_id": 2, "category_id": 10, "bbox": [0.0, 0.0, 50.0, 50.0], "score": 0.8},
            {"image_id": 2, "category_id": 20, "bbox": [60.0, 60.0, 50.0, 50.0], "score": 0.7},
        ]

        # First crop: high confidence for class 99; second crop: uniform (low confidence)
        logits = torch.zeros(2, 356)
        logits[0, 99] = 10.0  # High confidence -> will override
        # logits[1, :] stays at 0 -> uniform -> low confidence -> won't override

        classifier = MagicMock()
        classifier.return_value = logits
        classifier.to = MagicMock()

        with (
            patch("run.CLASSIFIER_CONFIDENCE_GATE", 0.15),
            patch("run.USE_CLASSIFIER_TTA", False),
            patch("run.USE_PROTOTYPE_MATCHING", False),
            patch("run.SCORE_FUSION_ALPHA", 1.0),
            patch.object(torch.Tensor, "to", lambda self, *a, **kw: self),
        ):
            result = classify_crops([img_path], predictions, [classifier])

        assert result[0]["category_id"] == 99  # Overridden
        assert result[1]["category_id"] == 20  # Kept original


class TestScoreFusion:
    """Tests for score fusion logic in classify_crops."""

    @staticmethod
    def _make_test_image(tmp_path: Path, filename: str, size: tuple[int, int] = (640, 480)) -> Path:
        img_path = tmp_path / filename
        img = Image.new("RGB", size, color=(128, 128, 128))
        img.save(str(img_path))
        return img_path

    def test_score_blended_when_alpha_less_than_one(self, tmp_path: Path) -> None:
        """When SCORE_FUSION_ALPHA < 1.0, score = alpha * yolo + (1-alpha) * classifier."""
        img_path = self._make_test_image(tmp_path, "img_00001.jpg")

        yolo_score = 0.9
        predictions = [
            {
                "image_id": 1,
                "category_id": 5,
                "bbox": [10.0, 20.0, 100.0, 100.0],
                "score": yolo_score,
            },
        ]

        # High-confidence classifier output for class 42
        logits = torch.zeros(1, 356)
        logits[0, 42] = 10.0

        classifier = MagicMock()
        classifier.return_value = logits
        classifier.to = MagicMock()

        alpha = 0.7
        with (
            patch("run.CLASSIFIER_CONFIDENCE_GATE", 0.15),
            patch("run.USE_CLASSIFIER_TTA", False),
            patch("run.USE_PROTOTYPE_MATCHING", False),
            patch("run.SCORE_FUSION_ALPHA", alpha),
            patch.object(torch.Tensor, "to", lambda self, *a, **kw: self),
        ):
            result = classify_crops([img_path], predictions, [classifier])

        # Compute expected classifier confidence (softmax of logits)
        expected_conf = torch.nn.functional.softmax(logits, dim=1).max().item()
        expected_score = alpha * yolo_score + (1.0 - alpha) * expected_conf

        assert result[0]["score"] == pytest.approx(expected_score, abs=1e-5)
        assert result[0]["score"] != yolo_score  # Should differ from original

    def test_score_preserved_when_alpha_is_one(self, tmp_path: Path) -> None:
        """When SCORE_FUSION_ALPHA = 1.0, the YOLO score is preserved."""
        img_path = self._make_test_image(tmp_path, "img_00001.jpg")

        yolo_score = 0.85
        predictions = [
            {
                "image_id": 1,
                "category_id": 5,
                "bbox": [10.0, 20.0, 100.0, 100.0],
                "score": yolo_score,
            },
        ]

        logits = torch.zeros(1, 356)
        logits[0, 42] = 10.0

        classifier = MagicMock()
        classifier.return_value = logits
        classifier.to = MagicMock()

        with (
            patch("run.CLASSIFIER_CONFIDENCE_GATE", 0.15),
            patch("run.USE_CLASSIFIER_TTA", False),
            patch("run.USE_PROTOTYPE_MATCHING", False),
            patch("run.SCORE_FUSION_ALPHA", 1.0),
            patch.object(torch.Tensor, "to", lambda self, *a, **kw: self),
        ):
            result = classify_crops([img_path], predictions, [classifier])

        assert result[0]["score"] == pytest.approx(yolo_score, abs=1e-5)
