"""Tests that constants are within valid ranges for the competition."""

from __future__ import annotations

from src.constants import (
    CONFIDENCE_THRESHOLD,
    HALF_PRECISION,
    IMAGE_SIZE,
    INFERENCE_BATCH_SIZE,
    INFERENCE_TIMEOUT,
    IOU_THRESHOLD,
    MAX_WEIGHT_SIZE_MB,
    MODEL_PATH,
    NUM_CLASSES,
)


class TestInferenceConstants:
    def test_confidence_threshold_in_range(self) -> None:
        assert 0.0 < CONFIDENCE_THRESHOLD < 1.0

    def test_iou_threshold_in_range(self) -> None:
        assert 0.0 < IOU_THRESHOLD < 1.0

    def test_image_size_positive(self) -> None:
        assert IMAGE_SIZE > 0
        assert IMAGE_SIZE % 32 == 0, "YOLO requires image size divisible by 32"

    def test_batch_size_positive(self) -> None:
        assert INFERENCE_BATCH_SIZE >= 1

    def test_half_precision_is_bool(self) -> None:
        assert isinstance(HALF_PRECISION, bool)


class TestCompetitionConstants:
    def test_num_classes(self) -> None:
        assert NUM_CLASSES == 356

    def test_model_path_is_pt(self) -> None:
        assert MODEL_PATH.endswith(".pt")

    def test_timeout(self) -> None:
        assert INFERENCE_TIMEOUT == 300

    def test_weight_limit(self) -> None:
        assert MAX_WEIGHT_SIZE_MB == 420
