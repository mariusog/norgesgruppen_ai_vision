"""Tests for run.py helper functions (no model weights needed)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Stub cv2 before importing run.py (requires libGL which isn't in CI)
if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

from run import collect_images


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
