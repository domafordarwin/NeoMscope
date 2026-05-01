"""Unit tests for tools/bootstrap_labeling.py — runs in main venv.

These tests cover the pure helpers (coord conversion, image iteration)
without touching mrcnn/TF1 — they pass on the dev PC and in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import bootstrap_labeling as bl


class TestBboxToYolo:
    def test_centered_full_image(self) -> None:
        # Image 100x100, bbox covering whole image
        cx, cy, w, h = bl._bbox_to_yolo((0, 0, 100, 100), 100, 100)
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)
        assert w == pytest.approx(1.0)
        assert h == pytest.approx(1.0)

    def test_corner_quarter(self) -> None:
        # Top-left quarter
        cx, cy, w, h = bl._bbox_to_yolo((0, 0, 50, 50), 100, 100)
        assert cx == pytest.approx(0.25)
        assert cy == pytest.approx(0.25)
        assert w == pytest.approx(0.5)
        assert h == pytest.approx(0.5)

    def test_clamps_out_of_bounds(self) -> None:
        # bbox extending past image (mrcnn sometimes returns these)
        cx, cy, w, h = bl._bbox_to_yolo((-10, -10, 110, 110), 100, 100)
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0
        assert 0.0 <= w <= 1.0
        assert 0.0 <= h <= 1.0

    def test_rectangular_image(self) -> None:
        # 200 wide, 100 tall, bbox in right half
        cx, cy, w, h = bl._bbox_to_yolo((25, 100, 75, 200), 100, 200)
        assert cx == pytest.approx(0.75)
        assert cy == pytest.approx(0.5)
        assert w == pytest.approx(0.5)
        assert h == pytest.approx(0.5)


class TestIterImages:
    def test_finds_jpg_and_png(self, tmp_path: Path) -> None:
        d = tmp_path / "imgs"
        d.mkdir()
        (d / "a.jpg").touch()
        (d / "b.png").touch()
        (d / "c.txt").touch()  # not an image

        result = list(bl._iter_images([d]))
        names = sorted(p.name for p in result)
        assert names == ["a.jpg", "b.png"]

    def test_dedupes_across_glob_patterns(self, tmp_path: Path) -> None:
        # Make sure case-insensitive matching doesn't double-yield
        d = tmp_path / "imgs"
        d.mkdir()
        (d / "img.jpg").touch()
        result = list(bl._iter_images([d]))
        assert len(result) == 1

    def test_multiple_dirs_in_order(self, tmp_path: Path) -> None:
        d1, d2 = tmp_path / "d1", tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "a.jpg").touch()
        (d2 / "b.jpg").touch()
        result = list(bl._iter_images([d1, d2]))
        assert [p.name for p in result] == ["a.jpg", "b.jpg"]

    def test_raises_on_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            list(bl._iter_images([tmp_path / "does_not_exist"]))


class TestBootstrapReportShape:
    """The dataclass shape is part of the contract with tools/visualize_labels.py."""

    def test_review_files_field_default_empty(self) -> None:
        report = bl.BootstrapReport()
        assert report.review_files == []

    def test_detections_per_class_default_empty(self) -> None:
        report = bl.BootstrapReport()
        assert report.detections_per_class == {}
