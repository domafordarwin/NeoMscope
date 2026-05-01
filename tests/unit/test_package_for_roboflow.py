"""Unit tests for tools/package_for_roboflow.py."""

from __future__ import annotations

import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import package_for_roboflow as pkg


def _make_jpeg(path: Path, h: int, w: int) -> None:
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


class TestBuildImageIndex:
    def test_collects_from_multiple_dirs(self, tmp_path: Path) -> None:
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        _make_jpeg(d1 / "a.jpg", 10, 10)
        _make_jpeg(d2 / "b.jpg", 10, 10)
        index = pkg._build_image_index([d1, d2])
        assert index["a"].name == "a.jpg"
        assert index["b"].name == "b.jpg"

    def test_later_dir_overrides_earlier(self, tmp_path: Path) -> None:
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        _make_jpeg(d1 / "x.jpg", 10, 10)
        _make_jpeg(d2 / "x.jpg", 10, 10)
        index = pkg._build_image_index([d1, d2])
        # Last dir wins
        assert index["x"].parent == d2

    def test_raises_on_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            pkg._build_image_index([tmp_path / "nope"])


class TestResizeToMax:
    def test_no_op_when_already_small(self) -> None:
        img = np.zeros((500, 800, 3), dtype=np.uint8)
        out = pkg._resize_to_max(img, 1280)
        assert out.shape == img.shape  # unchanged
        # Same buffer when no resize is needed
        assert np.shares_memory(out, img)

    def test_resizes_landscape(self) -> None:
        img = np.zeros((1500, 3000, 3), dtype=np.uint8)
        out = pkg._resize_to_max(img, 1280)
        h, w = out.shape[:2]
        assert max(h, w) == 1280
        assert w == 1280
        assert h == 640  # 3000 * 0.4267 ≈ 1280; 1500 * 0.4267 ≈ 640

    def test_resizes_portrait(self) -> None:
        img = np.zeros((3000, 1500, 3), dtype=np.uint8)
        out = pkg._resize_to_max(img, 1280)
        h, w = out.shape[:2]
        assert max(h, w) == 1280
        assert h == 1280
        assert w == 640


class TestPackage:
    def test_pairs_images_and_labels(self, tmp_path: Path) -> None:
        labels = tmp_path / "labels"
        images = tmp_path / "images"
        out = tmp_path / "out"
        labels.mkdir()
        _make_jpeg(images / "a.jpg", 100, 100)
        _make_jpeg(images / "b.jpg", 100, 100)
        (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (labels / "b.txt").write_text("")  # empty is valid

        report = pkg.package(labels, [images], out)

        assert report.paired == 2
        assert (out / "a.jpg").exists()
        assert (out / "a.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
        assert (out / "b.jpg").exists()
        assert (out / "b.txt").read_text() == ""

    def test_label_without_image_is_skipped(self, tmp_path: Path) -> None:
        labels = tmp_path / "labels"
        images = tmp_path / "images"
        out = tmp_path / "out"
        labels.mkdir()
        images.mkdir()
        (labels / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1")

        report = pkg.package(labels, [images], out)
        assert report.paired == 0
        assert "orphan" in report.skipped_no_image

    def test_resizes_large_image(self, tmp_path: Path) -> None:
        labels = tmp_path / "labels"
        images = tmp_path / "images"
        out = tmp_path / "out"
        labels.mkdir()
        _make_jpeg(images / "big.jpg", 3000, 4000)
        (labels / "big.txt").write_text("0 0.5 0.5 0.1 0.1")

        report = pkg.package(labels, [images], out, max_size=512)
        assert report.paired == 1

        # Verify the saved image was actually resized
        saved = cv2.imread(str(out / "big.jpg"))
        assert max(saved.shape[:2]) <= 512

    def test_raises_on_missing_labels_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            pkg.package(tmp_path / "nope", [tmp_path], tmp_path / "out")

    def test_raises_on_empty_labels_dir(self, tmp_path: Path) -> None:
        labels = tmp_path / "labels"
        labels.mkdir()
        with pytest.raises(RuntimeError, match=r"No \.txt labels"):
            pkg.package(labels, [tmp_path], tmp_path / "out")


class TestMakeZip:
    def test_zip_contains_flat_files(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "a.jpg").write_bytes(b"\xff\xd8\xff\xe0")
        (out_dir / "a.txt").write_text("0 0.5 0.5 0.1 0.1")
        (out_dir / "b.jpg").write_bytes(b"\xff\xd8\xff\xe0")
        (out_dir / "b.txt").write_text("")

        zip_path = tmp_path / "out.zip"
        n = pkg.make_zip(out_dir, zip_path)
        assert n == 4

        with zipfile.ZipFile(zip_path) as zf:
            names = sorted(zf.namelist())
            assert names == ["a.jpg", "a.txt", "b.jpg", "b.txt"]
            # No nested directories
            assert all("/" not in n for n in names)
