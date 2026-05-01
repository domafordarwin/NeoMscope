"""Unit tests for inference/types.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inference.types import (
    CLASS_COLORS,
    CLASS_NAMES,
    NUM_CLASSES,
    Detection,
    PipelineConfig,
    count_by_class,
    letterbox_size,
    unletterbox_bbox,
)


class TestClassTaxonomy:
    def test_class_names_count(self) -> None:
        assert len(CLASS_NAMES) == NUM_CLASSES == 5

    def test_class_names_order(self) -> None:
        # Order must match the legacy object_detection_classes_onion.txt
        # to keep bootstrap labeling consistent.
        assert CLASS_NAMES == ("Inter", "Pro", "Meta", "Ana", "Telo")

    def test_class_colors_complete(self) -> None:
        for i in range(NUM_CLASSES):
            assert i in CLASS_COLORS
            assert len(CLASS_COLORS[i]) == 3
            for ch in CLASS_COLORS[i]:
                assert 0 <= ch <= 255


class TestDetection:
    def test_basic_construction(self) -> None:
        d = Detection(bbox=(10, 20, 50, 80), class_id=2, confidence=0.85)
        assert d.class_name == "Meta"
        assert d.color == CLASS_COLORS[2]
        assert d.width == 40
        assert d.height == 60
        assert d.area == 2400

    def test_to_dict_is_json_friendly(self) -> None:
        import json

        d = Detection(bbox=(1, 2, 3, 4), class_id=0, confidence=0.123456)
        as_dict = d.to_dict()
        assert as_dict["class_name"] == "Inter"
        assert as_dict["confidence"] == 0.1235  # rounded to 4 decimals
        # Round-trip through JSON
        json.dumps(as_dict)

    def test_is_immutable(self) -> None:
        d = Detection(bbox=(0, 0, 1, 1), class_id=0, confidence=0.5)
        with pytest.raises(AttributeError):
            d.class_id = 1  # type: ignore[misc]


class TestPipelineConfig:
    def test_defaults(self, tmp_path: Path) -> None:
        cfg = PipelineConfig(hef_path=tmp_path / "best.hef")
        assert cfg.image_size == 640
        assert cfg.conf_threshold == 0.25
        assert cfg.iou_threshold == 0.5
        assert cfg.use_gstreamer is True

    def test_is_immutable(self, tmp_path: Path) -> None:
        cfg = PipelineConfig(hef_path=tmp_path / "best.hef")
        with pytest.raises(AttributeError):
            cfg.image_size = 1024  # type: ignore[misc]


class TestCountByClass:
    def test_empty(self) -> None:
        counts = count_by_class([])
        assert counts == {n: 0 for n in CLASS_NAMES}

    def test_mixed(self) -> None:
        dets = [
            Detection((0, 0, 10, 10), 0, 0.9),
            Detection((0, 0, 10, 10), 0, 0.8),
            Detection((0, 0, 10, 10), 2, 0.7),
        ]
        counts = count_by_class(dets)
        assert counts["Inter"] == 2
        assert counts["Meta"] == 1
        assert counts["Pro"] == 0


class TestLetterbox:
    def test_letterbox_size_square(self) -> None:
        h, w, scale = letterbox_size((640, 640), 640)
        assert h == 640
        assert w == 640
        assert scale == 1.0

    def test_letterbox_size_landscape(self) -> None:
        # 1280x720 → fit in 640x640
        h, w, scale = letterbox_size((720, 1280), 640)
        assert w == 640
        assert h == 360
        assert scale == pytest.approx(640 / 1280)

    def test_unletterbox_roundtrip(self) -> None:
        # bbox in 640x640 letterboxed space → back to 720x1280 original
        # Image was 1280x720 → scaled to 640x360 (scale=0.5), pad_y=140
        bbox_letterboxed = np.array([100.0, 240.0, 540.0, 400.0])  # x1,y1,x2,y2
        out = unletterbox_bbox(bbox_letterboxed, (720, 1280), 640)
        # Reverse: subtract pad (pad_x=0, pad_y=140), divide by 0.5
        assert out[0] == pytest.approx(200.0)
        assert out[1] == pytest.approx(200.0)
        assert out[2] == pytest.approx(1080.0)
        assert out[3] == pytest.approx(520.0)

    def test_unletterbox_clamps_to_image(self) -> None:
        # Detection partly outside the original frame
        bbox = np.array([-10.0, -10.0, 700.0, 700.0])
        out = unletterbox_bbox(bbox, (480, 640), 640)
        assert out[0] >= 0
        assert out[1] >= 0
        assert out[2] <= 640
        assert out[3] <= 480
