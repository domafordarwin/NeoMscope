"""Unit tests for inference/postprocess.py with synthetic raw outputs."""

from __future__ import annotations

import numpy as np
import pytest

from inference.postprocess import (
    decode_yolo_det_output,
    render_overlay,
    render_summary,
)
from inference.types import NUM_CLASSES, Detection


def _make_raw_output(
    n_anchors: int,
    detections_to_inject: list[tuple[int, float, float, float, float, float]],
) -> dict[str, np.ndarray]:
    """Build synthetic (bbox, cls) tensors with specified detections.

    Each tuple: (class_id, cx, cy, w, h, score).
    All non-injected anchors get score 0 so they're filtered out.
    """
    bbox = np.zeros((n_anchors, 4), dtype=np.float32)
    cls = np.zeros((n_anchors, NUM_CLASSES), dtype=np.float32)

    for i, (cls_id, cx, cy, w, h, score) in enumerate(detections_to_inject):
        bbox[i] = [cx, cy, w, h]
        cls[i, cls_id] = score

    return {"bbox": bbox, "cls": cls}


class TestDecodeYoloDetOutput:
    def test_empty_output_when_all_below_threshold(self) -> None:
        # All scores 0.1, threshold 0.25 → nothing passes
        raw = _make_raw_output(5, [(0, 0.5, 0.5, 0.2, 0.2, 0.1)])
        result = decode_yolo_det_output(raw, orig_shape=(640, 640), conf_threshold=0.25)
        assert result == []

    def test_single_detection_passes(self) -> None:
        raw = _make_raw_output(1, [(2, 0.5, 0.5, 0.4, 0.4, 0.9)])
        result = decode_yolo_det_output(raw, orig_shape=(640, 640))
        assert len(result) == 1
        det = result[0]
        assert det.class_id == 2
        assert det.confidence == pytest.approx(0.9)
        # bbox center ~ (320, 320), 40% size → ~ (192, 192)-(448, 448)
        x1, y1, x2, y2 = det.bbox
        assert 180 <= x1 <= 200
        assert 180 <= y1 <= 200
        assert 440 <= x2 <= 460
        assert 440 <= y2 <= 460

    def test_sorted_by_confidence_descending(self) -> None:
        raw = _make_raw_output(
            3,
            [
                (0, 0.2, 0.2, 0.1, 0.1, 0.5),
                (1, 0.5, 0.5, 0.1, 0.1, 0.95),
                (2, 0.8, 0.8, 0.1, 0.1, 0.7),
            ],
        )
        result = decode_yolo_det_output(raw, orig_shape=(640, 640))
        assert len(result) == 3
        # Highest conf first
        assert result[0].confidence == pytest.approx(0.95)
        assert result[1].confidence == pytest.approx(0.7)
        assert result[2].confidence == pytest.approx(0.5)

    def test_nms_suppresses_overlapping(self) -> None:
        # Two overlapping bboxes of same class — NMS should keep only the higher-confidence one
        raw = _make_raw_output(
            2,
            [
                (0, 0.5, 0.5, 0.4, 0.4, 0.9),  # IoU should be high with next
                (0, 0.51, 0.51, 0.4, 0.4, 0.8),
            ],
        )
        result = decode_yolo_det_output(raw, orig_shape=(640, 640), iou_threshold=0.3)
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.9)

    def test_max_detections_caps_output(self) -> None:
        # 50 detections, cap at 10
        injections = [(0, 0.05 + i * 0.015, 0.5, 0.01, 0.01, 0.5 + i * 0.005)
                      for i in range(50)]
        raw = _make_raw_output(50, injections)
        result = decode_yolo_det_output(raw, orig_shape=(640, 640), max_detections=10)
        assert len(result) <= 10

    def test_unletterbox_to_landscape(self) -> None:
        # Landscape source 1280x720 → letterboxed to 640x640 with 140px vertical padding
        raw = _make_raw_output(1, [(0, 0.5, 0.5, 0.5, 0.5, 0.9)])
        # Detection at center of 640x640 letterbox space
        # Center of 640x640 is (320, 320). After unletterbox: cx becomes 640 (1280/2), cy becomes 360 (720/2).
        result = decode_yolo_det_output(raw, orig_shape=(720, 1280))
        assert len(result) == 1
        x1, y1, x2, y2 = result[0].bbox
        assert 0 <= x1 <= 1280
        assert 0 <= y1 <= 720
        assert x2 <= 1280
        assert y2 <= 720

    def test_raises_on_shape_mismatch(self) -> None:
        bad = {
            "bbox": np.zeros((10, 4), dtype=np.float32),
            "cls": np.zeros((5, NUM_CLASSES), dtype=np.float32),
        }
        with pytest.raises(ValueError, match="row count mismatch"):
            decode_yolo_det_output(bad, orig_shape=(640, 640))

    def test_raises_on_wrong_class_count(self) -> None:
        bad = {
            "bbox": np.zeros((5, 4), dtype=np.float32),
            "cls": np.zeros((5, 3), dtype=np.float32),  # only 3 classes
        }
        with pytest.raises(ValueError, match="3 classes, expected"):
            decode_yolo_det_output(bad, orig_shape=(640, 640))


class TestRenderOverlay:
    def test_empty_detections_returns_copy(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = render_overlay(frame, [])
        assert out.shape == frame.shape
        # Should be unchanged
        assert np.array_equal(out, frame)

    def test_draws_at_least_one_colored_pixel(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Detection(bbox=(20, 20, 80, 80), class_id=1, confidence=0.9)
        out = render_overlay(frame, [det])
        # Some pixels should now be non-zero (from box draw or fill)
        assert (out > 0).any()

    def test_clamps_oob_bbox(self) -> None:
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        det = Detection(bbox=(-10, -10, 100, 100), class_id=0, confidence=0.5)
        # Should not crash, just clamp
        out = render_overlay(frame, [det])
        assert out.shape == frame.shape


class TestRenderSummary:
    def test_empty_returns_unmodified_size(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = render_summary(frame, [])
        assert out.shape == frame.shape

    def test_with_detections_draws_panel(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [
            Detection(bbox=(0, 0, 10, 10), class_id=0, confidence=0.9),
            Detection(bbox=(0, 0, 10, 10), class_id=0, confidence=0.8),
            Detection(bbox=(0, 0, 10, 10), class_id=2, confidence=0.7),
        ]
        out = render_summary(frame, dets)
        # Panel area should have some non-zero pixels (white text/border)
        panel = out[10:80, 10:180]
        assert (panel > 0).any()
