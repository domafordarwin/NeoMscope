"""Smoke tests for the inference pipeline using the mock=True path.

Verifies that the pipeline composes correctly without requiring HailoRT
or a real HEF — important so the dev PC can run integration-shaped
tests for the inference scripts.
"""

from __future__ import annotations

import numpy as np
import pytest

from inference.pipeline import HailoInferencePipeline, _letterbox
from inference.types import PipelineConfig


def test_letterbox_preserves_aspect(tmp_path):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out = _letterbox(frame, 320)
    assert out.shape == (320, 320, 3)
    # Should be filled with gray (114) outside the resized region
    assert out.dtype == np.uint8


def test_letterbox_square_no_padding():
    frame = np.full((640, 640, 3), 200, dtype=np.uint8)
    out = _letterbox(frame, 640)
    assert out.shape == (640, 640, 3)
    # No padding applied — entire image preserved
    assert (out == 200).all()


def test_mock_pipeline_constructs_without_hailo(tmp_path):
    cfg = PipelineConfig(hef_path=tmp_path / "doesnt_exist.hef")
    # mock=True should skip the file check entirely
    pipe = HailoInferencePipeline(cfg, mock=True)
    assert pipe._mock is True


def test_mock_pipeline_returns_synthetic_detection(tmp_path):
    cfg = PipelineConfig(hef_path=tmp_path / "doesnt_exist.hef")
    pipe = HailoInferencePipeline(cfg, mock=True)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    dets = pipe.infer(frame)
    # Mock injects one high-conf class-0 detection at center
    assert len(dets) == 1
    assert dets[0].class_id == 0
    assert dets[0].confidence > 0.9


def test_real_pipeline_raises_on_missing_hef(tmp_path):
    cfg = PipelineConfig(hef_path=tmp_path / "nope.hef")
    with pytest.raises(FileNotFoundError):
        HailoInferencePipeline(cfg, mock=False)


def test_pipeline_validates_frame_shape(tmp_path):
    cfg = PipelineConfig(hef_path=tmp_path / "doesnt_exist.hef")
    pipe = HailoInferencePipeline(cfg, mock=True)
    bad_frame = np.zeros((100, 100), dtype=np.uint8)  # missing channel dim
    with pytest.raises(ValueError, match="Expected"):
        pipe.infer(bad_frame)


def test_pipeline_context_manager_closes(tmp_path):
    cfg = PipelineConfig(hef_path=tmp_path / "doesnt_exist.hef")
    with HailoInferencePipeline(cfg, mock=True) as pipe:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipe.infer(frame)
    # close() should be safe to call after __exit__
    pipe.close()
