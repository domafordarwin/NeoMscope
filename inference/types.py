"""Domain types shared across the inference pipeline.

Source of truth for class taxonomy, color mapping, and detection record shape.
Designed to be importable on both dev PC (no Hailo) and Pi 5 (Hailo) — depends
only on `numpy` and stdlib, no GStreamer or hailo_platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

CLASS_NAMES: Final[tuple[str, ...]] = ("Inter", "Pro", "Meta", "Ana", "Telo")
"""5 mitotic phases: 간기, 전기, 중기, 후기, 말기. YOLO 0-indexed (no background)."""

CLASS_COLORS: Final[dict[int, tuple[int, int, int]]] = {
    0: (0, 50, 255),    # Inter — red-orange (BGR)
    1: (0, 255, 0),     # Pro — green
    2: (255, 0, 0),     # Meta — blue
    3: (255, 255, 0),   # Ana — cyan
    4: (0, 255, 255),   # Telo — yellow
}
"""BGR color per class (OpenCV convention) for overlay rendering."""

NUM_CLASSES: Final[int] = len(CLASS_NAMES)
DEFAULT_IMAGE_SIZE: Final[int] = 640
DEFAULT_CONF_THRESHOLD: Final[float] = 0.25
DEFAULT_IOU_THRESHOLD: Final[float] = 0.5


@dataclass(frozen=True, slots=True)
class Detection:
    """Single detection record produced by the YOLO det postprocessing pipeline."""

    bbox: tuple[int, int, int, int]
    """(x1, y1, x2, y2) in original frame pixel coordinates, ints."""

    class_id: int
    """0..NUM_CLASSES-1; index into CLASS_NAMES / CLASS_COLORS."""

    confidence: float
    """Sigmoid score in [0, 1]."""

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]

    @property
    def color(self) -> tuple[int, int, int]:
        return CLASS_COLORS[self.class_id]

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form for batch output."""
        return {
            "bbox": list(self.bbox),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
        }


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Runtime configuration for HailoInferencePipeline.

    All fields have safe defaults so callers only override what they need.
    """

    hef_path: Path
    """Path to compiled .hef file."""

    image_size: int = DEFAULT_IMAGE_SIZE
    """Square input edge in pixels. Must match the size used at HEF compile time."""

    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    """Minimum confidence to keep a detection."""

    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    """NMS IoU threshold."""

    max_detections: int = 300
    """Hard cap on detections per frame after NMS."""

    use_gstreamer: bool = True
    """If False, fall back to direct HailoRT Python API (no GStreamer pipeline)."""


def count_by_class(detections: list[Detection]) -> dict[str, int]:
    """Tally detections per class name. Used by all three inference scripts."""
    counts = dict.fromkeys(CLASS_NAMES, 0)
    for d in detections:
        counts[d.class_name] += 1
    return counts


def letterbox_size(orig_shape: tuple[int, int], target: int) -> tuple[int, int, float]:
    """Compute letterbox scaled size + scale factor for a target square edge.

    Args:
        orig_shape: (height, width) of original frame.
        target: target square edge length.

    Returns:
        (scaled_h, scaled_w, scale): scaled dims that fit in the target square,
        and the scale factor applied. Padding is applied by caller.
    """
    h, w = orig_shape
    scale = min(target / h, target / w)
    return round(h * scale), round(w * scale), scale


def unletterbox_bbox(
    bbox_xyxy: np.ndarray,
    orig_shape: tuple[int, int],
    target: int,
) -> np.ndarray:
    """Reverse letterbox transform on a bbox.

    Args:
        bbox_xyxy: shape (4,) array, (x1, y1, x2, y2) in target-square coordinates.
        orig_shape: (height, width) of original frame.
        target: target square edge that was used for the forward transform.

    Returns:
        shape (4,) array in original frame pixel coordinates, clamped to image bounds.
    """
    h, w = orig_shape
    scaled_h, scaled_w, scale = letterbox_size(orig_shape, target)
    pad_x = (target - scaled_w) // 2
    pad_y = (target - scaled_h) // 2

    x1, y1, x2, y2 = bbox_xyxy
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    return np.clip([x1, y1, x2, y2], [0, 0, 0, 0], [w, h, w, h])
