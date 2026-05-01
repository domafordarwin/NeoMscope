"""YOLOv11-det postprocessing for the Hailo-10H runtime path.

Converts raw NPU output tensors to typed `Detection` records and renders
overlays. All math is pure numpy + cv2 — no Hailo or PyTorch dependency,
which keeps this module unit-testable on the dev PC.

The postprocess for det is much simpler than seg: there is no mask
prototype, no mask coefficient multiplication, no per-instance mask
resize. This is one of the main reasons §1.3.1 of the design switched
from seg to det.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import cv2
import numpy as np

from inference.types import (
    CLASS_NAMES,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    NUM_CLASSES,
    Detection,
    unletterbox_bbox,
)

logger = logging.getLogger("postprocess")


def decode_yolo_det_output(
    raw_outputs: Mapping[str, np.ndarray],
    orig_shape: tuple[int, int],
    image_size: int = 640,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    max_detections: int = 300,
) -> list[Detection]:
    """Decode raw HEF/ONNX outputs into typed Detection records.

    Expected keys in `raw_outputs`:
        bbox: shape (N, 4), normalized (cx, cy, w, h) in [0, 1].
        cls:  shape (N, NUM_CLASSES), per-class sigmoid scores in [0, 1].

    Args:
        raw_outputs: Output tensors from the runtime (HailoRT or ONNX Runtime).
        orig_shape: (height, width) of the un-letterboxed source frame.
        image_size: Square edge used at HEF compile time (must match runtime).
        conf_threshold: Drop detections below this max-class score.
        iou_threshold: NMS IoU threshold.
        max_detections: Hard cap on returned detections after NMS.

    Returns:
        List of Detection records in original frame coordinates,
        sorted by descending confidence. Empty list if nothing passes.
    """
    bbox_proto = raw_outputs["bbox"]
    cls_proto = raw_outputs["cls"]

    if bbox_proto.shape[0] != cls_proto.shape[0]:
        raise ValueError(
            f"bbox/cls row count mismatch: {bbox_proto.shape[0]} vs {cls_proto.shape[0]}"
        )
    if cls_proto.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"cls has {cls_proto.shape[1]} classes, expected {NUM_CLASSES}"
        )

    # 1. Per-row best class
    cls_scores = cls_proto.max(axis=1)
    cls_ids = cls_proto.argmax(axis=1).astype(np.int32)

    # 2. Confidence filter
    keep_mask = cls_scores >= conf_threshold
    if not keep_mask.any():
        return []

    bb = bbox_proto[keep_mask]
    sc = cls_scores[keep_mask]
    ci = cls_ids[keep_mask]

    # 3. cx,cy,w,h (normalized) → xyxy in letterbox-pixel coordinates
    cx, cy, bw, bh = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
    x1 = (cx - bw / 2) * image_size
    y1 = (cy - bh / 2) * image_size
    x2 = (cx + bw / 2) * image_size
    y2 = (cy + bh / 2) * image_size

    # 4. NMS — cv2.dnn.NMSBoxes expects (x, y, w, h) in pixels
    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    keep_idx = cv2.dnn.NMSBoxes(
        boxes_xywh,
        sc.tolist(),
        score_threshold=conf_threshold,
        nms_threshold=iou_threshold,
    )

    if len(keep_idx) == 0:
        return []

    # cv2.dnn.NMSBoxes can return shape (N,) or (N, 1) depending on version
    keep_idx = np.asarray(keep_idx).flatten()[:max_detections]

    # 5. Sort by confidence descending (NMSBoxes already roughly sorted, but be explicit)
    order = np.argsort(-sc[keep_idx])
    keep_idx = keep_idx[order]

    # 6. Reverse letterbox to original frame coords
    detections: list[Detection] = []
    for i in keep_idx:
        bbox_letterboxed = np.array([x1[i], y1[i], x2[i], y2[i]], dtype=np.float32)
        bbox_orig = unletterbox_bbox(bbox_letterboxed, orig_shape, image_size)
        detections.append(
            Detection(
                bbox=(
                    int(bbox_orig[0]),
                    int(bbox_orig[1]),
                    int(bbox_orig[2]),
                    int(bbox_orig[3]),
                ),
                class_id=int(ci[i]),
                confidence=float(sc[i]),
            )
        )

    return detections


def render_overlay(
    frame: np.ndarray,
    detections: list[Detection],
    show_labels: bool = True,
    fill_alpha: float = 0.3,
) -> np.ndarray:
    """Draw detections on frame (BGR). Returns annotated copy.

    Args:
        frame: BGR image of shape (H, W, 3).
        detections: List of Detection records in original frame coordinates.
        show_labels: If True, draw class name + confidence above each box.
        fill_alpha: 0.0-1.0, transparency of bbox fill (0 = no fill).
    """
    if not detections:
        return frame.copy()

    out = frame.copy()
    h, w = out.shape[:2]

    for d in detections:
        color = d.color
        x1, y1, x2, y2 = (
            max(0, d.bbox[0]),
            max(0, d.bbox[1]),
            min(w - 1, d.bbox[2]),
            min(h - 1, d.bbox[3]),
        )

        if fill_alpha > 0:
            overlay = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            out = cv2.addWeighted(overlay, fill_alpha, out, 1 - fill_alpha, 0)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        if show_labels:
            label = f"{d.class_name} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(th + 4, y1 - 4)
            cv2.rectangle(
                out,
                (x1, label_y - th - 4),
                (x1 + tw + 4, label_y),
                color,
                -1,
            )
            cv2.putText(
                out,
                label,
                (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return out


def render_summary(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Overlay an FPS-corner summary box with per-class counts.

    Used by live_detect.py and capture_and_detect.py for the visible HUD.
    """
    out = frame.copy()
    if not detections:
        return out

    counts: dict[str, int] = dict.fromkeys(CLASS_NAMES, 0)
    for d in detections:
        counts[d.class_name] += 1

    lines = [f"{name}: {counts[name]}" for name in CLASS_NAMES if counts[name] > 0]
    if not lines:
        return out

    # Background panel
    panel_h = 20 + 18 * len(lines)
    cv2.rectangle(out, (10, 10), (180, 10 + panel_h), (0, 0, 0), -1)
    cv2.rectangle(out, (10, 10), (180, 10 + panel_h), (255, 255, 255), 1)

    cv2.putText(
        out, "Counts", (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    for i, line in enumerate(lines):
        cv2.putText(
            out, line, (16, 48 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return out
