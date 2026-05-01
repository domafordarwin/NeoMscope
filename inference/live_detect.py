"""Live USB/CSI camera inference for the science-fair demo (FR-05).

Replaces legacy/live_detect.py. Same UX (live window with overlays + FPS),
but powered by the Hailo NPU through HailoInferencePipeline.

CLI:
    neomscope-live --camera 0
    neomscope-live --camera /dev/video0 --hef models/hef/best.hef
    neomscope-live --camera folder:captured_raw_images   # offline preview
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from inference._camera import open_source
from inference.pipeline import HailoInferencePipeline
from inference.postprocess import render_overlay, render_summary
from inference.types import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU_THRESHOLD,
    PipelineConfig,
)

logger = logging.getLogger("live_detect")

WINDOW_NAME = "NeoMscope Live"


class FpsCounter:
    """Rolling FPS over the last `window` frame intervals."""

    def __init__(self, window: int = 60) -> None:
        self._window = window
        self._times: list[float] = []

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times = self._times[-self._window :]
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera index, /dev/videoN, or 'folder:<dir>'",
    )
    parser.add_argument(
        "--hef",
        type=Path,
        default=Path("models/hef/best.hef"),
        help="Path to compiled .hef model",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--no-window", action="store_true",
                        help="Skip GUI (headless mode)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock pipeline (for dev PC without NPU)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = PipelineConfig(
        hef_path=args.hef,
        image_size=args.imgsz,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    try:
        source = open_source(args.camera)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Camera error: %s", exc)
        return 2

    try:
        pipeline = HailoInferencePipeline(cfg, mock=args.mock)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Pipeline init error: %s", exc)
        source.close()
        return 3

    fps = FpsCounter()
    logger.info("Press 'q' in the window or Ctrl+C to quit.")

    try:
        with source, pipeline:
            for frame in source.frames():
                detections = pipeline.infer(frame)
                annotated = render_overlay(frame, detections)
                annotated = render_summary(annotated, detections)

                fps_value = fps.tick()
                cv2.putText(
                    annotated,
                    f"FPS: {fps_value:.1f}  Det: {len(detections)}",
                    (10, annotated.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                if not args.no_window:
                    cv2.imshow(WINDOW_NAME, annotated)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down.")
    finally:
        if not args.no_window:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
