"""Capture-and-detect interactive demo (FR-07).

Replaces legacy/capture_and_detect.py. Live preview window with:
  - SPACE: capture current frame, run inference, save raw + annotated + JSON.
  - Q: quit.

Used as the primary science-fair demo flow: judge looks through microscope
ocular onto USB camera, presses spacebar, sees instant detection + counts.

CLI:
    neomscope-capture --camera 0 --output detection_results_captured
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
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
    count_by_class,
)

logger = logging.getLogger("capture_and_detect")

WINDOW_NAME = "NeoMscope Capture (SPACE=capture, Q=quit)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--camera", default="0", help="Camera index, /dev/videoN, or 'folder:<dir>'")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("detection_results_captured"),
        help="Output root (raw + detected subfolders auto-created)",
    )
    parser.add_argument("--hef", type=Path, default=Path("models/hef/best.hef"))
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--mock", action="store_true",
                        help="Use mock pipeline (for dev PC without NPU)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    raw_dir = args.output / "raw"
    detected_dir = args.output / "detected"
    raw_dir.mkdir(parents=True, exist_ok=True)
    detected_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        hef_path=args.hef,
        image_size=args.imgsz,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    try:
        source = open_source(args.camera)
        pipeline = HailoInferencePipeline(cfg, mock=args.mock)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 2

    capture_count = 0
    logger.info("Window opened. SPACE = capture+detect, Q = quit.")

    try:
        with source, pipeline:
            for frame in source.frames():
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(" "):
                    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                    raw_path = raw_dir / f"{ts}.jpg"
                    cv2.imwrite(str(raw_path), frame)

                    detections = pipeline.infer(frame)
                    annotated = render_overlay(frame, detections)
                    annotated = render_summary(annotated, detections)

                    det_path = detected_dir / f"{ts}.jpg"
                    cv2.imwrite(str(det_path), annotated)

                    json_path = detected_dir / f"{ts}.json"
                    json_path.write_text(
                        json.dumps(
                            {
                                "timestamp": ts,
                                "raw_image": str(raw_path),
                                "detections": [d.to_dict() for d in detections],
                                "counts": count_by_class(detections),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

                    capture_count += 1
                    counts = count_by_class(detections)
                    summary_str = ", ".join(f"{k}:{v}" for k, v in counts.items() if v > 0)
                    logger.info(
                        "[capture #%d] %s — %d dets%s",
                        capture_count,
                        ts,
                        len(detections),
                        f" ({summary_str})" if summary_str else "",
                    )

                elif key == ord("q"):
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cv2.destroyAllWindows()

    logger.info("Total captures: %d", capture_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
