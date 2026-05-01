"""Batch inference on a folder of images (FR-06).

Replaces legacy/detect_onioncell.py. Saves annotated images plus a JSON
summary of per-image counts and per-detection records.

CLI:
    neomscope-batch --input captured_raw_images --output detection_results_captured
    neomscope-batch --input raws/JPEG_Export_Data --output runs/inference --hef models/hef/best.hef
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

from inference._camera import IMAGE_EXTS
from inference.pipeline import HailoInferencePipeline
from inference.postprocess import render_overlay
from inference.types import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU_THRESHOLD,
    PipelineConfig,
    count_by_class,
)

logger = logging.getLogger("batch_detect")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True, help="Input image folder")
    parser.add_argument("--output", type=Path, required=True, help="Output folder")
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

    if not args.input.is_dir():
        logger.error("Input directory not found: %s", args.input)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in args.input.iterdir()
                    if p.suffix in IMAGE_EXTS and p.is_file())
    if not images:
        logger.error("No images found in %s", args.input)
        return 2

    logger.info("Found %d images in %s", len(images), args.input)

    cfg = PipelineConfig(
        hef_path=args.hef,
        image_size=args.imgsz,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    try:
        pipeline = HailoInferencePipeline(cfg, mock=args.mock)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Pipeline init error: %s", exc)
        return 3

    summary: list[dict[str, object]] = []
    total_dets = 0
    total_time = 0.0

    with pipeline:
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning("Could not read %s, skipping", img_path)
                continue

            t0 = time.perf_counter()
            detections = pipeline.infer(frame)
            dt = time.perf_counter() - t0
            total_time += dt
            total_dets += len(detections)

            annotated = render_overlay(frame, detections)
            out_path = args.output / img_path.name
            cv2.imwrite(str(out_path), annotated)

            summary.append({
                "image": img_path.name,
                "infer_time_ms": round(dt * 1000, 2),
                "counts": count_by_class(detections),
                "detections": [d.to_dict() for d in detections],
            })
            logger.info("  %s: %d dets in %.1f ms", img_path.name, len(detections), dt * 1000)

    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "input": str(args.input),
                "image_count": len(images),
                "total_detections": total_dets,
                "total_infer_time_s": round(total_time, 3),
                "avg_infer_time_ms": round(total_time / max(len(images), 1) * 1000, 2),
                "results": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %s", summary_path)
    logger.info(
        "Done: %d images, %d detections, %.1f ms/img avg",
        len(images), total_dets, total_time / max(len(images), 1) * 1000,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
