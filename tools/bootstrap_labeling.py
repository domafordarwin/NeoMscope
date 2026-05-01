"""Bootstrap YOLO det labels from the legacy Mask R-CNN model.

Inputs:
    - Trained Mask R-CNN weights: weights/mask_rcnn_onioncell_1020_0089.h5
    - Raw images: raws/JPEG_Export_Data/ (105) + captured_raw_images/ (22)

Output:
    - YOLO det `.txt` files in <output>/labels/auto/
    - bootstrap-report.json with per-class counts, confidence stats,
      and a `review_files` list flagged for manual inspection in Roboflow.

Runs inside `.legacy-venv` (Python 3.8 + TF 1.15 + mrcnn). The mrcnn
import is delayed until `bootstrap()` is called so that this module
can be imported on the dev machine for unit tests without the legacy
deps installed.

CLI:
    python tools/bootstrap_labeling.py \
        --weights weights/mask_rcnn_onioncell_1020_0089.h5 \
        --images raws/JPEG_Export_Data captured_raw_images \
        --output datasets/onioncell \
        --conf-min 0.5 \
        --review-min 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Silence TF1 deprecation noise — there are dozens of warnings per inference call
# and they obscure the actual progress output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger("bootstrap_labeling")

IMAGE_GLOBS = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")


@dataclass
class BootstrapReport:
    """Summary written to bootstrap-report.json after a run."""

    total_images: int = 0
    images_with_detections: int = 0
    total_detections: int = 0
    detections_per_class: dict[int, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    review_files: list[str] = field(default_factory=list)


def _iter_images(image_dirs: list[Path]) -> Iterator[Path]:
    """Yield .jpg/.png files from given directories, sorted by name."""
    for d in image_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"Image directory not found: {d}")
        seen: set[Path] = set()
        for pattern in IMAGE_GLOBS:
            for p in sorted(d.glob(pattern)):
                if p not in seen:
                    seen.add(p)
                    yield p


def _bbox_to_yolo(
    bbox_yxyx: tuple[float, float, float, float],
    img_h: int,
    img_w: int,
) -> tuple[float, float, float, float]:
    """Convert mrcnn bbox (y1, x1, y2, x2) in pixels to YOLO (cx, cy, w, h) normalized.

    Clamps output to [0, 1].
    """
    y1, x1, y2, x2 = bbox_yxyx
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    )


def _build_inference_config():
    """Build mrcnn InferenceConfig matching the legacy training setup.

    Imported lazily inside this function so it only runs in .legacy-venv.
    """
    from mrcnn.config import Config

    class InferenceConfig(Config):
        NAME = "onioncell"
        NUM_CLASSES = 1 + 5  # background + 5 division stages
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        IMAGE_RESIZE_MODE = "square"
        IMAGE_MIN_DIM = 512
        IMAGE_MAX_DIM = 512
        DETECTION_MIN_CONFIDENCE = 0.5

    return InferenceConfig()


def bootstrap(
    weights_path: Path,
    image_dirs: list[Path],
    output_dir: Path,
    conf_min: float = 0.5,
    review_min: float = 0.5,
) -> BootstrapReport:
    """Run the bootstrap labeling pipeline and write YOLO det labels + report.

    Args:
        weights_path: Path to legacy `.h5` Mask R-CNN weights.
        image_dirs: One or more directories containing source images.
        output_dir: Output root. Labels go to <output_dir>/labels/auto/.
        conf_min: Drop detections below this confidence.
        review_min: Flag images containing any detection below this confidence
            (between conf_min and review_min) for manual inspection.

    Returns:
        BootstrapReport with aggregate stats. Side effects:
        - Writes one .txt per source image to <output_dir>/labels/auto/.
        - Writes bootstrap-report.json to <output_dir>/.
    """
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    # Lazy imports — only run inside .legacy-venv
    import cv2
    import numpy as np
    from mrcnn.model import MaskRCNN

    cfg = _build_inference_config()
    cfg.DETECTION_MIN_CONFIDENCE = conf_min

    logger.info("Loading Mask R-CNN model from %s ...", weights_path)
    model = MaskRCNN(mode="inference", model_dir=str(output_dir / "_logs"), config=cfg)
    model.load_weights(str(weights_path), by_name=True)
    logger.info("Model loaded.")

    labels_dir = output_dir / "labels" / "auto"
    labels_dir.mkdir(parents=True, exist_ok=True)

    report = BootstrapReport()
    confidences: list[float] = []

    for img_path in _iter_images(image_dirs):
        # Load image (BGR via cv2, convert to RGB for mrcnn)
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.warning("Could not read image, skipping: %s", img_path)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Inference
        try:
            results = model.detect([rgb], verbose=0)[0]
        except Exception as exc:
            # Inference can fail mid-batch on corrupt frames; log and continue.
            logger.error("Inference failed on %s: %s", img_path.name, exc)
            continue

        boxes = results["rois"]  # (N, 4) [y1, x1, y2, x2]
        cls_ids = results["class_ids"]  # (N,) 1-indexed (mrcnn) -> 0-indexed (yolo)
        scores = results["scores"]  # (N,)

        report.total_images += 1
        if len(boxes) > 0:
            report.images_with_detections += 1

        lines: list[str] = []
        needs_review = len(boxes) == 0  # zero detections is suspicious
        # Note: strict=True is Python 3.10+ only; this script also runs in
        # the legacy Python 3.7 venv, so we assert lengths manually instead.
        if not (len(boxes) == len(cls_ids) == len(scores)):
            logger.warning(
                "mrcnn output length mismatch on %s: %d/%d/%d boxes/cls/scores",
                img_path.name, len(boxes), len(cls_ids), len(scores),
            )
            continue
        for box, cls_id, score in zip(boxes, cls_ids, scores):  # noqa: B905
            if score < conf_min:
                continue
            if score < review_min:
                needs_review = True

            # mrcnn class_ids are 1-indexed (1..5); YOLO is 0-indexed (0..4)
            yolo_cls = int(cls_id) - 1
            if yolo_cls < 0 or yolo_cls >= 5:
                logger.warning(
                    "Skipping out-of-range class_id %s in %s", cls_id, img_path.name
                )
                continue

            cx, cy, bw, bh = _bbox_to_yolo(tuple(box), h, w)

            # Sanity: drop degenerate bboxes
            if bw <= 0 or bh <= 0:
                continue

            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            report.total_detections += 1
            report.detections_per_class[yolo_cls] = (
                report.detections_per_class.get(yolo_cls, 0) + 1
            )
            confidences.append(float(score))

        # Write label file (empty file is still valid YOLO — means "no objects")
        out_txt = labels_dir / f"{img_path.stem}.txt"
        out_txt.write_text("\n".join(lines), encoding="utf-8")

        if needs_review:
            report.review_files.append(img_path.name)

        logger.info(
            "  %s: %d kept, %d total raw  %s",
            img_path.name,
            len(lines),
            len(boxes),
            "[REVIEW]" if needs_review else "",
        )

    if confidences:
        report.avg_confidence = round(float(np.mean(confidences)), 4)
        report.min_confidence = round(float(np.min(confidences)), 4)
        report.max_confidence = round(float(np.max(confidences)), 4)

    # Write report
    report_path = output_dir / "bootstrap-report.json"
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    logger.info("Wrote report: %s", report_path)
    logger.info(
        "Summary: %d images, %d w/ detections, %d total dets, %d for review",
        report.total_images,
        report.images_with_detections,
        report.total_detections,
        len(report.review_files),
    )

    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to legacy Mask R-CNN .h5 weights",
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories containing source images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/onioncell"),
        help="Output root (labels go to <output>/labels/auto/)",
    )
    parser.add_argument(
        "--conf-min",
        type=float,
        default=0.5,
        help="Drop detections below this confidence (default: 0.5)",
    )
    parser.add_argument(
        "--review-min",
        type=float,
        default=0.5,
        help="Flag images for review if any det below this (default: 0.5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        bootstrap(
            weights_path=args.weights,
            image_dirs=args.images,
            output_dir=args.output,
            conf_min=args.conf_min,
            review_min=args.review_min,
        )
    except FileNotFoundError as exc:
        logger.error("Input error: %s", exc)
        return 2
    except ImportError as exc:
        logger.error(
            "Required legacy dependency missing: %s. "
            "Did you activate .legacy-venv? See legacy/README.md.",
            exc,
        )
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
