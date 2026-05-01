"""Render YOLO det labels onto images for human visual review.

Used after `bootstrap_labeling.py` and again after Roboflow export to
spot-check label quality before training. Produces a montage or
per-image overlay PNGs.

CLI:
    # Render all images with labels into <output>/<image>.png
    python tools/visualize_labels.py \
        --images datasets/onioncell/images/train \
        --labels datasets/onioncell/labels/train \
        --output datasets/onioncell/_review/

    # Render only the review_files from a bootstrap report
    python tools/visualize_labels.py \
        --images raws/JPEG_Export_Data captured_raw_images \
        --labels datasets/onioncell/labels/auto \
        --output datasets/onioncell/_review/ \
        --review-list datasets/onioncell/bootstrap-report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

from inference.types import CLASS_COLORS, CLASS_NAMES, NUM_CLASSES

logger = logging.getLogger("visualize_labels")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _read_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse YOLO det label file. Returns list of (cls, cx, cy, w, h)."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            logger.warning("Skipping malformed line in %s: %s", label_path, line)
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            logger.warning("Skipping unparseable line in %s: %s", label_path, line)
            continue
        out.append((cls, cx, cy, w, h))
    return out


def _draw(
    image: cv2.typing.MatLike,
    labels: list[tuple[int, float, float, float, float]],
) -> cv2.typing.MatLike:
    """Draw bboxes + class labels onto image (BGR). Returns annotated copy."""
    out = image.copy()
    h, w = out.shape[:2]

    for cls, cx, cy, bw, bh in labels:
        if not 0 <= cls < NUM_CLASSES:
            continue
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = CLASS_COLORS[cls]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = CLASS_NAMES[cls]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

    return out


def _resolve_image_dirs(args_images: list[Path]) -> dict[str, Path]:
    """Build stem -> path map across one or more image directories."""
    out: dict[str, Path] = {}
    for d in args_images:
        if not d.is_dir():
            raise FileNotFoundError(f"Image directory not found: {d}")
        for p in d.rglob("*"):
            if p.suffix in IMAGE_EXTS and p.is_file():
                out[p.stem] = p
    return out


def _filter_review_only(
    images: dict[str, Path], review_list: Path
) -> dict[str, Path]:
    data = json.loads(review_list.read_text(encoding="utf-8"))
    review_names = set(data.get("review_files", []))
    review_stems = {Path(n).stem for n in review_names}
    return {stem: p for stem, p in images.items() if stem in review_stems}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories containing source images",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Directory containing YOLO det .txt labels",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for annotated images",
    )
    parser.add_argument(
        "--review-list",
        type=Path,
        help="Optional bootstrap-report.json — render only review_files",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(message)s")

    try:
        images = _resolve_image_dirs(args.images)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    if args.review_list:
        images = _filter_review_only(images, args.review_list)
        logger.info("Filtered to %d review images.", len(images))

    args.output.mkdir(parents=True, exist_ok=True)
    rendered = 0
    skipped = 0

    for stem, img_path in images.items():
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.warning("Could not read: %s", img_path)
            skipped += 1
            continue

        labels = _read_labels(args.labels / f"{stem}.txt")
        annotated = _draw(bgr, labels)

        out_path = args.output / f"{stem}.png"
        cv2.imwrite(str(out_path), annotated)
        rendered += 1

    logger.info("Rendered %d images to %s (skipped %d)", rendered, args.output, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
