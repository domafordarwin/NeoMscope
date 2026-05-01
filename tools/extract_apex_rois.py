"""Extract per-root apex (meristem) ROIs from a high-res microscope scan.

Why this exists:
    The full 8000x31000 px scan is mostly elongation zone + background.
    Mitosis only happens in the apex (the rounded tip of each root,
    typically the top ~30% of each tissue blob in these scans).
    Cropping just the apex regions lets us run mrcnn on ~5-10x less area
    while keeping every cell of biological interest.

Pipeline:
    1. Build the tissue mask (same heuristic as tile_and_detect).
    2. Find connected components — one per root.
    3. For each root, take a vertical slice from the top of its bounding
       box (default top 35%) with some padding. This is the apex ROI.
    4. Save each ROI as a separate JPG at native resolution, plus an
       overview JPG showing where the ROIs were cut on the full scan.

Output:
    <output_dir>/
    ├── <stem>__apex0.jpg     # one ROI per root blob, native-res
    ├── <stem>__apex1.jpg
    ├── ...
    ├── <stem>__rois.json     # ROI coords in original-image space
    └── <stem>__rois_overview.jpg   # downsampled scan with ROI rects drawn

CLI:
    python tools/extract_apex_rois.py \
        --image "raws/JPEG_Export_Data/2020_10_07__17_31__0065-Image Export-01.jpg" \
        --output runs/apex_rois \
        --top-fraction 0.35 \
        --padding 0.05

    python tools/extract_apex_rois.py \
        --image-dir raws/JPEG_Export_Data \
        --output runs/apex_rois
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("extract_apex_rois")

IMAGE_GLOBS = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")


@dataclass
class ROI:
    """One apex ROI in the source image's pixel space."""
    index: int
    x1: int
    y1: int
    x2: int
    y2: int
    blob_x: int
    blob_y: int
    blob_w: int
    blob_h: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def build_tissue_mask(image_bgr, bg_threshold: int = 235, min_blob_px: int = 5000):
    """Same heuristic as tile_and_detect.build_tissue_mask, kept here for
    decoupling — the two scripts can run independently.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, bg_threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_blob_px:
            keep[labels == i] = 255

    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    return keep


def find_apex_rois(
    tissue_mask,
    image_shape: tuple[int, int],
    top_fraction: float = 0.35,
    padding: float = 0.05,
    min_blob_px: int = 50_000,
) -> list[ROI]:
    """Compute one apex ROI per tissue blob.

    Args:
        tissue_mask: H x W uint8 mask from build_tissue_mask.
        image_shape: (H, W) of source image — for clamping ROI coords.
        top_fraction: fraction of each blob's height to keep from the top.
            0.35 means "top 35% of the bounding box". Lower = tighter
            crop on just the apex.
        padding: extra fraction added on all sides as buffer (0.05 = 5%).
            Keeps cells right at the cut line from being clipped.
        min_blob_px: ignore blobs smaller than this (specks, debris that
            survived the tissue-mask filter).

    Returns:
        List of ROI records, one per qualifying blob, in label order.
    """
    H, W = image_shape
    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        tissue_mask, connectivity=8
    )

    rois: list[ROI] = []
    roi_idx = 0
    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_blob_px:
            continue

        bx = stats[label, cv2.CC_STAT_LEFT]
        by = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]

        # Apex = top top_fraction of the blob's bounding box
        apex_h = round(bh * top_fraction)
        apex_x1 = bx
        apex_y1 = by
        apex_x2 = bx + bw
        apex_y2 = by + apex_h

        # Pad
        pad_x = round(bw * padding)
        pad_y = round(apex_h * padding)
        apex_x1 = max(0, apex_x1 - pad_x)
        apex_y1 = max(0, apex_y1 - pad_y)
        apex_x2 = min(W, apex_x2 + pad_x)
        apex_y2 = min(H, apex_y2 + pad_y)

        rois.append(ROI(
            index=roi_idx,
            x1=int(apex_x1), y1=int(apex_y1),
            x2=int(apex_x2), y2=int(apex_y2),
            blob_x=int(bx), blob_y=int(by), blob_w=int(bw), blob_h=int(bh),
        ))
        roi_idx += 1

    return rois


def extract_one_image(
    image_path: Path,
    output_dir: Path,
    top_fraction: float = 0.35,
    padding: float = 0.05,
    min_blob_px: int = 50_000,
    save_overview: bool = True,
) -> dict:
    """Process one source image and write ROI crops + metadata."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]
    stem = image_path.stem

    logger.info("  %s: %dx%d", image_path.name, w, h)
    logger.info("    building tissue mask ...")
    mask = build_tissue_mask(image)

    rois = find_apex_rois(
        mask,
        image_shape=(h, w),
        top_fraction=top_fraction,
        padding=padding,
        min_blob_px=min_blob_px,
    )

    if not rois:
        logger.warning("    no qualifying blobs found in %s", image_path.name)
        return {"image": image_path.name, "n_rois": 0, "rois": []}

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for roi in rois:
        crop = image[roi.y1 : roi.y2, roi.x1 : roi.x2]
        out_jpg = output_dir / f"{stem}__apex{roi.index}.jpg"
        cv2.imwrite(str(out_jpg), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        metadata.append(asdict(roi))
        logger.info(
            "    apex%d: %dx%d  (blob %dx%d) -> %s",
            roi.index, roi.width, roi.height,
            roi.blob_w, roi.blob_h, out_jpg.name,
        )

    (output_dir / f"{stem}__rois.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    if save_overview:
        max_overview = 1600
        scale = min(1.0, max_overview / max(h, w))
        overview = (
            cv2.resize(image, (int(w * scale), int(h * scale)))
            if scale < 1.0 else image.copy()
        )
        for roi in rois:
            x1 = int(roi.x1 * scale)
            y1 = int(roi.y1 * scale)
            x2 = int(roi.x2 * scale)
            y2 = int(roi.y2 * scale)
            cv2.rectangle(overview, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                overview, f"apex{roi.index}",
                (x1 + 6, y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
            )
        cv2.imwrite(
            str(output_dir / f"{stem}__rois_overview.jpg"),
            overview,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )

    return {"image": image_path.name, "n_rois": len(rois), "rois": metadata}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--image", type=Path, help="Single image to process")
    p.add_argument("--image-dir", type=Path, help="Directory of images")
    p.add_argument("--output", type=Path, default=Path("runs/apex_rois"))
    p.add_argument("--top-fraction", type=float, default=0.35,
                   help="Fraction of each blob's height to keep from the top (default 0.35)")
    p.add_argument("--padding", type=float, default=0.05,
                   help="Extra fraction on all sides (default 0.05 = 5%%)")
    p.add_argument("--min-blob-px", type=int, default=50_000,
                   help="Ignore tissue blobs smaller than this (default 50,000 px)")
    p.add_argument("--no-overview", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help="With --image-dir, only process this many (0 = all)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if (args.image is None) == (args.image_dir is None):
        logger.error("Pass exactly one of --image or --image-dir")
        return 2

    if args.image:
        images = [args.image]
    else:
        images = []
        for ext in IMAGE_GLOBS:
            images.extend(sorted(args.image_dir.glob(ext)))
        if args.limit > 0:
            images = images[: args.limit]

    if not images:
        logger.error("No images found.")
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    aggregate = {"n_images": 0, "n_total_rois": 0, "per_image": []}

    for img_path in images:
        try:
            entry = extract_one_image(
                image_path=img_path,
                output_dir=args.output,
                top_fraction=args.top_fraction,
                padding=args.padding,
                min_blob_px=args.min_blob_px,
                save_overview=not args.no_overview,
            )
        except Exception as exc:
            logger.error("Failed on %s: %s", img_path.name, exc)
            continue

        aggregate["n_images"] += 1
        aggregate["n_total_rois"] += entry["n_rois"]
        aggregate["per_image"].append({
            "image": entry["image"],
            "n_rois": entry["n_rois"],
        })

    (args.output / "_aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    logger.info(
        "Done: %d images, %d total apex ROIs.",
        aggregate["n_images"], aggregate["n_total_rois"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
