"""Tile a high-resolution microscope scan and run the legacy Mask R-CNN
on each tile, then merge detections back to full-image coordinates.

Why this exists:
    The legacy `OnionCellConfig` has IMAGE_MIN_DIM=IMAGE_MAX_DIM=512, meaning
    mrcnn resizes any input to 512x512 before inference. Feeding it the full
    7400x26000 px scan compresses every cell into a few pixels and the model
    can only fit one bbox per visible structure. Tiling at native resolution
    keeps each cell at its true ~80-160 px size and lets mrcnn detect them
    individually.

Output (per source image):
    <output_dir>/
    ├── <stem>/
    │   ├── tile_<row>_<col>.jpg           # 512x512 tile crops
    │   ├── tile_<row>_<col>.json          # per-tile detection list
    │   └── _overview.jpg                  # downsampled full image with global bboxes
    ├── <stem>.txt                          # YOLO det labels (global coords, normalized)
    └── <stem>.report.json                  # tile count, total dets, per-class

Run inside the conda neomscope-legacy env (Python 3.7 + TF 1.15 + mrcnn).

CLI:
    python tools/tile_and_detect.py \
        --weights weights/mask_rcnn_onioncell_1020_0089.h5 \
        --image "raws/JPEG_Export_Data/2020_10_07__17_31__0065-Image Export-01.jpg" \
        --output runs/tiled_detect \
        --tile-size 512 \
        --tile-stride 384 \
        --conf-min 0.5 \
        --save-tiles

    # All 105 images at once (no per-tile JPG dump):
    python tools/tile_and_detect.py \
        --weights weights/mask_rcnn_onioncell_1020_0089.h5 \
        --image-dir raws/JPEG_Export_Data \
        --output runs/tiled_detect \
        --conf-min 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Silence TF1 deprecation noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger("tile_and_detect")

IMAGE_GLOBS = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")


@dataclass
class Detection:
    """One cell detection in global image coords."""
    class_id: int           # 0-indexed YOLO class (0..4)
    score: float            # confidence
    x1: int
    y1: int
    x2: int
    y2: int
    tile_row: int
    tile_col: int


@dataclass
class TileReport:
    image_name: str = ""
    image_width: int = 0
    image_height: int = 0
    tile_size: int = 0
    tile_stride: int = 0
    n_tiles: int = 0
    n_tiles_with_detections: int = 0
    total_detections: int = 0
    dropped_too_small: int = 0
    dropped_outside_tissue: int = 0
    detections_per_class: dict[int, int] = field(default_factory=dict)
    avg_confidence: float = 0.0


def _iter_tiles(h: int, w: int, tile_size: int, stride: int):
    """Yield (row, col, y, x, h_eff, w_eff) for each tile.

    Edges are clipped to the image (last tile may be shifted left/up to keep
    full size when possible, otherwise padded by mrcnn during letterbox).
    """
    n_rows = max(1, (h - tile_size) // stride + (1 if (h - tile_size) % stride else 0) + 1)
    n_cols = max(1, (w - tile_size) // stride + (1 if (w - tile_size) % stride else 0) + 1)
    if h <= tile_size:
        n_rows = 1
    if w <= tile_size:
        n_cols = 1

    for r in range(n_rows):
        y = min(r * stride, max(0, h - tile_size))
        for c in range(n_cols):
            x = min(c * stride, max(0, w - tile_size))
            h_eff = min(tile_size, h - y)
            w_eff = min(tile_size, w - x)
            yield r, c, y, x, h_eff, w_eff


def _build_inference_config(tile_size: int, conf_min: float):
    """Build mrcnn InferenceConfig sized for the tile."""
    from mrcnn.config import Config

    class TileInferenceConfig(Config):
        NAME = "onioncell_tile"
        NUM_CLASSES = 1 + 5
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        IMAGE_RESIZE_MODE = "square"
        IMAGE_MIN_DIM = tile_size
        IMAGE_MAX_DIM = tile_size
        DETECTION_MIN_CONFIDENCE = conf_min

    return TileInferenceConfig()


def _is_tile_skippable(tile_bgr, std_threshold: float = 8.0) -> bool:
    """Skip nearly-uniform tiles (background-only) to save inference time.

    Cell-rich tiles have visible texture (std > 20 typical). Pure white
    background has std < 5.
    """
    # Use grayscale std; cheaper than per-channel
    import cv2
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.std()) < std_threshold


def build_tissue_mask(image_bgr, bg_threshold: int = 235, min_blob_px: int = 5000):
    """Build a binary mask of where actual onion-root tissue is in the scan.

    Used to filter out detections that the model places in plain white
    background. Heuristic:
        1. Threshold: pixels darker than bg_threshold are candidate tissue.
        2. Morphological close to fill cell-wall gaps inside tissue.
        3. Drop tiny connected components (< min_blob_px) — these are
           specks, ink dots, scratches.
        4. Dilate slightly so tissue-edge cells don't get excluded.

    Returns a uint8 mask of the same H/W as image (0 = background, 255 = tissue).
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Tissue = darker than threshold
    _, mask = cv2.threshold(gray, bg_threshold, 255, cv2.THRESH_BINARY_INV)

    # Close small holes (cell walls between dark cells)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Drop tiny components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n_labels):  # skip background label 0
        if stats[i, cv2.CC_STAT_AREA] >= min_blob_px:
            keep[labels == i] = 255

    # Dilate so cells right at tissue edges aren't filtered out
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    return keep


def detect_one_image(
    model,
    image_path: Path,
    output_dir: Path,
    tile_size: int = 512,
    tile_stride: int = 384,
    skip_blank_tiles: bool = True,
    save_tiles: bool = False,
    save_overview: bool = True,
    min_bbox_size: int = 25,
    use_tissue_mask: bool = True,
) -> TileReport:
    """Run tiled inference on one image and return its report.

    The mrcnn model is reused across calls (loaded once by the caller).
    """
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]
    stem = image_path.stem
    img_out_dir = output_dir / stem
    img_out_dir.mkdir(parents=True, exist_ok=True)

    # Build tissue mask once per image — used to filter out detections
    # the model places in plain background (a known mrcnn failure mode here).
    tissue_mask = None
    if use_tissue_mask:
        logger.info("    building tissue mask ...")
        tissue_mask = build_tissue_mask(image)
        # Save mask preview for sanity
        mask_preview_size = 1600
        mask_scale = min(1.0, mask_preview_size / max(h, w))
        if mask_scale < 1.0:
            small_mask = cv2.resize(
                tissue_mask, (int(w * mask_scale), int(h * mask_scale)),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            small_mask = tissue_mask
        cv2.imwrite(str(img_out_dir / "_tissue_mask.jpg"), small_mask, [cv2.IMWRITE_JPEG_QUALITY, 80])

    report = TileReport(
        image_name=image_path.name,
        image_width=w,
        image_height=h,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )

    all_detections: list[Detection] = []
    tile_metas: dict[str, dict] = {}

    tiles = list(_iter_tiles(h, w, tile_size, tile_stride))
    report.n_tiles = len(tiles)
    logger.info(
        "  %s: %dx%d -> %d tiles (size=%d, stride=%d)",
        image_path.name, w, h, len(tiles), tile_size, tile_stride,
    )

    for r, c, y, x, h_eff, w_eff in tiles:
        tile_bgr = image[y : y + h_eff, x : x + w_eff]

        # Pad to tile_size if at edge (mrcnn expects square input via letterbox)
        if h_eff < tile_size or w_eff < tile_size:
            padded = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
            padded[:h_eff, :w_eff] = tile_bgr
            infer_tile = padded
        else:
            infer_tile = tile_bgr

        if skip_blank_tiles and _is_tile_skippable(infer_tile):
            continue

        # mrcnn wants RGB
        rgb = cv2.cvtColor(infer_tile, cv2.COLOR_BGR2RGB)

        try:
            results = model.detect([rgb], verbose=0)[0]
        except Exception as exc:
            logger.warning("    tile r=%d c=%d failed: %s", r, c, exc)
            continue

        boxes = results["rois"]      # (N, 4) [y1, x1, y2, x2] in tile coords
        cls_ids = results["class_ids"]
        scores = results["scores"]

        if len(boxes) == 0:
            continue

        report.n_tiles_with_detections += 1
        tile_dets: list[dict] = []

        if not (len(boxes) == len(cls_ids) == len(scores)):
            logger.warning("    tile r=%d c=%d output length mismatch", r, c)
            continue
        for box, cls_id, score in zip(boxes, cls_ids, scores):  # noqa: B905
            ty1, tx1, ty2, tx2 = box
            # Map back to global coords (clip to actual tile size, not padded)
            gx1 = int(x + min(tx1, w_eff))
            gy1 = int(y + min(ty1, h_eff))
            gx2 = int(x + min(tx2, w_eff))
            gy2 = int(y + min(ty2, h_eff))

            # Filter 1: drop bboxes mostly in the padded white region
            if gx2 - gx1 < 5 or gy2 - gy1 < 5:
                continue

            # Filter 2: minimum size (real cells are ~80-160 px at native res)
            bbox_w = gx2 - gx1
            bbox_h = gy2 - gy1
            if bbox_w < min_bbox_size or bbox_h < min_bbox_size:
                report.dropped_too_small += 1
                continue

            # Filter 3: bbox center must be inside tissue mask
            if tissue_mask is not None:
                cy = (gy1 + gy2) // 2
                cx = (gx1 + gx2) // 2
                if 0 <= cy < h and 0 <= cx < w and tissue_mask[cy, cx] == 0:
                    report.dropped_outside_tissue += 1
                    continue

            yolo_cls = int(cls_id) - 1  # mrcnn 1-indexed -> yolo 0-indexed
            if yolo_cls < 0 or yolo_cls >= 5:
                continue

            det = Detection(
                class_id=yolo_cls,
                score=float(score),
                x1=gx1, y1=gy1, x2=gx2, y2=gy2,
                tile_row=r, tile_col=c,
            )
            all_detections.append(det)
            tile_dets.append(asdict(det))

        if save_tiles:
            tile_jpg = img_out_dir / f"tile_{r:03d}_{c:03d}.jpg"
            cv2.imwrite(str(tile_jpg), tile_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            (img_out_dir / f"tile_{r:03d}_{c:03d}.json").write_text(
                json.dumps(tile_dets, indent=2), encoding="utf-8"
            )

        tile_metas[f"{r:03d}_{c:03d}"] = {"y": y, "x": x, "n_dets": len(tile_dets)}

    # Aggregate stats
    report.total_detections = len(all_detections)
    if all_detections:
        report.avg_confidence = round(
            float(sum(d.score for d in all_detections) / len(all_detections)), 4
        )
    for det in all_detections:
        report.detections_per_class[det.class_id] = (
            report.detections_per_class.get(det.class_id, 0) + 1
        )

    # Write YOLO det labels (global coords, normalized to full image size)
    yolo_lines = []
    for d in all_detections:
        cx = ((d.x1 + d.x2) / 2.0) / w
        cy = ((d.y1 + d.y2) / 2.0) / h
        bw = (d.x2 - d.x1) / w
        bh = (d.y2 - d.y1) / h
        # Clamp
        cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
        bw, bh = max(0.0, min(1.0, bw)), max(0.0, min(1.0, bh))
        yolo_lines.append(f"{d.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    (output_dir / f"{stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")
    (output_dir / f"{stem}.report.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )

    # Overview image (downsampled with global bboxes drawn)
    if save_overview:
        max_overview = 1600
        scale = min(1.0, max_overview / max(h, w))
        overview = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image.copy()
        # Class colors (BGR)
        colors = {
            0: (0, 50, 255), 1: (0, 255, 0), 2: (255, 0, 0),
            3: (255, 255, 0), 4: (0, 255, 255),
        }
        for d in all_detections:
            x1 = int(d.x1 * scale)
            y1 = int(d.y1 * scale)
            x2 = int(d.x2 * scale)
            y2 = int(d.y2 * scale)
            color = colors.get(d.class_id, (255, 255, 255))
            cv2.rectangle(overview, (x1, y1), (x2, y2), color, 1)
        cv2.imwrite(str(img_out_dir / "_overview.jpg"), overview, [cv2.IMWRITE_JPEG_QUALITY, 85])

    logger.info(
        "    -> %d kept dets across %d/%d tiles  (dropped %d small, %d outside-tissue)  avg conf %.3f",
        report.total_detections, report.n_tiles_with_detections, report.n_tiles,
        report.dropped_too_small, report.dropped_outside_tissue,
        report.avg_confidence,
    )
    return report


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--image", type=Path, help="Single image to process")
    p.add_argument("--image-dir", type=Path, help="Directory of images to process")
    p.add_argument("--output", type=Path, default=Path("runs/tiled_detect"))
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--tile-stride", type=int, default=384,
                   help="Default 384 = 25%% overlap with --tile-size 512")
    p.add_argument("--conf-min", type=float, default=0.5)
    p.add_argument("--no-skip-blank", action="store_true",
                   help="Run inference on every tile, even uniform-background ones")
    p.add_argument("--save-tiles", action="store_true",
                   help="Save each tile as a JPG (large output dir)")
    p.add_argument("--no-overview", action="store_true",
                   help="Skip writing the downsampled overview JPG")
    p.add_argument("--min-bbox-size", type=int, default=25,
                   help="Drop bboxes whose width or height < this (default 25 px)")
    p.add_argument("--no-tissue-mask", action="store_true",
                   help="Skip the tissue-mask filter (keeps all in-bounds detections)")
    p.add_argument("--limit", type=int, default=0,
                   help="If using --image-dir, only process this many (0 = all)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.weights.is_file():
        logger.error("Weights not found: %s", args.weights)
        return 2

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

    # Lazy heavy import (legacy venv only)
    from mrcnn.model import MaskRCNN

    logger.info("Loading mrcnn from %s ...", args.weights)
    cfg = _build_inference_config(args.tile_size, args.conf_min)
    model = MaskRCNN(mode="inference", model_dir=str(args.output / "_logs"), config=cfg)
    model.load_weights(str(args.weights), by_name=True)
    logger.info("Model loaded.")

    aggregate = {
        "n_images": 0,
        "n_total_dets": 0,
        "per_image": [],
    }

    for img_path in images:
        try:
            report = detect_one_image(
                model,
                image_path=img_path,
                output_dir=args.output,
                tile_size=args.tile_size,
                tile_stride=args.tile_stride,
                skip_blank_tiles=not args.no_skip_blank,
                save_tiles=args.save_tiles,
                save_overview=not args.no_overview,
                min_bbox_size=args.min_bbox_size,
                use_tissue_mask=not args.no_tissue_mask,
            )
        except Exception as exc:
            logger.error("Failed on %s: %s", img_path.name, exc)
            continue

        aggregate["n_images"] += 1
        aggregate["n_total_dets"] += report.total_detections
        aggregate["per_image"].append({
            "image": report.image_name,
            "n_tiles": report.n_tiles,
            "tiles_with_dets": report.n_tiles_with_detections,
            "total_dets": report.total_detections,
            "avg_conf": report.avg_confidence,
        })

    (args.output / "_aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    logger.info(
        "Done: %d images, %d total detections.",
        aggregate["n_images"], aggregate["n_total_dets"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
