"""Pair bootstrap labels with their source images and resize for Roboflow upload.

Roboflow's web uploader expects each image to have a matching `.txt` label
file in the same directory (same stem). The bootstrap output keeps labels
in `datasets/onioncell/labels/auto/`, separate from the gigantic source
images in `raws/JPEG_Export_Data/` (35-40 MB each, totaling 3.2 GB).

This tool:
  1. Walks the label directory.
  2. Looks up the source image (in one or more search dirs) by stem.
  3. Resizes the image to a max long-edge size (default 1280) and
     re-encodes as JPEG (default quality 85).
  4. Copies the label `.txt` next to the resized image.
  5. Optionally zips everything for one-shot upload.

Output is a flat directory of pairs:
    out/
    ├── IMG_001.jpg   (resized)
    ├── IMG_001.txt   (label, unchanged — YOLO coords are normalized)
    ├── IMG_002.jpg
    └── IMG_002.txt

Run from the project root in the main venv (Python 3.11):
    python tools/package_for_roboflow.py \
        --labels datasets/onioncell/labels/auto \
        --images raws/JPEG_Export_Data captured_raw_images \
        --output runs/roboflow-upload \
        --max-size 1280 \
        --jpeg-quality 85 \
        --zip
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2

logger = logging.getLogger("package_for_roboflow")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


@dataclass
class PackageReport:
    paired: int = 0
    skipped_no_image: list[str] = field(default_factory=list)
    skipped_unreadable: list[str] = field(default_factory=list)
    output_size_mb: float = 0.0


def _build_image_index(image_dirs: list[Path]) -> dict[str, Path]:
    """Map each image stem to its source path. Later dirs override earlier ones."""
    index: dict[str, Path] = {}
    for d in image_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"Image directory not found: {d}")
        for ext in IMAGE_EXTS:
            for p in d.glob(f"*{ext}"):
                index[p.stem] = p
    return index


def _resize_to_max(image: cv2.typing.MatLike, max_size: int) -> cv2.typing.MatLike:
    """Resize so the long edge <= max_size, preserving aspect. No-op if already small enough."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_size:
        return image
    scale = max_size / longest
    new_w = round(w * scale)
    new_h = round(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def package(
    labels_dir: Path,
    image_dirs: list[Path],
    output_dir: Path,
    max_size: int = 1280,
    jpeg_quality: int = 85,
) -> PackageReport:
    """Bundle (image, label) pairs into a Roboflow-ready directory."""
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    index = _build_image_index(image_dirs)
    label_files = sorted(labels_dir.glob("*.txt"))

    if not label_files:
        raise RuntimeError(f"No .txt labels in {labels_dir}")

    report = PackageReport()
    logger.info("Pairing %d label files against %d images...", len(label_files), len(index))

    for lbl_path in label_files:
        stem = lbl_path.stem
        src_img_path = index.get(stem)
        if src_img_path is None:
            report.skipped_no_image.append(stem)
            continue

        image = cv2.imread(str(src_img_path))
        if image is None:
            report.skipped_unreadable.append(src_img_path.name)
            continue

        resized = _resize_to_max(image, max_size)
        out_jpg = output_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_jpg), resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        # Labels are scale-invariant (YOLO normalized coords), just copy
        out_txt = output_dir / f"{stem}.txt"
        shutil.copyfile(lbl_path, out_txt)

        report.paired += 1
        logger.info("  %s: %s → %s", stem, src_img_path.name, out_jpg.name)

    # Tally output size
    report.output_size_mb = sum(p.stat().st_size for p in output_dir.iterdir()) / (1024**2)

    return report


def make_zip(output_dir: Path, zip_path: Path) -> int:
    """Zip the contents of output_dir (flat, no parent directory)."""
    files = sorted(p for p in output_dir.iterdir() if p.is_file())
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in files:
            zf.write(p, arcname=p.name)
    return len(files)


def _print_report(report: PackageReport, output_dir: Path, zip_path: Path | None) -> None:
    print()
    print("=" * 60)
    print("Roboflow Package Report")
    print("=" * 60)
    print(f"Pairs written:       {report.paired}")
    print(f"Output dir:          {output_dir}")
    print(f"Output size:         {report.output_size_mb:.1f} MB")
    if zip_path:
        zip_mb = zip_path.stat().st_size / (1024**2)
        print(f"Zip:                 {zip_path} ({zip_mb:.1f} MB)")
    if report.skipped_no_image:
        print(f"⚠️  {len(report.skipped_no_image)} labels with no matching image:")
        for s in report.skipped_no_image[:5]:
            print(f"    {s}")
        if len(report.skipped_no_image) > 5:
            print(f"    ... and {len(report.skipped_no_image) - 5} more")
    if report.skipped_unreadable:
        print(f"⚠️  {len(report.skipped_unreadable)} unreadable images skipped")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--labels", type=Path, required=True, help="Directory of YOLO .txt labels")
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories containing source images",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--max-size",
        type=int,
        default=1280,
        help="Resize so long edge <= this (default 1280)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        choices=range(50, 101),
        help="JPEG quality 50-100 (default 85)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also produce <output>.zip for one-click upload",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(message)s")

    try:
        report = package(
            labels_dir=args.labels,
            image_dirs=args.images,
            output_dir=args.output,
            max_size=args.max_size,
            jpeg_quality=args.jpeg_quality,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 2

    zip_path = None
    if args.zip:
        zip_path = args.output.with_suffix(".zip")
        n = make_zip(args.output, zip_path)
        logger.info("Wrote %s (%d files)", zip_path, n)

    _print_report(report, args.output, zip_path)
    return 0 if report.paired > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
