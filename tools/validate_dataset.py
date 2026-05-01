"""Validate a YOLO det dataset for training readiness.

Checks:
    1. All label coordinates are normalized (in [0, 1]).
    2. All bboxes have positive width and height.
    3. Every image has a corresponding .txt label (or empty .txt for no objects).
    4. Class distribution is reasonably balanced (warn if max:min > 10).
    5. No more than 10% of images have empty labels.
    6. Class indices are within [0, NUM_CLASSES).

CLI:
    python tools/validate_dataset.py datasets/onioncell
    python tools/validate_dataset.py datasets/onioncell --strict
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from inference.types import CLASS_NAMES, NUM_CLASSES

logger = logging.getLogger("validate_dataset")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@dataclass
class ValidationReport:
    total_images: int = 0
    total_labels: int = 0
    total_annotations: int = 0
    images_without_labels: list[str] = field(default_factory=list)
    labels_without_images: list[str] = field(default_factory=list)
    empty_label_files: list[str] = field(default_factory=list)
    invalid_lines: list[tuple[str, int, str]] = field(default_factory=list)
    class_counts: Counter[int] = field(default_factory=Counter)

    @property
    def passed(self) -> bool:
        return (
            not self.images_without_labels
            and not self.labels_without_images
            and not self.invalid_lines
        )

    def empty_ratio(self) -> float:
        return (len(self.empty_label_files) / self.total_images) if self.total_images else 0.0

    def class_imbalance(self) -> float:
        if not self.class_counts:
            return 0.0
        values = list(self.class_counts.values())
        return max(values) / max(min(values), 1)


def _list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.rglob("*") if p.suffix in IMAGE_EXTS and p.is_file())


def _list_labels(labels_dir: Path) -> list[Path]:
    return sorted(labels_dir.rglob("*.txt"))


def _validate_line(line: str) -> tuple[bool, str]:
    """Validate a single YOLO det line: '<cls> <cx> <cy> <w> <h>'."""
    parts = line.strip().split()
    if len(parts) != 5:
        return False, f"expected 5 fields, got {len(parts)}"

    try:
        cls = int(parts[0])
        cx, cy, w, h = (float(p) for p in parts[1:])
    except ValueError as exc:
        return False, f"could not parse numbers: {exc}"

    if not 0 <= cls < NUM_CLASSES:
        return False, f"class {cls} out of range [0, {NUM_CLASSES})"

    for name, val in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
        if not 0.0 <= val <= 1.0:
            return False, f"{name}={val} not in [0, 1]"

    if w <= 0 or h <= 0:
        return False, f"degenerate bbox: w={w}, h={h}"

    return True, ""


def validate(dataset_root: Path) -> ValidationReport:
    """Validate a YOLO det dataset rooted at <dataset_root>.

    Expected layout (auto-detects either flat 'images/'+'labels/' or split):
        <root>/images/{train,val,test}/*.jpg
        <root>/labels/{train,val,test}/*.txt
    or
        <root>/images/*.jpg
        <root>/labels/*.txt
    """
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {images_dir} and {labels_dir} to exist"
        )

    report = ValidationReport()
    images = _list_images(images_dir)
    labels = _list_labels(labels_dir)

    # Build a stem-based map ignoring split subdirs
    image_stems = {p.stem: p for p in images}
    label_stems = {p.stem: p for p in labels}

    report.total_images = len(images)
    report.total_labels = len(labels)

    # Cross-reference
    for stem, img_path in image_stems.items():
        if stem not in label_stems:
            report.images_without_labels.append(img_path.name)
    for stem, lbl_path in label_stems.items():
        if stem not in image_stems:
            report.labels_without_images.append(lbl_path.name)

    # Validate label content
    for lbl_path in labels:
        text = lbl_path.read_text(encoding="utf-8").strip()
        if not text:
            report.empty_label_files.append(lbl_path.name)
            continue

        for lineno, raw_line in enumerate(text.splitlines(), start=1):
            if not raw_line.strip():
                continue
            ok, err = _validate_line(raw_line)
            if not ok:
                report.invalid_lines.append((lbl_path.name, lineno, err))
                continue

            cls = int(raw_line.split()[0])
            report.class_counts[cls] += 1
            report.total_annotations += 1

    return report


def _print_report(report: ValidationReport) -> None:
    print()
    print("=" * 60)
    print("Dataset Validation Report")
    print("=" * 60)
    print(f"Images:              {report.total_images}")
    print(f"Labels:              {report.total_labels}")
    print(f"Total annotations:   {report.total_annotations}")
    print(f"Empty labels:        {len(report.empty_label_files)} "
          f"({report.empty_ratio() * 100:.1f}%)")
    print()
    print("Class distribution:")
    for cls_id in range(NUM_CLASSES):
        count = report.class_counts.get(cls_id, 0)
        print(f"  {cls_id} {CLASS_NAMES[cls_id]:>6}:  {count}")
    print(f"Imbalance ratio:     {report.class_imbalance():.1f}:1")
    print()

    if report.images_without_labels:
        print(f"⚠️  {len(report.images_without_labels)} images without labels:")
        for name in report.images_without_labels[:5]:
            print(f"    {name}")
        if len(report.images_without_labels) > 5:
            print(f"    ... and {len(report.images_without_labels) - 5} more")
        print()

    if report.labels_without_images:
        print(f"⚠️  {len(report.labels_without_images)} labels without images:")
        for name in report.labels_without_images[:5]:
            print(f"    {name}")
        print()

    if report.invalid_lines:
        print(f"❌ {len(report.invalid_lines)} invalid label lines:")
        for filename, lineno, err in report.invalid_lines[:10]:
            print(f"    {filename}:{lineno}  {err}")
        if len(report.invalid_lines) > 10:
            print(f"    ... and {len(report.invalid_lines) - 10} more")
        print()

    print("=" * 60)
    print("PASSED ✅" if report.passed else "FAILED ❌")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset_root", type=Path, help="Path to dataset root")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings (imbalance, empty ratio) as failures",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(message)s")

    try:
        report = validate(args.dataset_root)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    _print_report(report)

    if not report.passed:
        return 1
    if args.strict:
        if report.class_imbalance() > 10:
            logger.error("Strict mode: class imbalance > 10:1")
            return 1
        if report.empty_ratio() > 0.10:
            logger.error("Strict mode: empty label ratio > 10%%")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
