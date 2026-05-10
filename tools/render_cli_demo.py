"""Render a representative screenshot of the CLI-era interface.

Simulates exactly what `neomscope-live --camera /dev/video0` would show:
a single full-window image with detection bboxes drawn on top, the FPS
counter in the corner, and the per-class count summary panel — the same
overlay live_detect.py / capture_and_detect.py produce when running.

Used to document the v0/v1 era (cv2.imshow single window) for the
interface-evolution report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from inference.postprocess import render_overlay, render_summary
from inference.types import CLASS_COLORS, Detection


def _mock_detections(h: int, w: int, n: int = 12, seed: int = 7) -> list[Detection]:
    rng = np.random.default_rng(seed)
    plan = [(0, 2), (1, 4), (2, 1), (3, 3), (4, 2)]  # 12 dets matching the design comp
    dets: list[Detection] = []
    pad_x = w // 8
    pad_y = h // 8
    for class_id, count in plan:
        for _ in range(count):
            cx = rng.integers(pad_x, w - pad_x)
            cy = rng.integers(pad_y, h - pad_y)
            size = rng.integers(min(h, w) // 14, min(h, w) // 9)
            dets.append(Detection(
                bbox=(int(cx - size), int(cy - size), int(cx + size), int(cy + size)),
                class_id=int(class_id),
                confidence=float(0.62 + 0.32 * rng.random()),
            ))
    return dets


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--input",
        type=Path,
        default=Path("captured_raw_images/2022-07-18_18-02-29.jpg"),
        help="Source image (microscope view)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("docs/03-analysis/screenshots/v0/cli_live_simulation.png"),
        help="Output PNG path",
    )
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=600)
    args = p.parse_args()

    bgr = cv2.imread(str(args.input))
    if bgr is None:
        raise SystemExit(f"Could not read {args.input}")

    # Resize source to target window size, letterbox-style on black bg
    h, w = bgr.shape[:2]
    scale = min(args.width / w, args.height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((args.height, args.width, 3), 25, dtype=np.uint8)
    pad_x = (args.width - new_w) // 2
    pad_y = (args.height - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # Place mock detections in canvas coords so they sit on actual cells
    dets = _mock_detections(args.height, args.width)
    annotated = render_overlay(canvas, dets)
    annotated = render_summary(annotated, dets)

    # FPS + counts text exactly like live_detect.py
    fps_text = f"FPS: 28.3  Det: {len(dets)}"
    cv2.putText(
        annotated, fps_text,
        (12, args.height - 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
    )
    # OS-style title bar simulation (top strip)
    cv2.rectangle(annotated, (0, 0), (args.width, 24), (60, 60, 60), -1)
    cv2.putText(
        annotated,
        "NeoMscope Live  -  python -m inference.live_detect --camera /dev/video0",
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), annotated)
    print(f"Wrote {args.output} ({args.width}x{args.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
