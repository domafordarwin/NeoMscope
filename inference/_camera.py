"""Camera abstraction for inference scripts.

Hides the difference between USB UVC webcams (cv2.VideoCapture), the Pi
Camera (picamera2), and a folder of pre-recorded images. The three
inference entry points (live/batch/capture_and_detect) only see a
uniform iterator-of-frames interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("camera")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


class FrameSource(ABC):
    """Common interface across live and offline frame sources."""

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return (ok, frame). frame is BGR uint8 of shape (H, W, 3) when ok."""

    @abstractmethod
    def close(self) -> None:
        ...

    def frames(self) -> Iterator[np.ndarray]:
        """Generator that yields BGR frames until the source is exhausted."""
        while True:
            ok, frame = self.read()
            if not ok or frame is None:
                return
            yield frame

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class CV2VideoCapture(FrameSource):
    """Wraps `cv2.VideoCapture` for USB UVC webcams or video files."""

    def __init__(self, source: int | str, width: int = 1280, height: int = 720) -> None:
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera/video source: {source!r}. "
                f"Check 'lsof /dev/video0' on Linux or that the index is correct."
            )
        # Best-effort hinting; many USB cams ignore these
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Opened cv2 source %r at %dx%d", source, actual_w, actual_h)

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._cap.read()
        return ok, frame if ok else None

    def close(self) -> None:
        self._cap.release()


class ImageFolder(FrameSource):
    """Iterates over .jpg/.png files in one or more directories.

    Useful for offline batch inference and for replaying the captured
    sample images on a dev PC without a webcam.
    """

    def __init__(self, dirs: list[Path]) -> None:
        self._paths: list[Path] = []
        for d in dirs:
            if not d.is_dir():
                raise FileNotFoundError(f"Image folder not found: {d}")
            self._paths.extend(
                sorted(p for p in d.iterdir() if p.suffix in IMAGE_EXTS and p.is_file())
            )
        if not self._paths:
            raise RuntimeError(f"No images found in {dirs}")
        self._index = 0
        logger.info("ImageFolder source: %d files from %s", len(self._paths), dirs)

    @property
    def current_path(self) -> Path | None:
        """Path of the most recently returned frame (for filename-tracked output)."""
        if 0 < self._index <= len(self._paths):
            return self._paths[self._index - 1]
        return None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._paths):
            return False, None
        path = self._paths[self._index]
        self._index += 1
        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("Could not decode %s, skipping", path)
            return self.read()
        return True, frame

    def close(self) -> None:
        pass


def open_source(spec: str) -> FrameSource:
    """Resolve a CLI-style camera spec to a FrameSource.

    Examples:
        '/dev/video0'    → CV2VideoCapture(0) on Linux (after parsing)
        '0'              → CV2VideoCapture(0)
        'path/to/video.mp4' → CV2VideoCapture('path/to/video.mp4')
        'folder:datasets/onioncell/images/test' → ImageFolder([...])
    """
    if spec.startswith("folder:"):
        path = Path(spec.split(":", 1)[1])
        return ImageFolder([path])

    # Linux device path /dev/videoN — keep as-is (cv2 accepts string), but try
    # int conversion first since on Linux cv2.VideoCapture(0) is more reliable
    if spec.isdigit():
        return CV2VideoCapture(int(spec))

    return CV2VideoCapture(spec)
