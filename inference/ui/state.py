"""Shared GUI state — settings + helpers used by multiple tabs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from inference.types import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU_THRESHOLD,
)


@dataclass
class AppSettings:
    """Runtime configuration shared across tabs.

    Persistence is intentionally skipped for v1 — settings reset each launch
    and are configured via the Settings tab.
    """

    hef_path: Path = Path("models/hef/best.hef")
    camera_spec: str = "0"
    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    image_size: int = DEFAULT_IMAGE_SIZE
    output_dir_capture: Path = Path("detection_results_captured")
    output_dir_live: Path = Path("detection_results_live")
    fallback_image_dir: Path = Path("captured_raw_images")
    fallback_demo_image: Path = Path("captured_raw_images/2022-07-18_18-02-29.jpg")


class AppController(QObject):
    """Lightweight pub-sub for cross-tab signals.

    e.g. when Settings changes the conf threshold, Live tab listens and
    updates the slider.
    """

    settings_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.settings = AppSettings()

    def update_settings(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            if hasattr(self.settings, k):
                setattr(self.settings, k, v)
        self.settings_changed.emit()
