"""Reusable widgets for the NeoMscope GUI."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from inference.types import CLASS_COLORS, CLASS_NAMES


def bgr_to_pixmap(bgr: np.ndarray, target_w: int, target_h: int) -> QPixmap:
    """Convert OpenCV BGR ndarray to QPixmap, scaled to fit target preserving aspect."""
    h, w = bgr.shape[:2]
    if h == 0 or w == 0:
        return QPixmap()
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class LegendChip(QWidget):
    """Class chip — colored side bar + class name + count, in one row."""

    def __init__(self, class_id: int, count: int = 0) -> None:
        super().__init__()
        self.class_id = class_id
        bgr = CLASS_COLORS[class_id]
        r, g, b = bgr[2], bgr[1], bgr[0]

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        bar = QLabel()
        bar.setFixedSize(4, 18)
        bar.setStyleSheet(f"background: rgb({r},{g},{b}); border-radius: 2px;")

        self.text = QLabel(f"{CLASS_NAMES[class_id]}  {count}")
        self.text.setStyleSheet("font-size: 11px; font-weight: 600; color: #1f2328;")

        lay.addWidget(bar)
        lay.addWidget(self.text)
        lay.addStretch()

    def set_count(self, n: int) -> None:
        self.text.setText(f"{CLASS_NAMES[self.class_id]}  {n}")


class RoundButton(QPushButton):
    """Round button used for capture/record/save controls."""

    def __init__(self, text: str, variant: str = "default") -> None:
        super().__init__(text)
        self.setObjectName("round_record" if variant == "record" else "round")


def labeled_button(text: str, sub: str, variant: str = "default") -> QWidget:
    """Capture-control button with sub-label below."""
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    btn = RoundButton(text, variant)
    sub_lab = QLabel(sub)
    sub_lab.setObjectName("section")
    sub_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(sub_lab)
    box.button = btn  # type: ignore[attr-defined]  # expose for connect()
    return box
