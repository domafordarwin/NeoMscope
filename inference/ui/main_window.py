"""Main NeoMscope GUI window — 1024x600 single-screen layout.

Tabs: Live / Batch / Archive / Settings.
Live tab is fully implemented (the demo).
The other three tabs render placeholder content.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from inference.types import CLASS_COLORS, CLASS_NAMES, Detection

WINDOW_W, WINDOW_H = 1024, 600
VIDEO_W, VIDEO_H = 600, 380


def bgr_to_pixmap(bgr: np.ndarray, target_w: int, target_h: int) -> QPixmap:
    """Convert OpenCV BGR ndarray to QPixmap, scaled to fit target while preserving aspect."""
    h, w = bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def render_demo_frame(image_path: Path, detections: list[Detection]) -> np.ndarray:
    """Draw bboxes on the source image for the live preview demo."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        bgr = np.full((VIDEO_H, VIDEO_W, 3), 30, dtype=np.uint8)
        cv2.putText(bgr, "no demo image", (40, VIDEO_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return bgr

    for d in detections:
        color = CLASS_COLORS[d.class_id]
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
    return bgr


class LegendChip(QLabel):
    """Small colored chip showing a class name + count."""

    def __init__(self, class_id: int, count: int = 0) -> None:
        super().__init__()
        self.class_id = class_id
        self.set_count(count)
        self.setObjectName("legend_chip")
        bgr = CLASS_COLORS[class_id]
        # Convert BGR -> RGB for stylesheet
        r, g, b = bgr[2], bgr[1], bgr[0]
        self.setStyleSheet(
            f"#legend_chip {{ "
            f"  border-left: 4px solid rgb({r},{g},{b}); "
            f"  background: rgba({r},{g},{b}, 25); "
            f"  border-radius: 4px; "
            f"  padding: 4px 10px; "
            f"  font-size: 11px; "
            f"  font-weight: 600; "
            f"}}"
        )

    def set_count(self, n: int) -> None:
        self.setText(f"{CLASS_NAMES[self.class_id]}: {n}")


class LiveTab(QWidget):
    """Live detection tab — main demo screen."""

    def __init__(self) -> None:
        super().__init__()
        self._build()

    def _build(self) -> None:
        # ===== Top row: Video + Right panel =====
        self.video_label = QLabel()
        self.video_label.setObjectName("video")
        self.video_label.setFixedSize(VIDEO_W, VIDEO_H)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # ----- Right panel: Detection results + Tools -----
        result_group = QGroupBox("검출 결과")
        result_lay = QVBoxLayout(result_group)
        result_lay.setSpacing(8)
        result_lay.setContentsMargins(12, 16, 12, 12)

        # Total count row
        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("검출 수"))
        count_row.addStretch()
        self.count_label = QLabel("12")
        self.count_label.setObjectName("count_value")
        count_row.addWidget(self.count_label)
        result_lay.addLayout(count_row)

        # Per-class chips
        self.chips: list[LegendChip] = []
        chip_grid = QGridLayout()
        chip_grid.setSpacing(4)
        sample_counts = [2, 4, 1, 3, 2]  # demo per-class counts (sum=12)
        for i in range(5):
            chip = LegendChip(i, sample_counts[i])
            self.chips.append(chip)
            chip_grid.addWidget(chip, i // 2, i % 2)
        result_lay.addLayout(chip_grid)

        # Confidence threshold slider
        conf_label = QLabel("신뢰도 임계값")
        conf_label.setObjectName("section")
        result_lay.addWidget(conf_label)

        slider_row = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.conf_value = QLabel("0.50")
        self.conf_value.setMinimumWidth(36)
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_value.setText(f"{v / 100:.2f}")
        )
        slider_row.addWidget(self.conf_slider)
        slider_row.addWidget(self.conf_value)
        result_lay.addLayout(slider_row)

        # ----- Tools box -----
        tools_group = QGroupBox("도구")
        tools_lay = QVBoxLayout(tools_group)
        tools_lay.setSpacing(6)
        tools_lay.setContentsMargins(12, 16, 12, 12)
        for txt in ("확대", "축소", "초기화"):
            b = QPushButton(txt)
            b.setObjectName("tool")
            tools_lay.addWidget(b)

        right_col = QVBoxLayout()
        right_col.addWidget(result_group)
        right_col.addWidget(tools_group)
        right_col.addStretch()

        top_row = QHBoxLayout()
        top_row.addWidget(self.video_label)
        top_row.addLayout(right_col, 1)

        # ===== Bottom row: 3 panels =====
        # Capture control
        cap_group = QGroupBox("촬영 제어")
        cap_lay = QGridLayout(cap_group)
        cap_lay.setContentsMargins(12, 16, 12, 12)
        cap_lay.setHorizontalSpacing(16)

        cap_btn = QPushButton("📷")
        cap_btn.setObjectName("round")
        rec_btn = QPushButton("●")
        rec_btn.setObjectName("round_record")
        save_btn = QPushButton("💾")
        save_btn.setObjectName("round")

        for col, (b, lab) in enumerate([(cap_btn, "촬영"), (rec_btn, "녹화"), (save_btn, "저장")]):
            cap_lay.addWidget(b, 0, col, alignment=Qt.AlignmentFlag.AlignCenter)
            sublabel = QLabel(lab)
            sublabel.setObjectName("section")
            sublabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cap_lay.addWidget(sublabel, 1, col)

        # Detection mode
        mode_group = QGroupBox("검출 모드")
        mode_lay = QHBoxLayout(mode_group)
        mode_lay.setContentsMargins(12, 16, 12, 12)
        mode_lay.setSpacing(20)

        self.mode_buttons = QButtonGroup(self)
        for label_text in ("실시간", "정지화면", "일괄"):
            r = QRadioButton(label_text)
            mode_lay.addWidget(r)
            self.mode_buttons.addButton(r)
        self.mode_buttons.buttons()[0].setChecked(True)

        # Image adjust
        adj_group = QGroupBox("이미지 조정")
        adj_lay = QGridLayout(adj_group)
        adj_lay.setContentsMargins(12, 16, 12, 12)
        adj_lay.setHorizontalSpacing(8)

        adj_lay.addWidget(QLabel("노출 (ms)"), 0, 0)
        self.exp_slider = QSlider(Qt.Orientation.Horizontal)
        self.exp_slider.setRange(1, 100)
        self.exp_slider.setValue(10)
        self.exp_value = QLabel("10.0")
        self.exp_slider.valueChanged.connect(
            lambda v: self.exp_value.setText(f"{v:.1f}")
        )
        adj_lay.addWidget(self.exp_slider, 0, 1)
        adj_lay.addWidget(self.exp_value, 0, 2)

        adj_lay.addWidget(QLabel("게인"), 1, 0)
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(10, 100)
        self.gain_slider.setValue(10)
        self.gain_value = QLabel("1.0")
        self.gain_slider.valueChanged.connect(
            lambda v: self.gain_value.setText(f"{v / 10:.1f}")
        )
        adj_lay.addWidget(self.gain_slider, 1, 1)
        adj_lay.addWidget(self.gain_value, 1, 2)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(cap_group)
        bottom_row.addWidget(mode_group, 1)
        bottom_row.addWidget(adj_group, 1)

        # ===== Outer layout =====
        outer = QVBoxLayout(self)
        outer.setSpacing(8)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.addLayout(top_row)
        outer.addLayout(bottom_row)

    def set_frame(self, bgr: np.ndarray) -> None:
        """Update the live preview with a BGR ndarray."""
        pix = bgr_to_pixmap(bgr, VIDEO_W, VIDEO_H)
        self.video_label.setPixmap(pix)


class PlaceholderTab(QWidget):
    """Stub for tabs not implemented yet."""

    def __init__(self, name: str) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lab = QLabel(f"{name}\n(추후 구현 예정)")
        lab.setStyleSheet("font-size: 16px; color: #57606a;")
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lab)


class MainWindow(QMainWindow):
    """NeoMscope main window — single-screen 1024x600 layout."""

    def __init__(self, demo_image: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("NeoMscope")
        self.setFixedSize(WINDOW_W, WINDOW_H)

        # Title bar
        title = QLabel("NeoMscope — 양파 세포 분열 검출")
        title.setObjectName("title")

        # Tabs
        self.tabs = QTabWidget()
        self.live_tab = LiveTab()
        self.tabs.addTab(self.live_tab, "Live")
        self.tabs.addTab(PlaceholderTab("Batch (일괄 검출)"), "Batch")
        self.tabs.addTab(PlaceholderTab("Archive (저장된 결과)"), "Archive")
        self.tabs.addTab(PlaceholderTab("Settings (설정)"), "Settings")

        outer = QWidget()
        outer_lay = QVBoxLayout(outer)
        outer_lay.setSpacing(0)
        outer_lay.setContentsMargins(0, 0, 0, 0)
        outer_lay.addWidget(title)
        outer_lay.addWidget(self.tabs, 1)
        self.setCentralWidget(outer)

        # Push demo frame if provided
        if demo_image is not None:
            self.show_demo(demo_image)

    def show_demo(self, image_path: Path) -> None:
        """Populate Live tab with the sample image + 12 mock detections."""
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            return

        h, w = bgr.shape[:2]
        rng = np.random.default_rng(seed=42)
        # 12 detections, distribution matches the count chips above
        plan = [(0, 2), (1, 4), (2, 1), (3, 3), (4, 2)]
        dets = []
        for class_id, n in plan:
            for _ in range(n):
                cx = rng.integers(80, w - 80)
                cy = rng.integers(80, h - 80)
                size = rng.integers(35, 65)
                dets.append(Detection(
                    bbox=(cx - size, cy - size, cx + size, cy + size),
                    class_id=class_id,
                    confidence=0.65 + 0.30 * rng.random(),
                ))

        rendered = render_demo_frame(image_path, dets)
        self.live_tab.set_frame(rendered)
