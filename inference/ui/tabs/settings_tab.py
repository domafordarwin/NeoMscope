"""Settings tab — runtime configuration shared across all tabs.

Settings are not persisted; they reset each launch. Future v2 may add a
~/.config/neomscope.json store. For now, simplicity wins.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from inference.ui.state import AppController


class SettingsTab(QWidget):
    """Runtime configuration."""

    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self._controller = controller
        self._build()
        self._load_from_state()

    def _build(self) -> None:
        # Model
        model_group = QGroupBox("모델")
        model_lay = QGridLayout(model_group)
        model_lay.setContentsMargins(12, 18, 12, 12)
        model_lay.setHorizontalSpacing(8)

        model_lay.addWidget(QLabel("HEF 파일"), 0, 0)
        self.hef_edit = QLineEdit()
        hef_btn = QPushButton("…")
        hef_btn.setMaximumWidth(40)
        hef_btn.clicked.connect(self._pick_hef)
        model_lay.addWidget(self.hef_edit, 0, 1)
        model_lay.addWidget(hef_btn, 0, 2)

        model_lay.addWidget(QLabel("입력 사이즈"), 1, 0)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(128, 2048)
        self.imgsz_spin.setSingleStep(32)
        model_lay.addWidget(self.imgsz_spin, 1, 1)

        # Camera
        cam_group = QGroupBox("카메라")
        cam_lay = QGridLayout(cam_group)
        cam_lay.setContentsMargins(12, 18, 12, 12)
        cam_lay.addWidget(QLabel("디바이스"), 0, 0)
        self.cam_edit = QLineEdit()
        self.cam_edit.setPlaceholderText("0  또는  /dev/video0  또는  folder:path")
        cam_lay.addWidget(self.cam_edit, 0, 1)

        # Detection
        det_group = QGroupBox("검출 기본값")
        det_lay = QGridLayout(det_group)
        det_lay.setContentsMargins(12, 18, 12, 12)
        det_lay.setHorizontalSpacing(8)

        det_lay.addWidget(QLabel("신뢰도 임계값"), 0, 0)
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_value = QLabel("0.25")
        self.conf_value.setMinimumWidth(36)
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_value.setText(f"{v / 100:.2f}")
        )
        det_lay.addWidget(self.conf_slider, 0, 1)
        det_lay.addWidget(self.conf_value, 0, 2)

        det_lay.addWidget(QLabel("NMS IoU"), 1, 0)
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setRange(10, 90)
        self.iou_value = QLabel("0.50")
        self.iou_value.setMinimumWidth(36)
        self.iou_slider.valueChanged.connect(
            lambda v: self.iou_value.setText(f"{v / 100:.2f}")
        )
        det_lay.addWidget(self.iou_slider, 1, 1)
        det_lay.addWidget(self.iou_value, 1, 2)

        # Output dirs
        out_group = QGroupBox("출력 디렉터리")
        out_lay = QGridLayout(out_group)
        out_lay.setContentsMargins(12, 18, 12, 12)
        out_lay.setHorizontalSpacing(8)
        out_lay.setVerticalSpacing(6)

        out_lay.addWidget(QLabel("촬영"), 0, 0)
        self.cap_edit = QLineEdit()
        cap_btn = QPushButton("…")
        cap_btn.setMaximumWidth(40)
        cap_btn.clicked.connect(lambda: self._pick_dir(self.cap_edit, "촬영 폴더"))
        out_lay.addWidget(self.cap_edit, 0, 1)
        out_lay.addWidget(cap_btn, 0, 2)

        out_lay.addWidget(QLabel("Live 녹화"), 1, 0)
        self.live_edit = QLineEdit()
        live_btn = QPushButton("…")
        live_btn.setMaximumWidth(40)
        live_btn.clicked.connect(lambda: self._pick_dir(self.live_edit, "Live 녹화 폴더"))
        out_lay.addWidget(self.live_edit, 1, 1)
        out_lay.addWidget(live_btn, 1, 2)

        # Apply / Reset
        actions = QHBoxLayout()
        self.apply_btn = QPushButton("적용")
        self.apply_btn.setMinimumHeight(32)
        self.apply_btn.clicked.connect(self._apply)
        self.reset_btn = QPushButton("초기화")
        self.reset_btn.setMinimumHeight(32)
        self.reset_btn.clicked.connect(self._load_from_state)
        actions.addStretch()
        actions.addWidget(self.reset_btn)
        actions.addWidget(self.apply_btn)

        self.status = QLabel("")
        self.status.setObjectName("section")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)
        outer.addWidget(model_group)
        outer.addWidget(cam_group)
        outer.addWidget(det_group)
        outer.addWidget(out_group)
        outer.addLayout(actions)
        outer.addWidget(self.status)
        outer.addStretch()

    def _pick_hef(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "HEF 파일 선택", "", "Hailo HEF (*.hef)")
        if f:
            self.hef_edit.setText(f)

    def _pick_dir(self, line_edit: QLineEdit, title: str) -> None:
        d = QFileDialog.getExistingDirectory(self, title, line_edit.text() or "")
        if d:
            line_edit.setText(d)

    def _load_from_state(self) -> None:
        s = self._controller.settings
        self.hef_edit.setText(str(s.hef_path))
        self.imgsz_spin.setValue(s.image_size)
        self.cam_edit.setText(s.camera_spec)
        self.conf_slider.setValue(int(s.conf_threshold * 100))
        self.iou_slider.setValue(int(s.iou_threshold * 100))
        self.cap_edit.setText(str(s.output_dir_capture))
        self.live_edit.setText(str(s.output_dir_live))
        self.status.setText("설정 불러옴")

    def _apply(self) -> None:
        self._controller.update_settings(
            hef_path=Path(self.hef_edit.text()),
            image_size=self.imgsz_spin.value(),
            camera_spec=self.cam_edit.text() or "0",
            conf_threshold=self.conf_slider.value() / 100,
            iou_threshold=self.iou_slider.value() / 100,
            output_dir_capture=Path(self.cap_edit.text()),
            output_dir_live=Path(self.live_edit.text()),
        )
        self.status.setText("✅ 적용됨")
