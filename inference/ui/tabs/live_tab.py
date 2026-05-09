"""Live tab — real-time camera feed + Hailo inference pipeline.

Uses the real FrameSource abstraction (CV2VideoCapture / ImageFolder)
and HailoInferencePipeline(mock=True) until a compiled HEF is available.

If no camera is detected, falls back to the captured_raw_images/ folder
in slow-cycle mode so the GUI still demos meaningfully on the dev PC.
"""

from __future__ import annotations

import logging
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from inference._camera import CV2VideoCapture, FrameSource, ImageFolder
from inference.pipeline import HailoInferencePipeline
from inference.postprocess import render_overlay
from inference.types import CLASS_NAMES, Detection, PipelineConfig, count_by_class
from inference.ui.state import AppController
from inference.ui.widgets import LegendChip, bgr_to_pixmap, labeled_button

logger = logging.getLogger("ui.live")

VIDEO_W, VIDEO_H = 600, 380
TARGET_FPS = 15  # safe for mock pipeline + Pi 5 NPU later


class LiveTab(QWidget):
    """Real-time detection tab."""

    capture_requested = Signal(np.ndarray, list)  # frame, detections

    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self._controller = controller
        self._source: FrameSource | None = None
        self._pipeline: HailoInferencePipeline | None = None
        self._paused = False
        self._fallback_idx = 0
        self._build()
        self._init_runtime()

        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / TARGET_FPS))
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        controller.settings_changed.connect(self._on_settings_changed)

    # ----- UI construction -----
    def _build(self) -> None:
        # Video pane
        self.video_label = QLabel()
        self.video_label.setObjectName("video")
        self.video_label.setFixedSize(VIDEO_W, VIDEO_H)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("카메라 초기화 중…")
        self.video_label.setStyleSheet(
            "QLabel#video { color: #9198a1; }"
        )

        # Detection results panel
        result_group = QGroupBox("검출 결과")
        result_lay = QVBoxLayout(result_group)
        result_lay.setSpacing(6)
        result_lay.setContentsMargins(12, 18, 12, 12)

        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("총 검출"))
        count_row.addStretch()
        self.count_label = QLabel("0")
        self.count_label.setObjectName("count_value")
        count_row.addWidget(self.count_label)
        result_lay.addLayout(count_row)

        self.chips: list[LegendChip] = []
        for i in range(5):
            chip = LegendChip(i, 0)
            self.chips.append(chip)
            result_lay.addWidget(chip)

        conf_label = QLabel("신뢰도 임계값")
        conf_label.setObjectName("section")
        result_lay.addWidget(conf_label)

        slider_row = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(int(self._controller.settings.conf_threshold * 100))
        self.conf_value = QLabel(f"{self._controller.settings.conf_threshold:.2f}")
        self.conf_value.setMinimumWidth(36)
        self.conf_slider.valueChanged.connect(self._on_conf_change)
        slider_row.addWidget(self.conf_slider)
        slider_row.addWidget(self.conf_value)
        result_lay.addLayout(slider_row)

        # FPS display
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("section")
        result_lay.addWidget(self.fps_label)

        # Tools panel
        tools_group = QGroupBox("도구")
        tools_lay = QVBoxLayout(tools_group)
        tools_lay.setSpacing(6)
        tools_lay.setContentsMargins(12, 18, 12, 12)
        self.btn_zoom_in = QPushButton("확대")
        self.btn_zoom_out = QPushButton("축소")
        self.btn_reset = QPushButton("초기화")
        for b in (self.btn_zoom_in, self.btn_zoom_out, self.btn_reset):
            b.setObjectName("tool")
            tools_lay.addWidget(b)
        self.btn_reset.clicked.connect(self._reset_view)

        right_col = QVBoxLayout()
        right_col.setSpacing(6)
        right_col.addWidget(result_group)
        right_col.addWidget(tools_group)
        right_col.addStretch()

        top_row = QHBoxLayout()
        top_row.addWidget(self.video_label)
        right_wrap = QWidget()
        right_wrap.setLayout(right_col)
        right_wrap.setFixedWidth(360)
        top_row.addWidget(right_wrap)

        # Capture controls
        cap_group = QGroupBox("촬영 제어")
        cap_lay = QHBoxLayout(cap_group)
        cap_lay.setContentsMargins(12, 18, 12, 12)
        cap_lay.setSpacing(20)

        self.cap_box = labeled_button("📷", "촬영")
        self.rec_box = labeled_button("●", "녹화", variant="record")
        self.save_box = labeled_button("💾", "저장")
        for w in (self.cap_box, self.rec_box, self.save_box):
            cap_lay.addWidget(w, alignment=Qt.AlignmentFlag.AlignCenter)

        self.cap_box.button.clicked.connect(self._on_capture)
        self.save_box.button.clicked.connect(self._on_capture)  # save = capture for v1

        # Mode selector
        mode_group = QGroupBox("검출 모드")
        mode_lay = QHBoxLayout(mode_group)
        mode_lay.setContentsMargins(12, 18, 12, 12)
        mode_lay.setSpacing(20)
        self.mode_buttons = QButtonGroup(self)
        for txt in ("실시간", "정지화면", "일괄"):
            r = QRadioButton(txt)
            mode_lay.addWidget(r)
            self.mode_buttons.addButton(r)
        self.mode_buttons.buttons()[0].setChecked(True)
        self.mode_buttons.buttons()[1].toggled.connect(self._on_pause_toggled)

        # Image adjust
        adj_group = QGroupBox("이미지 조정")
        adj_lay = QGridLayout(adj_group)
        adj_lay.setContentsMargins(12, 18, 12, 12)
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

        outer = QVBoxLayout(self)
        outer.setSpacing(8)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.addLayout(top_row)
        outer.addLayout(bottom_row)

        # FPS bookkeeping
        self._frame_times: list[float] = []

    # ----- Runtime init -----
    def _init_runtime(self) -> None:
        s = self._controller.settings
        # Try real camera first
        try:
            cam = self._open_camera(s.camera_spec)
            self._source = cam
            logger.info("Live tab: opened camera %s", s.camera_spec)
        except Exception as exc:
            logger.warning("Camera open failed (%s); falling back to image folder.", exc)
            try:
                self._source = ImageFolder([s.fallback_image_dir])
            except Exception as exc2:
                logger.error("Fallback image folder failed: %s", exc2)
                self._source = None

        # Pipeline always starts in mock mode (HEF likely not yet compiled)
        cfg = PipelineConfig(
            hef_path=s.hef_path,
            image_size=s.image_size,
            conf_threshold=s.conf_threshold,
            iou_threshold=s.iou_threshold,
        )
        try:
            self._pipeline = HailoInferencePipeline(cfg, mock=True)
        except Exception as exc:
            logger.error("Pipeline init failed: %s", exc)
            self._pipeline = None

    def _open_camera(self, spec: str) -> CV2VideoCapture:
        idx = int(spec) if spec.isdigit() else spec  # type: ignore[assignment]
        return CV2VideoCapture(idx)

    # ----- Frame loop -----
    def _on_tick(self) -> None:
        if self._paused or self._source is None or self._pipeline is None:
            return

        ok, frame = self._source.read()
        if not ok or frame is None:
            # Image folder reached the end — restart
            if isinstance(self._source, ImageFolder):
                self._source = ImageFolder([self._controller.settings.fallback_image_dir])
            return

        # Throttle for static fallback so it doesn't flicker
        if isinstance(self._source, ImageFolder):
            self._fallback_idx += 1
            if self._fallback_idx % 30 != 0:
                # Re-render current frame instead of pulling new
                pass

        try:
            detections = self._pipeline.infer(frame)
        except Exception as exc:
            logger.error("Inference error: %s", exc)
            detections = []

        annotated = render_overlay(frame, detections)
        self._update_view(annotated, detections)

    def _update_view(self, frame: np.ndarray, detections: list[Detection]) -> None:
        pix = bgr_to_pixmap(frame, VIDEO_W, VIDEO_H)
        self.video_label.setPixmap(pix)
        self.video_label.setText("")

        # Update count + chips
        total = len(detections)
        self.count_label.setText(str(total))
        counts = count_by_class(detections)
        for chip in self.chips:
            chip.set_count(counts[CLASS_NAMES[chip.class_id]])

        # Rolling FPS
        from time import perf_counter
        now = perf_counter()
        self._frame_times.append(now)
        if len(self._frame_times) > 30:
            self._frame_times = self._frame_times[-30:]
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0
            self.fps_label.setText(f"FPS: {fps:.1f}")

        self._last_frame = frame
        self._last_detections = detections

    # ----- Event handlers -----
    def _on_conf_change(self, v: int) -> None:
        f = v / 100.0
        self.conf_value.setText(f"{f:.2f}")
        self._controller.update_settings(conf_threshold=f)
        # Hot-update pipeline config
        if self._pipeline is not None:
            self._pipeline._config = PipelineConfig(  # type: ignore[attr-defined]
                hef_path=self._controller.settings.hef_path,
                image_size=self._controller.settings.image_size,
                conf_threshold=f,
                iou_threshold=self._controller.settings.iou_threshold,
            )

    def _on_pause_toggled(self, paused: bool) -> None:
        self._paused = paused

    def _on_capture(self) -> None:
        frame = getattr(self, "_last_frame", None)
        dets = getattr(self, "_last_detections", [])
        if frame is None:
            return
        out_dir = self._controller.settings.output_dir_capture
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = out_dir / f"{ts}.jpg"
        cv2.imwrite(str(out), frame)
        logger.info("Captured %s with %d detections", out, len(dets))
        self.capture_requested.emit(frame, dets)

    def _reset_view(self) -> None:
        self.conf_slider.setValue(50)
        self.exp_slider.setValue(10)
        self.gain_slider.setValue(10)

    def _on_settings_changed(self) -> None:
        s = self._controller.settings
        self.conf_slider.blockSignals(True)
        self.conf_slider.setValue(int(s.conf_threshold * 100))
        self.conf_slider.blockSignals(False)
        self.conf_value.setText(f"{s.conf_threshold:.2f}")
