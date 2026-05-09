"""Batch tab — run inference on a folder of images, show progress + summary."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from inference._camera import IMAGE_EXTS
from inference.pipeline import HailoInferencePipeline
from inference.postprocess import render_overlay
from inference.types import CLASS_NAMES, PipelineConfig, count_by_class
from inference.ui.state import AppController

logger = logging.getLogger("ui.batch")


class BatchWorker(QThread):
    """Background thread running mrcnn-style mock inference on all images."""

    progress = Signal(int, int, str)  # done, total, current_filename
    finished_with_summary = Signal(dict)  # {"total_dets": int, "per_class": {...}, "files": int}

    def __init__(self, controller: AppController, input_dir: Path, output_dir: Path) -> None:
        super().__init__()
        self._controller = controller
        self._input = input_dir
        self._output = output_dir
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        s = self._controller.settings
        cfg = PipelineConfig(
            hef_path=s.hef_path,
            image_size=s.image_size,
            conf_threshold=s.conf_threshold,
            iou_threshold=s.iou_threshold,
        )
        pipeline = HailoInferencePipeline(cfg, mock=True)

        files = sorted(p for p in self._input.iterdir()
                       if p.is_file() and p.suffix in IMAGE_EXTS)
        self._output.mkdir(parents=True, exist_ok=True)

        total_dets = 0
        per_class = dict.fromkeys(CLASS_NAMES, 0)

        for i, img_path in enumerate(files):
            if self._cancelled:
                break
            self.progress.emit(i, len(files), img_path.name)
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            try:
                dets = pipeline.infer(frame)
            except Exception as exc:
                logger.error("Inference failed on %s: %s", img_path, exc)
                continue
            annotated = render_overlay(frame, dets)
            out_path = self._output / img_path.name
            cv2.imwrite(str(out_path), annotated)

            total_dets += len(dets)
            for k, v in count_by_class(dets).items():
                per_class[k] += v

        pipeline.close()
        self.progress.emit(len(files), len(files), "완료")
        self.finished_with_summary.emit({
            "total_dets": total_dets,
            "per_class": per_class,
            "files": len(files),
        })


class BatchTab(QWidget):
    """Batch detection tab."""

    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self._controller = controller
        self._worker: BatchWorker | None = None
        self._build()

    def _build(self) -> None:
        # Inputs
        path_group = QGroupBox("경로")
        grid = QGridLayout(path_group)
        grid.setContentsMargins(12, 18, 12, 12)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        grid.addWidget(QLabel("입력 폴더"), 0, 0)
        self.input_edit = QLineEdit("captured_raw_images")
        in_btn = QPushButton("…")
        in_btn.setMaximumWidth(40)
        in_btn.clicked.connect(lambda: self._pick_dir(self.input_edit, "입력 폴더 선택"))
        grid.addWidget(self.input_edit, 0, 1)
        grid.addWidget(in_btn, 0, 2)

        grid.addWidget(QLabel("출력 폴더"), 1, 0)
        self.output_edit = QLineEdit("detection_results_batch")
        out_btn = QPushButton("…")
        out_btn.setMaximumWidth(40)
        out_btn.clicked.connect(lambda: self._pick_dir(self.output_edit, "출력 폴더 선택"))
        grid.addWidget(self.output_edit, 1, 1)
        grid.addWidget(out_btn, 1, 2)

        # Run button
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("검출 시작")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn = QPushButton("중단")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        run_row.addWidget(self.run_btn, 1)
        run_row.addWidget(self.cancel_btn)

        # Progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.status_label = QLabel("대기 중")
        self.status_label.setObjectName("section")

        # Log/summary
        log_group = QGroupBox("진행 로그")
        log_lay = QVBoxLayout(log_group)
        log_lay.setContentsMargins(12, 18, 12, 12)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("QTextEdit { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; font-family: 'Consolas', monospace; font-size: 11px; }")
        log_lay.addWidget(self.log)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)
        outer.addWidget(path_group)
        outer.addLayout(run_row)
        outer.addWidget(self.progress)
        outer.addWidget(self.status_label)
        outer.addWidget(log_group, 1)

    def _pick_dir(self, line_edit: QLineEdit, title: str) -> None:
        d = QFileDialog.getExistingDirectory(self, title, line_edit.text())
        if d:
            line_edit.setText(d)

    def _on_run(self) -> None:
        in_dir = Path(self.input_edit.text())
        out_dir = Path(self.output_edit.text())
        if not in_dir.is_dir():
            self.status_label.setText(f"❌ 입력 폴더 없음: {in_dir}")
            return
        self.log.clear()
        self.log.append(f"입력: {in_dir.resolve()}")
        self.log.append(f"출력: {out_dir.resolve()}")

        self._worker = BatchWorker(self._controller, in_dir, out_dir)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_with_summary.connect(self._on_finished)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.status_label.setText("취소 요청됨…")

    def _on_progress(self, done: int, total: int, current: str) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        self.status_label.setText(f"{done} / {total}  —  {current}")
        self.log.append(f"  [{done:>3}/{total}] {current}")

    def _on_finished(self, summary: dict) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.log.append("")
        self.log.append("=" * 50)
        self.log.append(f"파일: {summary['files']}")
        self.log.append(f"총 검출: {summary['total_dets']}")
        for k, v in summary["per_class"].items():
            self.log.append(f"  {k:>5}: {v}")
        self.status_label.setText("✅ 완료")
