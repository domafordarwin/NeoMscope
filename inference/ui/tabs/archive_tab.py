"""Archive tab — browse saved detection results.

Lists JPEG files under detection_results_captured/ and detection_results_live/
with thumbnail previews. Click a thumbnail to open it in a larger view.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from inference._camera import IMAGE_EXTS
from inference.ui.state import AppController

logger = logging.getLogger("ui.archive")

THUMB_SIZE = QSize(120, 90)


class ArchiveTab(QWidget):
    """Browse past detection results."""

    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self._controller = controller
        self._build()
        self.refresh()

    def _build(self) -> None:
        # File list
        files_group = QGroupBox("저장된 결과")
        files_lay = QVBoxLayout(files_group)
        files_lay.setContentsMargins(12, 18, 12, 12)
        files_lay.setSpacing(6)

        toolbar = QHBoxLayout()
        self.refresh_btn = QPushButton("새로고침")
        self.refresh_btn.clicked.connect(self.refresh)
        self.count_label = QLabel("(0)")
        self.count_label.setObjectName("section")
        toolbar.addWidget(self.refresh_btn)
        toolbar.addStretch()
        toolbar.addWidget(self.count_label)
        files_lay.addLayout(toolbar)

        self.list = QListWidget()
        self.list.setIconSize(THUMB_SIZE)
        self.list.setSpacing(4)
        self.list.setStyleSheet(
            "QListWidget { background: white; border: 1px solid #d0d7de; border-radius: 6px; }"
            "QListWidget::item { padding: 4px; }"
            "QListWidget::item:selected { background: #ddf4ff; color: #1f2328; }"
        )
        self.list.itemClicked.connect(self._on_item_clicked)
        files_lay.addWidget(self.list, 1)

        # Preview
        preview_group = QGroupBox("미리보기")
        preview_lay = QVBoxLayout(preview_group)
        preview_lay.setContentsMargins(12, 18, 12, 12)
        self.preview = QLabel("파일 선택 시 여기 표시")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(280)
        self.preview.setStyleSheet(
            "QLabel { color: #9198a1; background: #f6f8fa; "
            "border: 1px dashed #d0d7de; border-radius: 6px; }"
        )
        preview_lay.addWidget(self.preview)

        self.meta_label = QLabel("")
        self.meta_label.setObjectName("section")
        preview_lay.addWidget(self.meta_label)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)
        outer.addWidget(files_group, 1)
        outer.addWidget(preview_group, 1)

    def refresh(self) -> None:
        self.list.clear()
        s = self._controller.settings
        candidates: list[Path] = []
        for d in (s.output_dir_capture, s.output_dir_live, Path("detection_results_batch")):
            if d.is_dir():
                candidates.extend(p for p in sorted(d.iterdir())
                                  if p.is_file() and p.suffix in IMAGE_EXTS)

        for p in candidates:
            item = QListWidgetItem(p.name)
            try:
                pix = QPixmap(str(p)).scaled(
                    THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                item.setIcon(pix)  # type: ignore[arg-type]
            except Exception as exc:
                logger.warning("Could not load thumb %s: %s", p, exc)
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self.list.addItem(item)

        self.count_label.setText(f"({self.list.count()})")
        if self.list.count() == 0:
            self.preview.setText(
                "저장된 결과 없음.\nLive 탭에서 촬영하거나 Batch 탭에서 일괄 처리하세요."
            )

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        if not path.is_file():
            return
        pix = QPixmap(str(path))
        if pix.isNull():
            return
        scaled = pix.scaled(
            self.preview.width() - 20, self.preview.height() - 20,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(scaled)
        size_kb = path.stat().st_size / 1024
        self.meta_label.setText(
            f"{path}  ·  {pix.width()}x{pix.height()}  ·  {size_kb:.0f} KB"
        )
