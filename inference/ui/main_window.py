"""NeoMscope main window — 1024x600 single-screen layout with 4 tabs."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget

from inference.ui.state import AppController
from inference.ui.tabs.archive_tab import ArchiveTab
from inference.ui.tabs.batch_tab import BatchTab
from inference.ui.tabs.live_tab import LiveTab
from inference.ui.tabs.settings_tab import SettingsTab

WINDOW_W, WINDOW_H = 1024, 600


class MainWindow(QMainWindow):
    """Main window — Live / Batch / Archive / Settings tabs."""

    def __init__(self, controller: AppController | None = None) -> None:
        super().__init__()
        self.setWindowTitle("NeoMscope")
        self.setFixedSize(WINDOW_W, WINDOW_H)

        self._controller = controller or AppController()

        title = QLabel("NeoMscope — 양파 세포 분열 검출")
        title.setObjectName("title")

        self.tabs = QTabWidget()
        self.live_tab = LiveTab(self._controller)
        self.batch_tab = BatchTab(self._controller)
        self.archive_tab = ArchiveTab(self._controller)
        self.settings_tab = SettingsTab(self._controller)
        self.tabs.addTab(self.live_tab, "Live")
        self.tabs.addTab(self.batch_tab, "Batch")
        self.tabs.addTab(self.archive_tab, "Archive")
        self.tabs.addTab(self.settings_tab, "Settings")

        # Refresh archive when its tab is shown
        self.tabs.currentChanged.connect(self._on_tab_changed)

        outer = QWidget()
        lay = QVBoxLayout(outer)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(title)
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(outer)

    def _on_tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.archive_tab:
            self.archive_tab.refresh()
