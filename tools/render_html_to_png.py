"""Render an HTML file to a PNG using PySide6's QtWebEngine.

Usage:
    python tools/render_html_to_png.py input.html output.png [--width 1024 --height 600 --wait 2000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QSize, QTimer, QUrl
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication


def render(input_path: Path, output_path: Path, width: int, height: int, wait_ms: int) -> int:
    app = QApplication(sys.argv)

    view = QWebEngineView()
    view.resize(width, height)
    view.setFixedSize(width, height)

    settings = view.settings()
    settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
    settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
    settings.setAttribute(QWebEngineSettings.WebAttribute.AllowRunningInsecureContent, True)

    page: QWebEnginePage = view.page()
    page.setBackgroundColor("#F6F4ED")  # match design paper bg in case fonts late-load

    url = QUrl.fromLocalFile(str(input_path.resolve()))

    finished = {"loaded": False, "exit_code": 0}

    def grab_and_quit() -> None:
        # Render the view into a QImage of exact target size
        img = QImage(QSize(width, height), QImage.Format.Format_ARGB32)
        img.fill(0xFFF6F4ED)
        painter = QPainter(img)
        view.render(painter)
        painter.end()
        if not img.save(str(output_path)):
            print(f"FAILED to save {output_path}", file=sys.stderr)
            finished["exit_code"] = 2
        else:
            print(f"Saved {output_path} ({width}x{height})")
        QTimer.singleShot(50, app.quit)

    def on_loaded(ok: bool) -> None:  # noqa: FBT001
        if not ok:
            print("page load failed", file=sys.stderr)
            finished["exit_code"] = 1
            QTimer.singleShot(50, app.quit)
            return
        # Wait for fonts/network resources, then grab
        QTimer.singleShot(wait_ms, grab_and_quit)

    page.loadFinished.connect(on_loaded)
    view.show()  # required so off-screen surface initializes
    view.load(url)

    return app.exec() or finished["exit_code"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--wait", type=int, default=2000, help="ms to wait after load (font fetch)")
    args = p.parse_args()

    if not args.input.is_file():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)

    return render(args.input, args.output, args.width, args.height, args.wait)


if __name__ == "__main__":
    sys.exit(main())
