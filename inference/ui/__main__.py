"""Run the NeoMscope GUI.

CLI:
    python -m inference.ui                              # interactive
    python -m inference.ui --screenshot out.png         # render-and-quit
    python -m inference.ui --screenshot-tab Live --screenshot out.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from inference.ui.main_window import MainWindow

TAB_NAMES = ("Live", "Batch", "Archive", "Settings")


def _load_stylesheet() -> str:
    css_path = Path(__file__).parent / "styles.qss"
    return css_path.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Render once, save PNG to this path, then exit",
    )
    parser.add_argument(
        "--screenshot-tab",
        choices=TAB_NAMES,
        default="Live",
        help="Which tab to show in --screenshot mode (default: Live)",
    )
    parser.add_argument(
        "--screenshot-all",
        type=Path,
        help="Render every tab to <dir>/<tab>.png",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    app = QApplication(sys.argv)
    app.setStyleSheet(_load_stylesheet())

    window = MainWindow()
    window.show()

    if args.screenshot_all:
        out_dir = args.screenshot_all
        out_dir.mkdir(parents=True, exist_ok=True)

        def _grab_all(idx: int = 0) -> None:
            if idx >= len(TAB_NAMES):
                app.quit()
                return
            window.tabs.setCurrentIndex(idx)
            QTimer.singleShot(500, lambda: _save_and_next(idx))

        def _save_and_next(idx: int) -> None:
            out_path = out_dir / f"{TAB_NAMES[idx]}.png"
            window.grab().save(str(out_path))
            print(f"  Saved {out_path}")
            QTimer.singleShot(50, lambda: _grab_all(idx + 1))

        QTimer.singleShot(300, _grab_all)
        return app.exec()

    if args.screenshot:
        window.tabs.setCurrentIndex(TAB_NAMES.index(args.screenshot_tab))

        def _grab() -> None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.screenshot))
            print(f"Saved {args.screenshot}")
            app.quit()

        QTimer.singleShot(500, _grab)
        return app.exec()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
