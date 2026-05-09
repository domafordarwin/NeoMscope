"""Run the NeoMscope GUI demo.

CLI:
    .venv\\Scripts\\python.exe -m inference.ui                       # interactive
    .venv\\Scripts\\python.exe -m inference.ui --screenshot out.png   # render-and-quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from inference.ui.main_window import MainWindow


def _load_stylesheet() -> str:
    css_path = Path(__file__).parent / "styles.qss"
    return css_path.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--demo-image",
        type=Path,
        default=Path("captured_raw_images/2022-07-18_18-02-29.jpg"),
        help="Image to use for the live preview demo",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Render once, save PNG to this path, then exit",
    )
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    app.setStyleSheet(_load_stylesheet())

    window = MainWindow(demo_image=args.demo_image)

    if args.screenshot:
        window.show()

        def _grab() -> None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.screenshot))
            print(f"Saved {args.screenshot}")
            app.quit()

        QTimer.singleShot(200, _grab)
        return app.exec()

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
