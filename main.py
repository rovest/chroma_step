from __future__ import annotations

import os
import sys

from PyQt5.QtWidgets import QApplication

from ui.main_window import MainWindow


def _sanitize_qt_plugin_env() -> None:
    """Avoid OpenCV Qt plugin path overriding PyQt platform plugins."""
    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
        value = os.environ.get(key)
        if value and "cv2/qt" in value:
            os.environ.pop(key, None)


def main() -> int:
    _sanitize_qt_plugin_env()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
