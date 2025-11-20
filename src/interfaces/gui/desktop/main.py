import os
import sys

import qdarkstyle
from PyQt6.QtWidgets import QApplication

# Add project root to sys.path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from src.interfaces.gui.desktop.windows.main_window import MainWindow  # noqa: E402


def main():
    app = QApplication(sys.argv)

    # Apply dark theme
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt6"))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
