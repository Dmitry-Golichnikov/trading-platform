from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional

from PySide6.QtWidgets import QApplication, QMessageBox

from .services import GUIServiceBundle, create_default_services
from .windows.main_window import MainWindow

logger = logging.getLogger(__name__)


ThemeLiteral = Literal["light", "dark"]


@dataclass(slots=True)
class GUIApplication:
    """Высокоуровневый фасад для запуска desktop GUI."""

    theme: ThemeLiteral = "dark"
    services: Optional[GUIServiceBundle] = None
    _app: QApplication = field(init=False)
    _window: MainWindow = field(init=False)

    def __post_init__(self) -> None:
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        self._app = app
        self._app.setApplicationName("Trading Platform")
        self._app.setOrganizationName("TradingPlatform")
        self._app.setDesktopFileName("trading-platform")

        self.services = self.services or create_default_services()
        self._window = MainWindow(services=self.services, theme=self.theme)

        try:
            self._apply_theme(self.theme)
        except Exception as exc:  # pragma: no cover - best effort theming
            logger.warning("Не удалось применить тему %s: %s", self.theme, exc)

    def _apply_theme(self, theme: ThemeLiteral) -> None:
        app = self._app
        if theme == "dark":
            try:
                import qdarkstyle

                app.setStyleSheet(qdarkstyle.load_stylesheet_pyside6())
                return
            except ImportError:
                logger.debug("qdarkstyle не установлен, fallback на qdarktheme")
            except Exception as exc:  # pragma: no cover - система тем
                logger.debug("qdarkstyle не применился: %s", exc)

            try:
                import qdarktheme

                app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
                return
            except Exception as exc:  # pragma: no cover - best effort theme
                logger.warning("qdarktheme не доступен: %s", exc)

        # Light/Fallback
        app.setStyleSheet("")

    def run(self) -> int:
        """Запустить главный цикл."""
        try:
            self._window.show()
            return self._app.exec()
        except Exception as exc:  # pragma: no cover - UI bootstrap
            logger.exception("GUI crashed: %s", exc)
            QMessageBox.critical(None, "GUI error", str(exc))
            return 1


def launch_gui(theme: ThemeLiteral = "dark") -> int:
    """Функция-запускалка для cli/scripts."""
    app = GUIApplication(theme=theme)
    return app.run()
