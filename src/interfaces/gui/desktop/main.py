"""
Точка входа для desktop GUI приложения.
"""

import sys
from pathlib import Path

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
except ImportError:
    print("❌ PyQt6 не установлен!")
    print("Установите с помощью: pip install PyQt6 pyqtgraph qdarkstyle")
    sys.exit(1)


def _ensure_project_root() -> None:
    """Добавить корневую директорию проекта в PYTHONPATH."""
    project_root = Path(__file__).resolve().parents[4]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def main() -> int:
    """
    Запустить desktop GUI приложение.

    Returns:
        Код выхода приложения
    """
    # Гарантировать наличие корня проекта в PYTHONPATH перед локальными импортами
    _ensure_project_root()

    # Импортировать главное окно только после подготовки окружения
    from src.interfaces.gui.desktop.windows.main_window import MainWindow

    # Создать приложение
    app = QApplication(sys.argv)
    app.setApplicationName("Trading Platform")
    app.setOrganizationName("Trading Platform")
    app.setApplicationVersion("0.1.0")

    # Проверить зависимости
    missing_deps = []

    try:
        import pyqtgraph  # noqa: F401
    except ImportError:
        missing_deps.append("pyqtgraph")

    if missing_deps:
        QMessageBox.warning(
            None,
            "Отсутствуют зависимости",
            f"Следующие пакеты не установлены:\n- {', '.join(missing_deps)}\n\n"
            f"Установите их с помощью:\npip install {' '.join(missing_deps)}",
        )

    # Создать и показать главное окно
    try:
        main_window = MainWindow()
        main_window.show()
        main_window.log_message("🚀 Trading Platform запущен")
        main_window.update_status("Готов к работе")

        # Запустить event loop
        return app.exec()

    except Exception as e:
        QMessageBox.critical(
            None,
            "Ошибка запуска",
            f"Не удалось запустить приложение:\n{str(e)}",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
