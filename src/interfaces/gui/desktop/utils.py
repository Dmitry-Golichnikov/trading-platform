"""
Вспомогательные утилиты для GUI.
"""

from typing import TYPE_CHECKING, cast

from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from src.interfaces.gui.desktop.windows.main_window import MainWindow


def log_to_parent(widget: QWidget, message: str) -> None:
    """
    Отправить сообщение в панель логов родительского окна, если оно поддерживает log_message.
    """

    parent = widget.parentWidget()
    if parent is None or not hasattr(parent, "log_message"):
        return

    main_window = cast("MainWindow", parent)
    main_window.log_message(message)
