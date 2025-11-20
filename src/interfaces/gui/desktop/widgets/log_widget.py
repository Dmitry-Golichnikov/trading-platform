from __future__ import annotations

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QPlainTextEdit


class LogWidget(QPlainTextEdit):
    """Виджет логов с автопрокруткой."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        font = self.document().defaultFont()
        font.setFamily("JetBrains Mono")
        self.document().setDefaultFont(font)

    def append_line(self, message: str) -> None:
        self.appendPlainText(message)
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)
