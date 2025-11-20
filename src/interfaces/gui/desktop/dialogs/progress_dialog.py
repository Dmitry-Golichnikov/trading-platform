from __future__ import annotations

from PySide6.QtWidgets import QProgressDialog


class ProgressDialog(QProgressDialog):
    """Модальное окно прогресса с кнопкой отмены."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__("Пожалуйста, подождите...", "Отмена", 0, 100, parent)
        self.setWindowTitle(title)
        self.setMinimumDuration(500)
        self.setAutoClose(False)
        self.setAutoReset(False)

    def update_progress(self, value: float, message: str) -> None:
        self.setLabelText(message)
        self.setValue(int(value * 100))
