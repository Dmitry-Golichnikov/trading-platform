from __future__ import annotations

import json
from typing import Any

import yaml
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QVBoxLayout, QWidget


class ConfigEditor(QWidget):
    """Простой редактор YAML/JSON с валидацией."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.editor = QPlainTextEdit(self)
        self.editor.setTabStopDistance(4 * self.editor.fontMetrics().horizontalAdvance(" "))

        self.status_label = QLabel(self)
        self.status_label.setStyleSheet("color: #a3be8c;")

        layout = QVBoxLayout(self)
        layout.addWidget(self.editor)
        layout.addWidget(self.status_label)

        self.editor.textChanged.connect(self._validate_text)

    def set_text(self, text: str) -> None:
        self.editor.setPlainText(text)
        self._validate_text()

    def get_config(self) -> dict[str, Any]:
        text = self.editor.toPlainText()
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError:
            return json.loads(text)

    def _validate_text(self) -> None:
        text = self.editor.toPlainText()
        if not text.strip():
            self.status_label.setText("Конфигурация пуста")
            return

        try:
            yaml.safe_load(text)
            self.status_label.setText("OK: YAML")
            self.status_label.setStyleSheet("color: #8ec07c;")
        except yaml.YAMLError:
            try:
                json.loads(text)
                self.status_label.setText("OK: JSON")
                self.status_label.setStyleSheet("color: #8ec07c;")
            except json.JSONDecodeError as exc:
                self.status_label.setText(f"Ошибка: {exc}")
                self.status_label.setStyleSheet("color: #fb4934;")
