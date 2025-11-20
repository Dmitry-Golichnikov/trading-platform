from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout

from ..widgets.config_editor import ConfigEditor


class ConfigDialog(QDialog):
    """Диалог редактирования конфигурации (YAML/JSON)."""

    def __init__(self, title: str, config: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)

        self.editor = ConfigEditor(self)
        self.editor.set_text(config and self._dump_config(config) or "")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.editor)
        layout.addWidget(buttons)

    def get_config(self) -> dict[str, Any]:
        return self.editor.get_config()

    def _dump_config(self, config: dict[str, Any]) -> str:
        import yaml

        return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
