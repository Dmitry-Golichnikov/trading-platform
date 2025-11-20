from __future__ import annotations

from typing import Sequence

from PySide6.QtWidgets import QDialog, QGridLayout, QLabel

from ..services.training_service import TrainingJobResult


class ComparisonDialog(QDialog):
    """Показывает ключевые метрики нескольких моделей."""

    def __init__(self, title: str, results: Sequence[TrainingJobResult], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)

        layout = QGridLayout(self)
        headers = ["Модель", "Train Acc", "Val Acc", "Artifacts"]
        for col, header in enumerate(headers):
            layout.addWidget(QLabel(f"<b>{header}</b>"), 0, col)

        for row, result in enumerate(results, start=1):
            layout.addWidget(QLabel(result.artifacts.get("model_name", "unknown")), row, 0)
            layout.addWidget(QLabel(f"{result.metrics.get('train_accuracy', 0):.4f}"), row, 1)
            layout.addWidget(QLabel(f"{result.metrics.get('val_accuracy', 0):.4f}"), row, 2)
            layout.addWidget(QLabel(result.artifacts.get("model_path", "-")), row, 3)
