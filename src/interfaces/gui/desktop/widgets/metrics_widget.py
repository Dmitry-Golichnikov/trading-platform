from __future__ import annotations

from typing import Mapping

import pandas as pd
from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget

from .chart_widget import ChartWidget


class MetricsWidget(QWidget):
    """Комбинация графиков и текстового лога для отображения метрик."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.loss_chart = ChartWidget(self)
        self.metrics_chart = ChartWidget(self)
        self.metrics_view = QPlainTextEdit(self)
        self.metrics_view.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.loss_chart)
        layout.addWidget(self.metrics_chart)
        layout.addWidget(self.metrics_view)

    def update_metrics(self, metrics: Mapping[str, float]) -> None:
        lines = [f"{key}: {value:.4f}" for key, value in metrics.items()]
        self.metrics_view.setPlainText("\n".join(lines))

    def update_history(self, history: Mapping[str, list[float]]) -> None:
        for name, values in history.items():
            if not values:
                continue
            self.metrics_chart.add_indicator(series=pd.Series(values), name=name)
