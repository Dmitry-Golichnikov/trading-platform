from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from ..services.experiment_service import ExperimentResult


class ExperimentTableModel(QAbstractTableModel):
    """Отображение результатов batch-экспериментов."""

    headers = [
        "Dataset",
        "Features",
        "Labeling",
        "Model",
        "Strategy",
        "Train Accuracy",
        "Val Accuracy",
        "Sharpe",
        "Win Rate",
    ]

    def __init__(self, results: Sequence[ExperimentResult] | None = None) -> None:
        super().__init__()
        self._results: list[ExperimentResult] = list(results or [])

    def set_results(self, results: Sequence[ExperimentResult]) -> None:
        self.beginResetModel()
        self._results = list(results)
        self.endResetModel()

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # noqa: N802
        if parent and parent.isValid():
            return 0
        return len(self._results)

    def columnCount(self, parent: QModelIndex | None = None) -> int:  # noqa: N802
        if parent and parent.isValid():
            return 0
        return len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: ANN001
        if not index.isValid() or role != Qt.DisplayRole:
            return None

        result = self._results[index.row()]
        combo = result.combination
        metrics = result.training.metrics
        backtest_metrics = result.backtest.metrics

        column_map = [
            combo["dataset"],
            combo["features"],
            combo["labeling"],
            combo["model"],
            combo["strategy"],
            f"{metrics.get('train_accuracy', 0):.4f}",
            f"{metrics.get('val_accuracy', 0):.4f}",
            f"{backtest_metrics.get('sharpe_ratio', 0):.2f}",
            f"{backtest_metrics.get('win_rate', 0)*100:.1f}%",
        ]

        try:
            return column_map[index.column()]
        except IndexError:
            return ""

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # noqa: ANN001
        if role != Qt.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return super().headerData(section, orientation, role)
        try:
            return self.headers[section]
        except IndexError:
            return ""
