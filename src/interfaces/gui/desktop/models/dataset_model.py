from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from src.data.schemas import DatasetMetadata

COLUMNS = [
    ("ticker", "Тикер"),
    ("timeframe", "Таймфрейм"),
    ("start_date", "Начало"),
    ("end_date", "Конец"),
    ("source", "Источник"),
    ("total_bars", "Баров"),
]


class DatasetTableModel(QAbstractTableModel):
    """Qt модель метаданных датасетов."""

    def __init__(self, datasets: Sequence[DatasetMetadata] | None = None) -> None:
        super().__init__()
        self._datasets: list[DatasetMetadata] = list(datasets or [])

    def set_datasets(self, datasets: Sequence[DatasetMetadata]) -> None:
        self.beginResetModel()
        self._datasets = list(datasets)
        self.endResetModel()

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # noqa: N802
        if parent and parent.isValid():
            return 0
        return len(self._datasets)

    def columnCount(self, parent: QModelIndex | None = None) -> int:  # noqa: N802
        if parent and parent.isValid():
            return 0
        return len(COLUMNS)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: ANN001
        if not index.isValid() or role != Qt.DisplayRole:
            return None

        dataset = self._datasets[index.row()]
        attr = COLUMNS[index.column()][0]
        value = getattr(dataset, attr)

        if attr in {"start_date", "end_date", "created_at"}:
            return value.strftime("%Y-%m-%d %H:%M")

        return str(value)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # noqa: ANN001
        if role != Qt.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return super().headerData(section, orientation, role)
        try:
            return COLUMNS[section][1]
        except IndexError:
            return ""

    def dataset_at(self, row: int) -> DatasetMetadata:
        return self._datasets[row]
