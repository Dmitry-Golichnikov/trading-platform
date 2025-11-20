from typing import List, Optional

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt

from src.data.schemas import DatasetMetadata


class DatasetModel(QAbstractTableModel):
    def __init__(self, datasets: Optional[List[DatasetMetadata]] = None):
        super().__init__()
        self.datasets = datasets or []
        self.headers = ["Ticker", "Timeframe", "Start Date", "End Date", "Bars", "Source"]

    def rowCount(self, parent=QModelIndex()):
        return len(self.datasets)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None

        dataset = self.datasets[index.row()]
        col = index.column()

        if col == 0:
            return dataset.ticker
        if col == 1:
            return dataset.timeframe
        if col == 2:
            return dataset.start_date.strftime("%Y-%m-%d")
        if col == 3:
            return dataset.end_date.strftime("%Y-%m-%d")
        if col == 4:
            return str(dataset.total_bars)
        if col == 5:
            return dataset.source
        return None

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return None

    def update_data(self, datasets: List[DatasetMetadata]):
        self.beginResetModel()
        self.datasets = datasets
        self.endResetModel()

    def get_dataset_at(self, row: int) -> Optional[DatasetMetadata]:
        if 0 <= row < len(self.datasets):
            return self.datasets[row]
        return None
