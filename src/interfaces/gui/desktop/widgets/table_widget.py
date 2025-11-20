from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import QTableView


class PandasTableModel(QAbstractTableModel):
    """Легковесная модель для DataFrame с поддержкой виртуализации."""

    def __init__(self, dataframe: Optional[pd.DataFrame] = None) -> None:
        super().__init__()
        self._df = dataframe if dataframe is not None else pd.DataFrame()

    def set_dataframe(self, dataframe: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = dataframe
        self.endResetModel()

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # noqa: N802 (Qt API)
        if parent and parent.isValid():
            return 0
        return len(self._df)

    def columnCount(self, parent: QModelIndex | None = None) -> int:  # noqa: N802
        if parent and parent.isValid():
            return 0
        return len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: ANN001 - Qt signature
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        value = self._df.iat[index.row(), index.column()]
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # noqa: ANN001
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            try:
                return str(self._df.columns[section])
            except IndexError:
                return ""
        return str(section)


class VirtualizedTableWidget(QTableView):
    """QTableView с настройками для отображения больших DataFrame."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._model = PandasTableModel()
        self.setModel(self._model)

        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)

        self.verticalHeader().setDefaultSectionSize(22)

    def set_dataframe(self, dataframe: pd.DataFrame) -> None:
        self._model.set_dataframe(dataframe.reset_index(drop=False))
        self.resizeColumnsToContents()
