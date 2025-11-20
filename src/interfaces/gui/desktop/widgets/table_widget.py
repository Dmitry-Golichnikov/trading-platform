"""
Виртуализированный виджет таблицы для больших датасетов.
"""

from typing import Any, Optional

import pandas as pd

try:
    from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, QVariant
    from PyQt6.QtWidgets import QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableView, QVBoxLayout, QWidget
except ImportError:
    raise ImportError("Требуется установка: pip install PyQt6")


class PandasTableModel(QAbstractTableModel):
    """
    Qt модель для отображения pandas DataFrame.

    Поддерживает:
    - Виртуализацию (загружает только видимые строки)
    - Сортировка
    - Фильтрация
    """

    def __init__(self, data: pd.DataFrame, parent: Optional[QWidget] = None):
        """
        Инициализировать модель.

        Args:
            data: DataFrame для отображения
            parent: Родительский виджет
        """
        super().__init__(parent)
        self._data = data
        self._original_data = data.copy()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Количество строк."""
        if parent.isValid():
            return 0
        return len(self._data)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Количество колонок."""
        if parent.isValid():
            return 0
        return len(self._data.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """
        Получить данные для ячейки.

        Args:
            index: Индекс ячейки
            role: Роль данных

        Returns:
            Данные ячейки
        """
        if not index.isValid():
            return QVariant()

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            value = self._data.iloc[index.row(), index.column()]

            # Форматирование значений
            if pd.isna(value):
                return "NaN"
            elif isinstance(value, float):
                return f"{value:.6f}"
            else:
                return str(value)

        return QVariant()

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        """
        Получить данные заголовка.

        Args:
            section: Номер секции
            orientation: Ориентация (горизонтальная/вертикальная)
            role: Роль данных

        Returns:
            Данные заголовка
        """
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            else:
                return str(section)

        return QVariant()

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        """
        Сортировать данные.

        Args:
            column: Номер колонки
            order: Порядок сортировки
        """
        self.layoutAboutToBeChanged.emit()

        col_name = self._data.columns[column]
        ascending = order == Qt.SortOrder.AscendingOrder
        self._data = self._data.sort_values(by=col_name, ascending=ascending)
        self._data = self._data.reset_index(drop=True)

        self.layoutChanged.emit()

    def update_data(self, data: pd.DataFrame) -> None:
        """
        Обновить данные модели.

        Args:
            data: Новый DataFrame
        """
        self.beginResetModel()
        self._data = data
        self._original_data = data.copy()
        self.endResetModel()

    def reset_filters(self) -> None:
        """Сбросить фильтры и вернуть исходные данные."""
        self.beginResetModel()
        self._data = self._original_data.copy()
        self.endResetModel()


class VirtualizedTableWidget(QWidget):
    """
    Виджет таблицы с виртуализацией для больших датасетов.

    Особенности:
    - Виртуализация строк (отображаются только видимые)
    - Сортировка по колонкам
    - Быстрая прокрутка
    - Экспорт в CSV
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Основной layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Панель управления
        control_panel = QHBoxLayout()
        control_panel.setSpacing(5)

        self.info_label = QLabel("Строк: 0")
        control_panel.addWidget(self.info_label)

        control_panel.addStretch()

        self.reset_filters_btn = QPushButton("Сбросить фильтры")
        self.reset_filters_btn.clicked.connect(self._reset_filters)
        control_panel.addWidget(self.reset_filters_btn)

        self.export_btn = QPushButton("Экспорт в CSV")
        self.export_btn.clicked.connect(self._export_csv)
        control_panel.addWidget(self.export_btn)

        layout.addLayout(control_panel)

        # Таблица
        self.table_view = QTableView()
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)

        # Настройки заголовков
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_view.verticalHeader().setDefaultSectionSize(25)

        layout.addWidget(self.table_view)

        # Модель данных
        self.model: Optional[PandasTableModel] = None

    def set_data(self, data: pd.DataFrame) -> None:
        """
        Установить данные для отображения.

        Args:
            data: DataFrame для отображения
        """
        if self.model is None:
            self.model = PandasTableModel(data)
            self.table_view.setModel(self.model)
        else:
            self.model.update_data(data)

        self._update_info()

    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Получить текущие данные.

        Returns:
            DataFrame с данными или None
        """
        if self.model is not None:
            return self.model._data.copy()
        return None

    def _update_info(self) -> None:
        """Обновить информацию о количестве строк."""
        if self.model is not None:
            rows = self.model.rowCount()
            cols = self.model.columnCount()
            self.info_label.setText(f"Строк: {rows:,} | Колонок: {cols}")

    def _reset_filters(self) -> None:
        """Сбросить фильтры."""
        if self.model is not None:
            self.model.reset_filters()
            self._update_info()

    def _export_csv(self) -> None:
        """Экспорт данных в CSV."""
        if self.model is not None:
            from PyQt6.QtWidgets import QFileDialog

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Экспорт в CSV",
                "",
                "CSV файлы (*.csv)",
            )

            if file_path:
                try:
                    self.model._data.to_csv(file_path, index=False)
                    from PyQt6.QtWidgets import QMessageBox

                    QMessageBox.information(
                        self,
                        "Экспорт завершён",
                        f"Данные успешно экспортированы в:\n{file_path}",
                    )
                except Exception as e:
                    from PyQt6.QtWidgets import QMessageBox

                    QMessageBox.critical(
                        self,
                        "Ошибка экспорта",
                        f"Не удалось экспортировать данные:\n{str(e)}",
                    )
