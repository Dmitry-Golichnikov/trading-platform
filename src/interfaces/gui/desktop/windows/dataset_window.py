from __future__ import annotations

from pathlib import Path
from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.data.storage.parquet_storage import TimeframeLiteral

from ..models.dataset_model import DatasetTableModel
from ..services.dataset_service import DatasetFilters, DatasetService
from ..widgets.chart_widget import ChartWidget
from ..widgets.table_widget import VirtualizedTableWidget


class DatasetWindow(QWidget):
    """Модуль управления датасетами."""

    def __init__(self, dataset_service: DatasetService, parent=None) -> None:
        super().__init__(parent)
        self.service = dataset_service

        self.filters = DatasetFilters()
        self.table_model = DatasetTableModel()

        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        filter_layout = QFormLayout()
        self.ticker_input = QLineEdit(self)
        self.timeframe_input = QLineEdit(self)
        self.source_input = QLineEdit(self)

        filter_layout.addRow("Тикер:", self.ticker_input)
        filter_layout.addRow("Таймфрейм:", self.timeframe_input)
        filter_layout.addRow("Источник:", self.source_input)

        filter_buttons = QHBoxLayout()
        refresh_btn = QPushButton("Обновить", self)
        refresh_btn.clicked.connect(self.refresh)
        import_btn = QPushButton("Импортировать...", self)
        import_btn.clicked.connect(self.import_dataset)
        delete_btn = QPushButton("Удалить", self)
        delete_btn.clicked.connect(self.delete_selected)

        filter_buttons.addWidget(refresh_btn)
        filter_buttons.addWidget(import_btn)
        filter_buttons.addWidget(delete_btn)

        top_container = QVBoxLayout()
        top_container.addLayout(filter_layout)
        top_container.addLayout(filter_buttons)

        main_layout.addLayout(top_container)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        self.table = VirtualizedTableWidget(self)
        self.table.setModel(self.table_model)
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self.table)

        self.chart = ChartWidget(self)
        splitter.addWidget(self.chart)
        splitter.setSizes([600, 400])

        main_layout.addWidget(splitter)

        self.summary_label = QLabel("Нет данных", self)
        main_layout.addWidget(self.summary_label)

    def refresh(self) -> None:
        self.filters = DatasetFilters(
            ticker=self.ticker_input.text() or None,
            timeframe=self.timeframe_input.text() or None,
            source=self.source_input.text() or None,
        )
        datasets = self.service.list_datasets(self.filters)
        self.table_model.set_datasets(datasets)
        self.summary_label.setText(f"Найдено {len(datasets)} датасетов")

    def import_dataset(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбор файла", "", "Parquet (*.parquet);;CSV (*.csv)")
        if not file_path:
            return

        ticker = self.ticker_input.text().strip()
        timeframe_str = self.timeframe_input.text().strip() or "1h"
        if not ticker:
            QMessageBox.warning(self, "Импорт", "Укажите тикер в фильтре перед импортом.")
            return

        valid_timeframes: tuple[TimeframeLiteral, ...] = ("1m", "5m", "15m", "1h", "4h", "1d")
        if timeframe_str not in valid_timeframes:
            QMessageBox.warning(
                self,
                "Импорт",
                f"Недопустимый таймфрейм: {timeframe_str}. Допустимые: {', '.join(valid_timeframes)}",
            )
            return

        timeframe = cast(TimeframeLiteral, timeframe_str)
        metadata = self.service.import_from_file(Path(file_path), ticker=ticker, timeframe=timeframe)
        QMessageBox.information(self, "Импорт завершен", f"{metadata.ticker}/{metadata.timeframe}")
        self.refresh()

    def delete_selected(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return

        reply = QMessageBox.question(
            self,
            "Удаление",
            f"Удалить выбранные {len(selected)} датасетов?",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        for index in selected:
            metadata = self.table_model.dataset_at(index.row())
            self.service.delete_dataset(metadata)

        self.refresh()

    def _on_selection_changed(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return
        metadata = self.table_model.dataset_at(selected[0].row())
        data = self.service.load_preview(metadata)
        self.chart.plot_candles(data)
