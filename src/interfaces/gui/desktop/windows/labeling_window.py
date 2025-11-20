from __future__ import annotations

import yaml
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..services.dataset_service import DatasetService
from ..services.labeling_service import LabelingService
from ..widgets.chart_widget import ChartWidget
from ..widgets.config_editor import ConfigEditor
from ..widgets.table_widget import VirtualizedTableWidget


class LabelingWindow(QWidget):
    """Модуль разметки данных."""

    def __init__(
        self,
        dataset_service: DatasetService,
        labeling_service: LabelingService,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.dataset_service = dataset_service
        self.labeling_service = labeling_service

        self._build_ui()
        self._reload_datasets()
        self._load_default_config()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.dataset_combo = QComboBox(self)
        form.addRow("Датасет:", self.dataset_combo)

        self.method_combo = QComboBox(self)
        for key, label in self.labeling_service.list_methods().items():
            self.method_combo.addItem(label, key)
        form.addRow("Метод:", self.method_combo)

        generate_btn = QPushButton("Сгенерировать метки", self)
        generate_btn.clicked.connect(self.run_labeling)
        form.addWidget(generate_btn)

        layout.addLayout(form)

        splitter = QSplitter(self)
        self.chart = ChartWidget(self)
        splitter.addWidget(self.chart)

        right_panel = QVBoxLayout()
        self.config_editor = ConfigEditor(self)
        right_panel.addWidget(self.config_editor)

        right_widget = QWidget(self)
        right_widget.setLayout(right_panel)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 400])

        layout.addWidget(splitter)

        self.table = VirtualizedTableWidget(self)
        layout.addWidget(self.table)

    def _reload_datasets(self) -> None:
        datasets = self.dataset_service.list_datasets()
        self.dataset_combo.clear()
        for meta in datasets:
            self.dataset_combo.addItem(f"{meta.ticker}/{meta.timeframe}", meta)

    def _load_default_config(self) -> None:
        template = {
            "method": "triple_barrier",
            "params": {"upper_multiplier": 2.0, "lower_multiplier": 2.0, "max_holding_period": 24},
            "filters": [],
            "output_dir": "artifacts/labels",
            "dataset_id": "",
        }
        self.config_editor.set_text(yaml.safe_dump(template, sort_keys=False, allow_unicode=True))

    def run_labeling(self) -> None:
        idx = self.dataset_combo.currentIndex()
        if idx < 0:
            QMessageBox.warning(self, "Labeling", "Выберите датасет")
            return

        metadata = self.dataset_combo.currentData()
        data = self.dataset_service.load_preview(metadata, limit=50_000)

        try:
            config = self.config_editor.get_config()
        except Exception as exc:
            QMessageBox.critical(self, "Конфиг", str(exc))
            return

        config["method"] = self.method_combo.currentData()
        config["dataset_id"] = str(metadata.dataset_id)

        try:
            result = self.labeling_service.run_labeling(data, config)
        except Exception as exc:  # pragma: no cover - gui
            QMessageBox.critical(self, "Labeling", str(exc))
            return

        self.table.set_dataframe(result.data.tail(50_000))
        self.chart.plot_candles(data)
        if "label" in result.data.columns:
            self.chart.add_indicator(result.data["label"], "Labels", color="#ff006e")
        QMessageBox.information(self, "Labeling", f"Метки сохранены: {result.metadata.labeling_id}")
