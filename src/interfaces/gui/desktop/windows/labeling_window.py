import json

import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.interfaces.gui.desktop.services.dataset_service import DatasetService
from src.interfaces.gui.desktop.services.labeling_service import LabelingService
from src.interfaces.gui.desktop.widgets.chart_widget import ChartWidget


class LabelingWorker(QThread):
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, service, data, config):
        super().__init__()
        self.service = service
        self.data = data
        self.config = config

    def run(self):
        try:
            result = self.service.run_labeling(self.data, self.config)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class LabelingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.service = LabelingService()
        self.dataset_service = DatasetService()
        self.data = None

        self.setup_ui()
        self.load_datasets()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Panel: Configuration
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Dataset Selection
        left_layout.addWidget(QLabel("Select Dataset:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self.on_dataset_changed)
        left_layout.addWidget(self.dataset_combo)

        # Method Selection
        left_layout.addWidget(QLabel("Labeling Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(self.service.get_available_methods())
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        left_layout.addWidget(self.method_combo)

        # Parameters
        left_layout.addWidget(QLabel("Parameters (JSON):"))
        self.params_edit = QTextEdit()
        left_layout.addWidget(self.params_edit)

        # Run Button
        self.run_btn = QPushButton("Run Labeling")
        self.run_btn.clicked.connect(self.run_labeling)
        left_layout.addWidget(self.run_btn)

        left_layout.addStretch()

        # Right Panel: Visualization
        self.chart = ChartWidget()

        splitter.addWidget(left_widget)
        splitter.addWidget(self.chart)
        layout.addWidget(splitter)

        # Trigger initial param load
        self.on_method_changed(self.method_combo.currentText())

    def load_datasets(self):
        datasets = self.dataset_service.get_all_datasets()
        self.datasets_map = {f"{d.ticker} {d.timeframe}": d for d in datasets}
        self.dataset_combo.addItems(self.datasets_map.keys())

    def on_dataset_changed(self):
        key = self.dataset_combo.currentText()
        if key in self.datasets_map:
            metadata = self.datasets_map[key]
            try:
                self.data = self.dataset_service.load_dataset_data(metadata.ticker, metadata.timeframe)
                self.chart.plot_data(self.data)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load dataset: {e}")

    def on_method_changed(self, method):
        defaults = self.service.get_default_params(method)
        self.params_edit.setText(json.dumps(defaults, indent=2))

    def run_labeling(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "No dataset loaded")
            return

        try:
            params = json.loads(self.params_edit.toPlainText())
        except json.JSONDecodeError:
            QMessageBox.critical(self, "Error", "Invalid JSON parameters")
            return

        config = {"method": self.method_combo.currentText(), "params": params, "dataset_id": "gui_preview"}

        self.run_btn.setEnabled(False)
        self.worker = LabelingWorker(self.service, self.data, config)
        self.worker.finished.connect(self.on_labeling_finished)
        self.worker.error.connect(self.on_labeling_error)
        self.worker.start()

    def on_labeling_finished(self, labeled_data):
        self.run_btn.setEnabled(True)
        # In a real app, we would overlay labels on the chart
        # For now, just show success message
        QMessageBox.information(self, "Success", f"Labeling complete. Generated {len(labeled_data)} labels.")
        # Here we could add markers to self.chart if we extended ChartWidget

    def on_labeling_error(self, error_msg):
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Labeling failed: {error_msg}")
