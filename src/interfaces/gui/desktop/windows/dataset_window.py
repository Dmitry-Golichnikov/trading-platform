import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QHeaderView, QMessageBox, QPushButton, QTableView, QVBoxLayout, QWidget

from src.interfaces.gui.desktop.models.dataset_model import DatasetModel
from src.interfaces.gui.desktop.services.dataset_service import DatasetService
from src.interfaces.gui.desktop.widgets.chart_widget import ChartWidget


class DataLoaderThread(QThread):
    data_loaded = pyqtSignal(pd.DataFrame)
    error_occurred = pyqtSignal(str)

    def __init__(self, service, ticker, timeframe):
        super().__init__()
        self.service = service
        self.ticker = ticker
        self.timeframe = timeframe

    def run(self):
        try:
            df = self.service.load_dataset_data(self.ticker, self.timeframe)
            self.data_loaded.emit(df)
        except Exception as e:
            self.error_occurred.emit(str(e))


class DatasetWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.service = DatasetService()
        self.model = DatasetModel()
        self.loader_thread = None

        self.setup_ui()
        self.refresh_data()

    def setup_ui(self):
        layout = QHBoxLayout(self)

        # Left Panel: List
        left_panel = QVBoxLayout()

        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.selectionModel().currentRowChanged.connect(self.on_selection_changed)

        left_panel.addWidget(self.table)

        # Buttons
        btn_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_data)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self.delete_dataset)
        self.delete_btn.setEnabled(False)

        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addWidget(self.delete_btn)
        left_panel.addLayout(btn_layout)

        # Right Panel: Chart
        self.chart = ChartWidget()

        layout.addLayout(left_panel, 1)  # 1/3 width
        layout.addWidget(self.chart, 2)  # 2/3 width

    def refresh_data(self):
        try:
            datasets = self.service.get_all_datasets()
            self.model.update_data(datasets)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to refresh datasets: {e}")

    def on_selection_changed(self, current, previous):
        if not current.isValid():
            self.delete_btn.setEnabled(False)
            return

        self.delete_btn.setEnabled(True)
        dataset = self.model.get_dataset_at(current.row())

        # Load data asynchronously
        if self.loader_thread and self.loader_thread.isRunning():
            self.loader_thread.terminate()

        self.loader_thread = DataLoaderThread(self.service, dataset.ticker, dataset.timeframe)
        self.loader_thread.data_loaded.connect(self.on_data_loaded)
        self.loader_thread.error_occurred.connect(self.on_load_error)
        self.loader_thread.start()

    def on_data_loaded(self, df):
        self.chart.plot_data(df)

    def on_load_error(self, error_msg):
        # Don't show annoying popups if data just doesn't exist for empty dataset or similar,
        # log it or show in status bar (if available)
        print(f"Failed to load dataset data: {error_msg}")

    def delete_dataset(self):
        index = self.table.currentIndex()
        if not index.isValid():
            return

        dataset = self.model.get_dataset_at(index.row())
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete {dataset.ticker} {dataset.timeframe}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.service.delete_dataset(dataset.dataset_id)
                self.refresh_data()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete dataset: {e}")
