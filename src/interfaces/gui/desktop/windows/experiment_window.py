import time

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ExperimentWorker(QThread):
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal()

    def __init__(self, experiments):
        super().__init__()
        self.experiments = experiments

    def run(self):
        total = len(self.experiments)
        for i, exp in enumerate(self.experiments):
            # Simulate work
            time.sleep(0.5)
            # In real app, we would call TrainingService or BacktestService here
            self.progress.emit(i + 1, total)
        self.finished.emit()


class ExperimentWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.experiments = []

    def setup_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Batch Experiments"))

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Name", "Status", "Result"])
        layout.addWidget(self.table)

        # Controls
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add Mock Experiment")
        self.add_btn.clicked.connect(self.add_mock_experiment)
        self.run_btn = QPushButton("Run All")
        self.run_btn.clicked.connect(self.run_experiments)

        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.run_btn)
        layout.addLayout(btn_layout)

        # Progress
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

    def add_mock_experiment(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(f"Exp_{row+1}"))
        self.table.setItem(row, 1, QTableWidgetItem("Pending"))
        self.table.setItem(row, 2, QTableWidgetItem("-"))
        self.experiments.append({"id": row})

    def run_experiments(self):
        if not self.experiments:
            return

        self.run_btn.setEnabled(False)
        self.progress_bar.setMaximum(len(self.experiments))
        self.progress_bar.setValue(0)

        self.worker = ExperimentWorker(self.experiments)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def update_progress(self, current, total):
        self.progress_bar.setValue(current)
        self.table.setItem(current - 1, 1, QTableWidgetItem("Completed"))
        self.table.setItem(current - 1, 2, QTableWidgetItem("Success"))

    def on_finished(self):
        self.run_btn.setEnabled(True)
