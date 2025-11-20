import json

import pandas as pd
from PyQt6.QtCore import Qt
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
from src.interfaces.gui.desktop.services.training_service import TrainingService
from src.interfaces.gui.desktop.widgets.training_monitor import TrainingMonitorWidget
from src.interfaces.gui.desktop.workers.training_worker import TrainingWorker


class TrainingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.service = TrainingService()
        self.dataset_service = DatasetService()
        # In real app, we would need access to created features/labels
        # For prototype, we might need to generate them on the fly or load from disk

        self.setup_ui()
        self.load_models()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Config
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Model Selection
        left_layout.addWidget(QLabel("Model Type:"))
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        left_layout.addWidget(self.model_combo)

        # Params
        left_layout.addWidget(QLabel("Hyperparameters (JSON):"))
        self.params_edit = QTextEdit()
        left_layout.addWidget(self.params_edit)

        # Data selection (Simplified for prototype)
        left_layout.addWidget(QLabel("Select Artifacts (Dataset/Features/Labels):"))
        self.artifacts_combo = QComboBox()
        self.artifacts_combo.addItem("Load from disk (Mock)")  # Placeholder
        left_layout.addWidget(self.artifacts_combo)

        # Train Button
        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.start_training)
        left_layout.addWidget(self.train_btn)

        left_layout.addStretch()

        # Right: Monitor
        self.monitor = TrainingMonitorWidget()

        splitter.addWidget(left_widget)
        splitter.addWidget(self.monitor)
        layout.addWidget(splitter)

    def load_models(self):
        models = self.service.get_available_models()
        self.model_combo.addItems(models)
        self.on_model_changed(self.model_combo.currentText())

    def on_model_changed(self, model_name):
        defaults = self.service.get_default_params(model_name)
        self.params_edit.setText(json.dumps(defaults, indent=2))

    def start_training(self):
        # Mock Data Generation for Prototype
        # In real app, load from self.artifacts_combo selection
        import numpy as np

        try:
            params = json.loads(self.params_edit.toPlainText())
        except json.JSONDecodeError:
            QMessageBox.critical(self, "Error", "Invalid JSON parameters")
            return

        # Create dummy data
        dates = pd.date_range(start="2023-01-01", periods=1000, freq="1H")
        X = pd.DataFrame(np.random.randn(1000, 5), columns=[f"feat_{i}" for i in range(5)], index=dates)
        y = pd.Series(np.random.randint(0, 2, 1000), index=dates, name="label")

        split = 800
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_val, y_val = X.iloc[split:], y.iloc[split:]

        self.monitor.clear()
        self.train_btn.setEnabled(False)

        self.worker = TrainingWorker(self.model_combo.currentText(), params, X_train, y_train, X_val, y_val)
        self.worker.signals.epoch_finished.connect(self.monitor.update_epoch)
        self.worker.signals.training_finished.connect(self.on_training_finished)
        self.worker.signals.error_occurred.connect(self.on_error)
        self.worker.start()

    def on_training_finished(self, logs):
        self.train_btn.setEnabled(True)
        QMessageBox.information(self, "Success", f"Training finished.\nFinal metrics: {logs}")

    def on_error(self, msg):
        self.train_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Training failed: {msg}")
