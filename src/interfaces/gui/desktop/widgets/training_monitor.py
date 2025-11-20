import pyqtgraph as pg
from PyQt6.QtWidgets import QHBoxLayout, QWidget


class TrainingMonitorWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.loss_history = {"train": [], "val": []}
        self.epochs = []

    def setup_ui(self):
        layout = QHBoxLayout(self)

        # Loss Chart
        self.loss_chart = pg.PlotWidget(title="Loss")
        self.loss_chart.showGrid(x=True, y=True, alpha=0.3)
        self.loss_chart.addLegend()
        self.train_curve = self.loss_chart.plot(pen="g", name="Train Loss")
        self.val_curve = self.loss_chart.plot(pen="r", name="Val Loss")

        layout.addWidget(self.loss_chart)

        # Metrics Chart (Placeholder for now, could be Accuracy/F1)
        self.metrics_chart = pg.PlotWidget(title="Metrics")
        layout.addWidget(self.metrics_chart)

    def update_epoch(self, epoch: int, logs: dict):
        self.epochs.append(epoch)

        if "train_loss" in logs:
            self.loss_history["train"].append(logs["train_loss"])
            self.train_curve.setData(self.epochs, self.loss_history["train"])

        if "val_loss" in logs:
            self.loss_history["val"].append(logs["val_loss"])
            self.val_curve.setData(self.epochs, self.loss_history["val"])

    def clear(self):
        self.loss_history = {"train": [], "val": []}
        self.epochs = []
        self.train_curve.setData([], [])
        self.val_curve.setData([], [])
