import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from pyqtgraph import DateAxisItem


class ChartWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Date Axis
        self.axis = DateAxisItem(orientation="bottom")

        # Plot Widget
        self.plot_widget = pg.PlotWidget(axisItems={"bottom": self.axis})
        self.plot_widget.setBackground("k")  # Black background
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "Price")
        self.plot_widget.setLabel("bottom", "Date")

        self.layout.addWidget(self.plot_widget)

        self.plot_item = self.plot_widget.getPlotItem()
        self.curve = None

    def plot_data(self, df: pd.DataFrame):
        self.plot_item.clear()

        if df.empty:
            return

        # Simple line chart for Close price for now
        # Timestamps need to be in seconds for pyqtgraph DateAxisItem
        timestamps = df["timestamp"].astype(np.int64) // 10**9
        close_prices = df["close"].values

        self.curve = self.plot_item.plot(x=timestamps, y=close_prices, pen="w", name="Close")
        self.plot_widget.autoRange()
