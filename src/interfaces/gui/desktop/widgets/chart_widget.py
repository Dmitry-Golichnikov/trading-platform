from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainter


@dataclass(slots=True)
class IndicatorLine:
    name: str
    plot: pg.PlotDataItem


class ChartWidget(pg.PlotWidget):
    """Универсальный виджет для свечных графиков и кривых equity."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent, background="default")
        self.showGrid(x=True, y=True, alpha=0.2)
        self.addLegend()
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self._price_plot = self.plot(name="Price", pen=pg.mkPen("#00d1b2", width=1.5))
        self._equity_plot = self.plot(name="Equity", pen=pg.mkPen("#ffd166", width=2))
        self._indicator_lines: dict[str, IndicatorLine] = {}

        self._crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#555555"))
        self._crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#555555"))
        self.addItem(self._crosshair_v, ignoreBounds=True)
        self.addItem(self._crosshair_h, ignoreBounds=True)
        self.scene().sigMouseMoved.connect(self._update_crosshair)

    def plot_candles(self, data: pd.DataFrame) -> None:
        if data.empty:
            self._price_plot.clear()
            return

        closes = data["close"].astype(float).values
        x = range(len(closes))
        self._price_plot.setData(x=x, y=closes)
        self.setLabel("bottom", "Bars")
        self.setLabel("left", "Price")

    def plot_equity(self, equity: pd.Series) -> None:
        if equity.empty:
            self._equity_plot.clear()
            return
        x = range(len(equity))
        self._equity_plot.setData(x, equity.values.astype(float))
        self.setLabel("bottom", "Steps")
        self.setLabel("left", "Equity")

    def add_indicator(self, series: pd.Series, name: str, color: str = "#8892bf") -> None:
        if name in self._indicator_lines:
            plot_item = self._indicator_lines[name].plot
        else:
            plot_item = self.plot(name=name, pen=pg.mkPen(color, width=1.2))
            self._indicator_lines[name] = IndicatorLine(name=name, plot=plot_item)

        x = range(len(series))
        plot_item.setData(x=x, y=series.values.astype(float))

    def clear_indicators(self) -> None:
        for indicator in self._indicator_lines.values():
            self.removeItem(indicator.plot)
        self._indicator_lines.clear()

    def _update_crosshair(self, pos: QPointF) -> None:
        mouse_point = self.plotItem.vb.mapSceneToView(pos)
        self._crosshair_v.setPos(mouse_point.x())
        self._crosshair_h.setPos(mouse_point.y())
