"""
Высокопроизводительный виджет для отображения свечных графиков и индикаторов.
"""

from typing import Optional, cast

import numpy as np
import pandas as pd

try:
    import pyqtgraph as pg
    from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
except ImportError:
    raise ImportError("Требуется установка: pip install PyQt6 pyqtgraph")


class CandlestickItem(pg.GraphicsObject):
    """
    Свечной график для pyqtgraph.

    Оптимизирован для больших объёмов данных.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Инициализировать свечной график.

        Args:
            data: DataFrame с колонками [timestamp, open, high, low, close, volume]
        """
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.generatePicture()

    def generatePicture(self) -> None:
        """Сгенерировать изображение свечей."""
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)

        # Цвета свечей
        green = pg.mkBrush(50, 200, 50, 200)
        red = pg.mkBrush(200, 50, 50, 200)
        green_pen = pg.mkPen(50, 200, 50)
        red_pen = pg.mkPen(200, 50, 50)

        w = 0.4  # Ширина свечи

        for position, row in enumerate(self.data.itertuples(index=False)):
            x_center = float(position)
            open_price = cast(float, row.open)
            close_price = cast(float, row.close)
            high_price = cast(float, row.high)
            low_price = cast(float, row.low)

            # Цвет свечи (зелёный если рост, красный если падение)
            if close_price > open_price:
                p.setPen(green_pen)
                p.setBrush(green)
            else:
                p.setPen(red_pen)
                p.setBrush(red)

            # Тело свечи (прямоугольник)
            body_height = abs(close_price - open_price)
            body_y = min(open_price, close_price)
            p.drawRect(pg.QtCore.QRectF(x_center - w / 2, body_y, w, body_height))

            # Тени (линии high-low)
            p.drawLine(
                pg.QtCore.QPointF(x_center, low_price),
                pg.QtCore.QPointF(x_center, high_price),
            )

        p.end()

    def paint(self, p: pg.QtGui.QPainter, *args) -> None:
        """Отрисовать график."""
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self) -> pg.QtCore.QRectF:
        """Вернуть ограничивающий прямоугольник."""
        return pg.QtCore.QRectF(self.picture.boundingRect())


class ChartWidget(QWidget):
    """
    Виджет для отображения свечных графиков с индикаторами.

    Особенности:
    - OpenGL-акселерация (опционально)
    - Zoom/Pan с помощью мыши
    - Crosshair с координатами
    - Наложение индикаторов
    - Метки для Long/Short сигналов
    """

    def __init__(self, parent: Optional[QWidget] = None, use_opengl: bool = False):
        super().__init__(parent)

        # Основной layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Панель управления
        control_panel = QHBoxLayout()
        control_panel.setSpacing(5)

        self.zoom_in_btn = QPushButton("🔍+")
        self.zoom_in_btn.setMaximumWidth(40)
        self.zoom_in_btn.setToolTip("Увеличить")
        control_panel.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = QPushButton("🔍−")
        self.zoom_out_btn.setMaximumWidth(40)
        self.zoom_out_btn.setToolTip("Уменьшить")
        control_panel.addWidget(self.zoom_out_btn)

        self.reset_btn = QPushButton("🔄")
        self.reset_btn.setMaximumWidth(40)
        self.reset_btn.setToolTip("Сбросить масштаб")
        control_panel.addWidget(self.reset_btn)

        control_panel.addStretch()

        self.info_label = QLabel("")
        control_panel.addWidget(self.info_label)

        layout.addLayout(control_panel)

        # График (pyqtgraph)
        pg.setConfigOptions(antialias=True)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "Цена")
        self.plot_widget.setLabel("bottom", "Время")

        # Включить OpenGL если запрошено
        if use_opengl:
            try:
                self.plot_widget.useOpenGL(True)
            except Exception as e:
                print(f"Не удалось включить OpenGL: {e}")

        layout.addWidget(self.plot_widget)

        # Данные
        self.candlestick_item: Optional[CandlestickItem] = None
        self.indicator_plots: dict[str, pg.PlotDataItem] = {}

        # Подключить сигналы
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        self.zoom_out_btn.clicked.connect(self._zoom_out)
        self.reset_btn.clicked.connect(self._reset_view)

    def plot_candlesticks(self, data: pd.DataFrame) -> None:
        """
        Отобразить свечной график.

        Args:
            data: DataFrame с колонками [timestamp, open, high, low, close, volume]
        """
        # Очистить предыдущий график
        self.plot_widget.clear()
        self.indicator_plots.clear()

        # Подготовить данные
        if "timestamp" in data.columns:
            # Использовать индекс вместо timestamp для упрощения
            data = data.copy()
            data = data.reset_index(drop=True)

        # Создать и добавить свечной график
        self.candlestick_item = CandlestickItem(data)
        self.plot_widget.addItem(self.candlestick_item)

        # Автоматически подогнать масштаб
        self._reset_view()

        self.info_label.setText(f"Отображено баров: {len(data)}")

    def add_indicator_overlay(
        self,
        indicator: pd.Series,
        name: str,
        color: str = "blue",
        width: int = 2,
    ) -> None:
        """
        Добавить индикатор на график.

        Args:
            indicator: Series с значениями индикатора
            name: Имя индикатора
            color: Цвет линии
            width: Толщина линии
        """
        # Подготовить данные
        x = np.arange(len(indicator))
        y = indicator.to_numpy(dtype=float, copy=False)

        # Удалить NaN
        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        # Создать линию
        pen = pg.mkPen(color=color, width=width)
        plot_item = self.plot_widget.plot(x, y, pen=pen, name=name)

        self.indicator_plots[name] = plot_item

    def mark_labels(
        self,
        labels: pd.Series,
        marker_size: int = 10,
    ) -> None:
        """
        Добавить метки Long/Short на график.

        Args:
            labels: Series с метками (-1: Short, 0: Hold, 1: Long)
            marker_size: Размер маркера
        """
        if self.candlestick_item is None or self.candlestick_item.data.empty:
            return

        data = self.candlestick_item.data

        # Найти Long сигналы
        long_mask = (labels == 1).to_numpy(dtype=bool, copy=False)
        if long_mask.any():
            long_x = np.where(long_mask)[0]
            long_y = data.loc[labels == 1, "low"].to_numpy(dtype=float, copy=False)
            self.plot_widget.plot(
                long_x,
                long_y,
                pen=None,
                symbol="t",  # Треугольник вверх
                symbolBrush=(50, 200, 50),
                symbolSize=marker_size,
                name="Long",
            )

        # Найти Short сигналы
        short_mask = (labels == -1).to_numpy(dtype=bool, copy=False)
        if short_mask.any():
            short_x = np.where(short_mask)[0]
            short_y = data.loc[labels == -1, "high"].to_numpy(dtype=float, copy=False)
            self.plot_widget.plot(
                short_x,
                short_y,
                pen=None,
                symbol="t1",  # Треугольник вниз
                symbolBrush=(200, 50, 50),
                symbolSize=marker_size,
                name="Short",
            )

    def _zoom_in(self) -> None:
        """Увеличить масштаб."""
        self.plot_widget.getViewBox().scaleBy((0.8, 0.8))

    def _zoom_out(self) -> None:
        """Уменьшить масштаб."""
        self.plot_widget.getViewBox().scaleBy((1.25, 1.25))

    def _reset_view(self) -> None:
        """Сбросить масштаб к исходному."""
        self.plot_widget.autoRange()
