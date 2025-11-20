"""
Модуль разметки данных.
"""

from typing import Optional

import pandas as pd

try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QProgressDialog,
        QPushButton,
        QSizePolicy,
        QSpacerItem,
        QSpinBox,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    raise ImportError("Требуется установка: pip install PyQt6")

from src.interfaces.gui.desktop.utils import log_to_parent
from src.interfaces.gui.desktop.widgets import ChartWidget, VirtualizedTableWidget


class LabelingWorker(QThread):
    """
    Worker для разметки данных в фоновом режиме.
    """

    progress = pyqtSignal(int, str)  # progress, message
    finished = pyqtSignal(pd.DataFrame, str)  # labels, message
    error = pyqtSignal(str)  # error message

    def __init__(self, data: pd.DataFrame, config: dict):
        super().__init__()
        self.data = data
        self.config = config

    def run(self) -> None:
        """Выполнить разметку."""
        try:
            # TODO: Интегрировать с src.labeling
            self.progress.emit(25, "Подготовка данных...")
            self.msleep(300)

            self.progress.emit(50, "Применение метода разметки...")
            self.msleep(300)

            self.progress.emit(75, "Применение фильтров...")
            self.msleep(300)

            # Заглушка - случайные метки
            import numpy as np

            labels = pd.DataFrame(
                {
                    "timestamp": self.data["timestamp"],
                    "label": np.random.choice([-1, 0, 1], size=len(self.data)),
                    "barrier_hit": np.random.choice(["take_profit", "stop_loss", "time"], size=len(self.data)),
                    "holding_period": np.random.randint(1, 50, size=len(self.data)),
                    "realized_return": np.random.uniform(-0.05, 0.05, size=len(self.data)),
                }
            )

            self.progress.emit(100, "Завершено!")
            self.finished.emit(labels, "Разметка успешно выполнена")

        except Exception as e:
            self.error.emit(f"Ошибка разметки: {str(e)}")


class LabelingWindow(QWidget):
    """
    Окно разметки данных.

    Функции:
    - Конфигуратор методов разметки
    - Просмотр графика с метками
    - Статистика распределения классов
    - Балансировка классов
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Основной layout
        layout = QVBoxLayout(self)

        # Панель управления
        control_panel = QHBoxLayout()

        apply_labeling_btn = QPushButton("▶️ Применить разметку")
        apply_labeling_btn.clicked.connect(self._apply_labeling)
        apply_labeling_btn.setStyleSheet("QPushButton { background-color: #0e639c; color: white; font-weight: bold; }")
        control_panel.addWidget(apply_labeling_btn)

        control_panel.addStretch()

        show_stats_btn = QPushButton("📊 Статистика")
        show_stats_btn.clicked.connect(self._show_statistics)
        control_panel.addWidget(show_stats_btn)

        balance_btn = QPushButton("⚖️ Балансировка")
        balance_btn.clicked.connect(self._balance_classes)
        control_panel.addWidget(balance_btn)

        layout.addLayout(control_panel)

        # Основной сплиттер
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Левая панель - конфигуратор
        config_group = QGroupBox("Конфигурация разметки")
        config_layout = QFormLayout(config_group)

        # Метод разметки
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Triple Barrier", "Horizon", "Regression Targets"])
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        config_layout.addRow("Метод:", self.method_combo)

        # Параметры Triple Barrier
        self.tp_threshold = QDoubleSpinBox()
        self.tp_threshold.setRange(0.001, 1.0)
        self.tp_threshold.setValue(0.02)
        self.tp_threshold.setDecimals(3)
        self.tp_threshold.setSuffix(" (2%)")
        config_layout.addRow("Take Profit:", self.tp_threshold)

        self.sl_threshold = QDoubleSpinBox()
        self.sl_threshold.setRange(0.001, 1.0)
        self.sl_threshold.setValue(0.02)
        self.sl_threshold.setDecimals(3)
        self.sl_threshold.setSuffix(" (2%)")
        config_layout.addRow("Stop Loss:", self.sl_threshold)

        self.time_barrier = QSpinBox()
        self.time_barrier.setRange(1, 1000)
        self.time_barrier.setValue(50)
        self.time_barrier.setSuffix(" баров")
        config_layout.addRow("Time Barrier:", self.time_barrier)

        # Режим торговли
        self.trade_mode_combo = QComboBox()
        self.trade_mode_combo.addItems(["Long + Short", "Long Only", "Short Only"])
        config_layout.addRow("Режим:", self.trade_mode_combo)

        # Фильтры
        config_layout.addRow(QLabel(""))  # Разделитель
        config_layout.addRow(QLabel("<b>Фильтры</b>"))

        self.min_return = QDoubleSpinBox()
        self.min_return.setRange(0.0, 1.0)
        self.min_return.setValue(0.005)
        self.min_return.setDecimals(3)
        self.min_return.setSuffix(" (0.5%)")
        config_layout.addRow("Мин. доходность:", self.min_return)

        config_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        main_splitter.addWidget(config_group)

        # Правая панель - просмотр результатов
        view_group = QGroupBox("Просмотр результатов")
        view_layout = QVBoxLayout(view_group)

        # Селектор режима просмотра
        view_selector = QHBoxLayout()
        view_selector.addWidget(QLabel("Режим:"))

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["График с метками", "Таблица меток"])
        self.view_mode_combo.currentIndexChanged.connect(self._switch_view_mode)
        view_selector.addWidget(self.view_mode_combo)

        view_selector.addStretch()
        view_layout.addLayout(view_selector)

        # Виджеты просмотра
        self.chart_widget = ChartWidget()
        self.table_widget = VirtualizedTableWidget()

        view_layout.addWidget(self.chart_widget)
        view_layout.addWidget(self.table_widget)

        # По умолчанию показываем график
        self.table_widget.hide()

        main_splitter.addWidget(view_group)

        # Пропорции
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)

        layout.addWidget(main_splitter)

        # Данные
        self.current_data: Optional[pd.DataFrame] = None
        self.current_labels: Optional[pd.DataFrame] = None
        self.worker: Optional[LabelingWorker] = None

    def _on_method_changed(self, index: int) -> None:
        """
        Обработчик изменения метода разметки.

        Args:
            index: Индекс выбранного метода
        """
        # TODO: Показать/скрыть соответствующие параметры
        pass

    def _apply_labeling(self) -> None:
        """Применить разметку к данным."""
        # TODO: Получить данные из DatasetWindow
        if self.current_data is None:
            QMessageBox.warning(
                self,
                "Нет данных",
                "Сначала загрузите датасет в модуле 'Данные'.",
            )
            return

        # Собрать конфигурацию
        config = {
            "method": self.method_combo.currentText(),
            "tp_threshold": self.tp_threshold.value(),
            "sl_threshold": self.sl_threshold.value(),
            "time_barrier": self.time_barrier.value(),
            "trade_mode": self.trade_mode_combo.currentText(),
            "min_return": self.min_return.value(),
        }

        # Создать progress dialog
        progress = QProgressDialog("Разметка данных...", "Отмена", 0, 100, self)
        progress.setWindowTitle("Разметка данных")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        # Запустить worker
        self.worker = LabelingWorker(self.current_data, config)

        def on_progress(value: int, message: str) -> None:
            progress.setValue(value)
            progress.setLabelText(message)
            log_to_parent(self, f"[PROGRESS] {message}")

        def on_finished(labels: pd.DataFrame, message: str) -> None:
            progress.close()
            self.current_labels = labels
            self._display_results()
            QMessageBox.information(self, "Разметка завершена", message)
            log_to_parent(self, f"[OK] {message}")

        def on_error(error: str) -> None:
            progress.close()
            QMessageBox.critical(self, "Ошибка разметки", error)
            log_to_parent(self, f"[ERROR] {error}")

        self.worker.progress.connect(on_progress)
        self.worker.finished.connect(on_finished)
        self.worker.error.connect(on_error)
        progress.canceled.connect(self.worker.terminate)

        self.worker.start()

    def _display_results(self) -> None:
        """Отобразить результаты разметки."""
        if self.current_labels is None or self.current_data is None:
            return

        mode = self.view_mode_combo.currentText()

        if mode == "График с метками":
            # Показать график с метками
            self.chart_widget.plot_candlesticks(self.current_data)
            self.chart_widget.mark_labels(self.current_labels["label"])
        elif mode == "Таблица меток":
            # Показать таблицу меток
            self.table_widget.set_data(self.current_labels)

    def _switch_view_mode(self, index: int) -> None:
        """
        Переключить режим просмотра.

        Args:
            index: Индекс выбранного режима
        """
        if index == 0:  # График
            self.chart_widget.show()
            self.table_widget.hide()
        elif index == 1:  # Таблица
            self.chart_widget.hide()
            self.table_widget.show()

        # Обновить отображение
        self._display_results()

    def _show_statistics(self) -> None:
        """Показать статистику распределения классов."""
        if self.current_labels is None:
            QMessageBox.warning(
                self,
                "Нет меток",
                "Сначала примените разметку.",
            )
            return

        # Посчитать статистику
        label_counts = self.current_labels["label"].value_counts().to_dict()
        total = len(self.current_labels)

        stats_text = "Распределение классов:\n\n"
        for label, count in sorted(label_counts.items()):
            label_name = {-1: "Short", 0: "Hold", 1: "Long"}.get(label, str(label))
            percentage = (count / total) * 100
            stats_text += f"{label_name}: {count} ({percentage:.1f}%)\n"

        stats_text += f"\nВсего: {total}"

        QMessageBox.information(self, "Статистика", stats_text)

    def _balance_classes(self) -> None:
        """Балансировка классов."""
        if self.current_labels is None:
            QMessageBox.warning(
                self,
                "Нет меток",
                "Сначала примените разметку.",
            )
            return

        # TODO: Интегрировать с src.labeling.balancing
        QMessageBox.information(
            self,
            "В разработке",
            "Балансировка классов будет реализована в следующей версии.",
        )

    def set_data(self, data: pd.DataFrame) -> None:
        """
        Установить данные для разметки.

        Args:
            data: DataFrame с OHLCV данными
        """
        self.current_data = data
