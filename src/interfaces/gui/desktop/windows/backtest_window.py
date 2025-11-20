"""
Модуль бэктестинга стратегий.
"""

from typing import Optional

import numpy as np
import pandas as pd

try:
    import pyqtgraph as pg
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QSpacerItem,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    raise ImportError("Требуется установка: pip install PyQt6 pyqtgraph")

from src.interfaces.gui.desktop.utils import log_to_parent
from src.interfaces.gui.desktop.widgets import VirtualizedTableWidget


class BacktestWorker(QThread):
    """
    Worker для бэктестинга в фоновом режиме.
    """

    progress = pyqtSignal(int, str)  # progress, message
    finished = pyqtSignal(pd.DataFrame, dict, str)  # trades, metrics, message
    error = pyqtSignal(str)  # error message

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self) -> None:
        """Выполнить бэктест."""
        try:
            self.progress.emit(25, "Загрузка модели и данных...")
            self.msleep(300)

            self.progress.emit(50, "Генерация сигналов...")
            self.msleep(300)

            self.progress.emit(75, "Расчёт метрик...")
            self.msleep(300)

            # Заглушка - случайные сделки
            num_trades = 50
            dates = pd.date_range("2023-01-01", periods=num_trades, freq="D")

            holding_periods = np.random.randint(1, 10, size=num_trades)
            exit_times = dates + pd.to_timedelta(holding_periods, unit="D")

            trades = pd.DataFrame(
                {
                    "entry_time": dates,
                    "exit_time": exit_times,
                    "direction": np.random.choice(["Long", "Short"], size=num_trades),
                    "entry_price": np.random.uniform(100, 200, size=num_trades),
                    "exit_price": np.random.uniform(100, 200, size=num_trades),
                    "pnl": np.random.uniform(-5, 10, size=num_trades),
                    "pnl_pct": np.random.uniform(-0.05, 0.1, size=num_trades),
                }
            )

            # Заглушка - метрики
            total_pnl = trades["pnl"].sum()
            win_rate = (trades["pnl"] > 0).sum() / len(trades)

            metrics = {
                "total_trades": len(trades),
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "sharpe_ratio": np.random.uniform(0.5, 2.0),
                "sortino_ratio": np.random.uniform(0.5, 2.5),
                "max_drawdown": np.random.uniform(-0.3, -0.1),
                "profit_factor": np.random.uniform(1.0, 3.0),
            }

            self.progress.emit(100, "Завершено!")
            self.finished.emit(trades, metrics, "Бэктест успешно выполнен")

        except Exception as e:
            self.error.emit(f"Ошибка бэктестинга: {str(e)}")


class BacktestWindow(QWidget):
    """
    Окно бэктестинга стратегий.

    Функции:
    - Конфигуратор бэктеста
    - Equity curve
    - Таблица сделок
    - Метрики стратегии
    - Strategy optimization
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Основной layout
        layout = QVBoxLayout(self)

        # Панель управления
        control_panel = QHBoxLayout()

        run_backtest_btn = QPushButton("▶️ Запустить бэктест")
        run_backtest_btn.clicked.connect(self._run_backtest)
        run_backtest_btn.setStyleSheet("QPushButton { background-color: #0e639c; color: white; font-weight: bold; }")
        control_panel.addWidget(run_backtest_btn)

        control_panel.addStretch()

        optimize_btn = QPushButton("🔍 Оптимизация стратегии")
        optimize_btn.clicked.connect(self._optimize_strategy)
        control_panel.addWidget(optimize_btn)

        compare_btn = QPushButton("📊 Сравнение стратегий")
        compare_btn.clicked.connect(self._compare_strategies)
        control_panel.addWidget(compare_btn)

        layout.addLayout(control_panel)

        # Основной сплиттер
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Левая панель - конфигуратор
        config_group = QGroupBox("Конфигурация бэктеста")
        config_layout = QFormLayout(config_group)

        # Модель
        self.model_combo = QComboBox()
        self.model_combo.addItems(["model_1", "model_2", "model_3"])
        config_layout.addRow("Модель:", self.model_combo)

        # Датасет
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["test_set", "validation_set", "full_dataset"])
        config_layout.addRow("Датасет:", self.dataset_combo)

        config_layout.addRow(QLabel(""))  # Разделитель
        config_layout.addRow(QLabel("<b>Параметры стратегии</b>"))

        # Порог входа
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setDecimals(2)
        config_layout.addRow("Порог входа:", self.threshold_spin)

        # Take Profit
        self.take_profit_spin = QDoubleSpinBox()
        self.take_profit_spin.setRange(0.001, 1.0)
        self.take_profit_spin.setValue(0.03)
        self.take_profit_spin.setDecimals(3)
        self.take_profit_spin.setSuffix(" (3%)")
        config_layout.addRow("Take Profit:", self.take_profit_spin)

        # Stop Loss
        self.stop_loss_spin = QDoubleSpinBox()
        self.stop_loss_spin.setRange(0.001, 1.0)
        self.stop_loss_spin.setValue(0.02)
        self.stop_loss_spin.setDecimals(3)
        self.stop_loss_spin.setSuffix(" (2%)")
        config_layout.addRow("Stop Loss:", self.stop_loss_spin)

        # Размер позиции
        self.position_size_spin = QDoubleSpinBox()
        self.position_size_spin.setRange(0.0, 1.0)
        self.position_size_spin.setValue(1.0)
        self.position_size_spin.setDecimals(2)
        config_layout.addRow("Размер позиции:", self.position_size_spin)

        config_layout.addRow(QLabel(""))  # Разделитель
        config_layout.addRow(QLabel("<b>Комиссии и проскальзывание</b>"))

        # Комиссия
        self.commission_spin = QDoubleSpinBox()
        self.commission_spin.setRange(0.0, 0.1)
        self.commission_spin.setValue(0.001)
        self.commission_spin.setDecimals(4)
        self.commission_spin.setSuffix(" (0.1%)")
        config_layout.addRow("Комиссия:", self.commission_spin)

        # Проскальзывание
        self.slippage_spin = QDoubleSpinBox()
        self.slippage_spin.setRange(0.0, 0.1)
        self.slippage_spin.setValue(0.0005)
        self.slippage_spin.setDecimals(4)
        self.slippage_spin.setSuffix(" (0.05%)")
        config_layout.addRow("Проскальзывание:", self.slippage_spin)

        config_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        main_splitter.addWidget(config_group)

        # Правая панель - результаты
        tabs = QTabWidget()

        # Вкладка Equity Curve
        equity_tab = QWidget()
        equity_layout = QVBoxLayout(equity_tab)
        equity_layout.setContentsMargins(0, 0, 0, 0)

        self.equity_plot = pg.PlotWidget(title="Equity Curve")
        self.equity_plot.setLabel("left", "Equity")
        self.equity_plot.setLabel("bottom", "Time")
        self.equity_plot.addLegend()
        self.equity_plot.showGrid(x=True, y=True, alpha=0.3)

        self.equity_curve = self.equity_plot.plot(pen=pg.mkPen(color="g", width=2), name="Strategy")
        self.benchmark_curve = self.equity_plot.plot(
            pen=pg.mkPen(color="gray", width=1, style=Qt.PenStyle.DashLine), name="Buy & Hold"
        )

        equity_layout.addWidget(self.equity_plot)

        tabs.addTab(equity_tab, "📈 Equity Curve")

        # Вкладка сделок
        trades_tab = QWidget()
        trades_layout = QVBoxLayout(trades_tab)
        trades_layout.setContentsMargins(0, 0, 0, 0)

        self.trades_table = VirtualizedTableWidget()
        trades_layout.addWidget(self.trades_table)

        tabs.addTab(trades_tab, "📋 Сделки")

        # Вкладка метрик
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setPlaceholderText("Метрики стратегии будут отображаться здесь...")
        metrics_layout.addWidget(self.metrics_text)

        tabs.addTab(metrics_tab, "📊 Метрики")

        main_splitter.addWidget(tabs)

        # Пропорции
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)

        layout.addWidget(main_splitter)

        # Worker
        self.worker: Optional[BacktestWorker] = None

    def _run_backtest(self) -> None:
        """Запустить бэктест."""
        # Собрать конфигурацию
        config = {
            "model": self.model_combo.currentText(),
            "dataset": self.dataset_combo.currentText(),
            "threshold": self.threshold_spin.value(),
            "take_profit": self.take_profit_spin.value(),
            "stop_loss": self.stop_loss_spin.value(),
            "position_size": self.position_size_spin.value(),
            "commission": self.commission_spin.value(),
            "slippage": self.slippage_spin.value(),
        }

        # Создать progress dialog
        from PyQt6.QtWidgets import QProgressDialog

        progress = QProgressDialog("Выполнение бэктеста...", "Отмена", 0, 100, self)
        progress.setWindowTitle("Бэктест")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        # Запустить worker
        self.worker = BacktestWorker(config)

        def on_progress(value: int, message: str) -> None:
            progress.setValue(value)
            progress.setLabelText(message)
            log_to_parent(self, f"[PROGRESS] {message}")

        def on_finished(trades: pd.DataFrame, metrics: dict, message: str) -> None:
            progress.close()
            self._display_results(trades, metrics)
            QMessageBox.information(self, "Бэктест завершён", message)
            log_to_parent(self, f"[OK] {message}")

        def on_error(error: str) -> None:
            progress.close()
            QMessageBox.critical(self, "Ошибка бэктеста", error)
            log_to_parent(self, f"[ERROR] {error}")

        self.worker.progress.connect(on_progress)
        self.worker.finished.connect(on_finished)
        self.worker.error.connect(on_error)
        progress.canceled.connect(self.worker.terminate)

        self.worker.start()

    def _display_results(self, trades: pd.DataFrame, metrics: dict) -> None:
        """
        Отобразить результаты бэктеста.

        Args:
            trades: DataFrame со сделками
            metrics: Словарь с метриками
        """
        # Отобразить таблицу сделок
        self.trades_table.set_data(trades)

        # Отобразить equity curve
        cumulative_pnl = trades["pnl"].cumsum().values
        self.equity_curve.setData(np.arange(len(cumulative_pnl)), cumulative_pnl)

        # Benchmark (Buy & Hold) - простая линия
        benchmark = np.linspace(0, cumulative_pnl[-1] * 0.7, len(cumulative_pnl))
        self.benchmark_curve.setData(np.arange(len(benchmark)), benchmark)

        # Отобразить метрики
        metrics_text = "Результаты бэктестинга\n\n"
        metrics_text += f"Всего сделок: {metrics['total_trades']}\n"
        metrics_text += f"Total PnL: {metrics['total_pnl']:.2f}\n"
        metrics_text += f"Win Rate: {metrics['win_rate']:.2%}\n"
        metrics_text += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        metrics_text += f"Sortino Ratio: {metrics['sortino_ratio']:.2f}\n"
        metrics_text += f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
        metrics_text += f"Profit Factor: {metrics['profit_factor']:.2f}\n"

        self.metrics_text.setPlainText(metrics_text)

    def _optimize_strategy(self) -> None:
        """Оптимизация параметров стратегии."""
        # TODO: Реализовать grid search / random search
        QMessageBox.information(
            self,
            "В разработке",
            "Оптимизация стратегии будет реализована в следующей версии.",
        )

    def _compare_strategies(self) -> None:
        """Сравнение нескольких стратегий."""
        # TODO: Реализовать сравнение
        QMessageBox.information(
            self,
            "В разработке",
            "Сравнение стратегий будет реализовано в следующей версии.",
        )
