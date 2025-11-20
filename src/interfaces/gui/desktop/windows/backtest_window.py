import json

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.interfaces.gui.desktop.services.backtest_service import BacktestService
from src.interfaces.gui.desktop.services.dataset_service import DatasetService
from src.interfaces.gui.desktop.widgets.chart_widget import ChartWidget


class BacktestWorker(QThread):
    finished = pyqtSignal(object)  # BacktestResult
    error = pyqtSignal(str)

    def __init__(self, service, strategy, data, strat_params, bt_config):
        super().__init__()
        self.service = service
        self.strategy = strategy
        self.data = data
        self.strat_params = strat_params
        self.bt_config = bt_config

    def run(self):
        try:
            result = self.service.run_backtest(self.strategy, self.data, self.strat_params, self.bt_config)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class BacktestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.service = BacktestService()
        self.dataset_service = DatasetService()
        self.data = None

        self.setup_ui()
        self.load_datasets()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Config
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Dataset
        left_layout.addWidget(QLabel("Dataset:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self.on_dataset_changed)
        left_layout.addWidget(self.dataset_combo)

        # Strategy
        left_layout.addWidget(QLabel("Strategy:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(self.service.get_available_strategies())
        left_layout.addWidget(self.strategy_combo)

        # Configs
        left_layout.addWidget(QLabel("Strategy Params (JSON):"))
        self.strat_params_edit = QTextEdit()
        self.strat_params_edit.setText('{"fast_period": 10, "slow_period": 30}')
        left_layout.addWidget(self.strat_params_edit)

        left_layout.addWidget(QLabel("Backtest Config (JSON):"))
        self.bt_config_edit = QTextEdit()
        self.bt_config_edit.setText(json.dumps(self.service.get_default_config(), indent=2))
        left_layout.addWidget(self.bt_config_edit)

        self.run_btn = QPushButton("Run Backtest")
        self.run_btn.clicked.connect(self.run_backtest)
        left_layout.addWidget(self.run_btn)

        # Metrics Table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        left_layout.addWidget(self.metrics_table)

        # Right: Equity Curve
        self.chart = ChartWidget()

        splitter.addWidget(left_widget)
        splitter.addWidget(self.chart)
        layout.addWidget(splitter)

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
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load dataset: {e}")

    def run_backtest(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "No dataset loaded")
            return

        try:
            strat_params = json.loads(self.strat_params_edit.toPlainText())
            bt_config = json.loads(self.bt_config_edit.toPlainText())
        except json.JSONDecodeError:
            QMessageBox.critical(self, "Error", "Invalid JSON")
            return

        # Ensure we have SMAs for SimpleMAStrategy
        # Quick hack for prototype: calculate SMAs here if missing
        if "sma_10" not in self.data.columns:
            self.data["sma_10"] = self.data["close"].rolling(10).mean()
            self.data["sma_30"] = self.data["close"].rolling(30).mean()

        self.run_btn.setEnabled(False)
        self.worker = BacktestWorker(
            self.service, self.strategy_combo.currentText(), self.data, strat_params, bt_config
        )
        self.worker.finished.connect(self.on_backtest_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_backtest_finished(self, result):
        self.run_btn.setEnabled(True)

        # Plot Equity
        # Rename 'equity' to 'close' so ChartWidget can plot it without modification
        equity_data = result.equity_curve.copy()
        equity_data["close"] = equity_data["equity"]
        self.chart.plot_data(equity_data)

        # Show Metrics
        self.metrics_table.setRowCount(len(result.metrics))
        for i, (k, v) in enumerate(result.metrics.items()):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(str(k)))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{v:.4f}"))

    def on_error(self, msg):
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Backtest failed: {msg}")
