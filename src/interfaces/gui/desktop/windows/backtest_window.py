from __future__ import annotations

import pandas as pd
import yaml
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..services.backtest_service import BacktestService
from ..services.dataset_service import DatasetService
from ..widgets.chart_widget import ChartWidget
from ..widgets.config_editor import ConfigEditor
from ..widgets.table_widget import VirtualizedTableWidget
from ..workers.backtest_worker import BacktestWorker


class BacktestWindow(QWidget):
    """Конфигуратор бэктестов."""

    def __init__(
        self,
        dataset_service: DatasetService,
        backtest_service: BacktestService,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.dataset_service = dataset_service
        self.backtest_service = backtest_service
        self.worker: BacktestWorker | None = None

        self._build_ui()
        self._reload_datasets()
        self._load_template()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.dataset_combo = QComboBox(self)
        form.addRow("Датасет:", self.dataset_combo)

        choose_model_btn = QPushButton("Выбрать модель...", self)
        choose_model_btn.clicked.connect(self._choose_model_file)
        form.addRow("Модель:", choose_model_btn)

        layout.addLayout(form)

        splitter = QSplitter(self)
        self.config_editor = ConfigEditor(self)
        splitter.addWidget(self.config_editor)

        right_panel = QVBoxLayout()
        self.chart = ChartWidget(self)
        right_panel.addWidget(self.chart)
        self.trades_table = VirtualizedTableWidget(self)
        right_panel.addWidget(self.trades_table)

        right_widget = QWidget(self)
        right_widget.setLayout(right_panel)
        splitter.addWidget(right_widget)
        splitter.setSizes([500, 500])

        layout.addWidget(splitter)

        controls = QVBoxLayout()
        run_btn = QPushButton("Запустить бэктест", self)
        run_btn.clicked.connect(self.run_backtest)
        controls.addWidget(run_btn)
        self.status_label = QLabel("Готово", self)
        controls.addWidget(self.status_label)
        layout.addLayout(controls)

    def _reload_datasets(self) -> None:
        datasets = self.dataset_service.list_datasets()
        self.dataset_combo.clear()
        for meta in datasets:
            self.dataset_combo.addItem(f"{meta.ticker}/{meta.timeframe}", meta)

    def _load_template(self) -> None:
        template = {
            "data_path": "artifacts/features/latest.parquet",
            "model_path": "artifacts/models/latest.pkl",
            "strategy_type": "model_based",
            "strategy_config": {"threshold": 0.55},
            "initial_capital": 100000,
        }
        self.config_editor.set_text(yaml.safe_dump(template, sort_keys=False, allow_unicode=True))

    def _choose_model_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Модель", "artifacts/models", "Pickle (*.pkl)")
        if not file_path:
            return
        config = self.config_editor.get_config()
        config["model_path"] = file_path
        self.config_editor.set_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    def run_backtest(self) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Backtest", "Процесс уже запущен")
            return

        try:
            config = self.config_editor.get_config()
        except Exception as exc:
            QMessageBox.critical(self, "Конфиг", str(exc))
            return

        dataset_index = self.dataset_combo.currentIndex()
        if dataset_index >= 0:
            metadata = self.dataset_combo.currentData()
            config.setdefault("dataset", {})["id"] = str(metadata.dataset_id)

        self.worker = BacktestWorker(self.backtest_service, config)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, value: float, message: str) -> None:
        self.status_label.setText(f"{message} ({value*100:.0f}%)")

    def _on_finished(self, result) -> None:
        metrics = result.metrics
        QMessageBox.information(self, "Backtest", f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
        equity_path = result.artifacts.get("equity_path")
        if equity_path:
            equity_df = pd.read_parquet(equity_path)
            self.chart.plot_equity(equity_df["equity"])
        trades_path = result.artifacts.get("trades_path")
        if trades_path:
            trades_df = pd.read_parquet(trades_path)
            self.trades_table.set_dataframe(trades_df)
        self.status_label.setText("Готово")

    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Backtest", message)
