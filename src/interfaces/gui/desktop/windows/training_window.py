from __future__ import annotations

from pathlib import Path

import yaml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..services.dataset_service import DatasetService
from ..services.feature_service import FeatureService
from ..services.labeling_service import LabelingService
from ..services.training_service import TrainingService
from ..widgets.config_editor import ConfigEditor
from ..widgets.metrics_widget import MetricsWidget
from ..widgets.table_widget import VirtualizedTableWidget
from ..workers.training_worker import TrainingWorker


class TrainingWindow(QWidget):
    """Конфигуратор обучения и монитор."""

    def __init__(
        self,
        dataset_service: DatasetService,
        training_service: TrainingService,
        feature_service: FeatureService,
        labeling_service: LabelingService,
        log_widget,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.dataset_service = dataset_service
        self.training_service = training_service
        self.feature_service = feature_service
        self.labeling_service = labeling_service
        self.log_widget = log_widget

        self.worker: TrainingWorker | None = None

        self._build_ui()
        self._reload_datasets()
        self._load_template()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Left panel – config
        left_panel = QVBoxLayout()
        form = QFormLayout()
        self.dataset_combo = QComboBox(self)
        form.addRow("Датасет:", self.dataset_combo)
        left_panel.addLayout(form)

        button_row = QHBoxLayout()
        load_btn = QPushButton("Загрузить конфиг...", self)
        load_btn.clicked.connect(self._load_from_file)
        save_btn = QPushButton("Сохранить конфиг...", self)
        save_btn.clicked.connect(self._save_to_file)
        button_row.addWidget(load_btn)
        button_row.addWidget(save_btn)
        left_panel.addLayout(button_row)

        self.config_editor = ConfigEditor(self)
        left_panel.addWidget(self.config_editor)

        start_btn = QPushButton("Запустить обучение", self)
        start_btn.clicked.connect(self.start_training)
        left_panel.addWidget(start_btn)

        left_widget = QWidget(self)
        left_widget.setLayout(left_panel)
        splitter.addWidget(left_widget)

        # Right panel – monitor
        right_panel = QVBoxLayout()
        self.status_label = QLabel("Готово", self)
        right_panel.addWidget(self.status_label)

        self.metrics_widget = MetricsWidget(self)
        right_panel.addWidget(self.metrics_widget)

        self.trials_table = VirtualizedTableWidget(self)
        right_panel.addWidget(self.trials_table)

        right_widget = QWidget(self)
        right_widget.setLayout(right_panel)
        splitter.addWidget(right_widget)
        splitter.setSizes([500, 700])

        layout.addWidget(splitter)

    def _reload_datasets(self) -> None:
        datasets = self.dataset_service.list_datasets()
        self.dataset_combo.clear()
        for meta in datasets:
            self.dataset_combo.addItem(f"{meta.ticker}/{meta.timeframe}", meta)

    def _load_template(self) -> None:
        template = {
            "data_path": "artifacts/data/example.parquet",
            "target_column": "label",
            "exclude_columns": ["timestamp"],
            "model_type": "lightgbm",
            "model_config": {"num_leaves": 64, "learning_rate": 0.05},
            "trainer_config": {
                "experiment_name": "gui_training",
                "verbose": True,
            },
        }
        self.config_editor.set_text(yaml.safe_dump(template, sort_keys=False, allow_unicode=True))

    def _load_from_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбор конфига", "configs/models", "YAML (*.yaml *.yml)")
        if not file_path:
            return
        config = self.training_service.load_config(Path(file_path))
        self.config_editor.set_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    def _save_to_file(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить конфиг", "configs/models/gui_training.yaml", "YAML (*.yaml)"
        )
        if not file_path:
            return
        config = self.config_editor.get_config()
        with open(file_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)

    def start_training(self) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Обучение", "Процесс уже запущен")
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

        self.worker = TrainingWorker(self.training_service, config)
        self.worker.progress.connect(self._on_progress)
        self.worker.metrics.connect(self.metrics_widget.update_metrics)
        self.worker.history.connect(self.metrics_widget.update_history)
        self.worker.pipeline_update.connect(self._on_pipeline_update)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()
        self.status_label.setText("Обучение запущено...")

    def _on_progress(self, value: float, message: str) -> None:
        self.status_label.setText(f"{message} ({value*100:.0f}%)")
        self.log_widget.append_line(message)

    def _on_pipeline_update(self, payload: dict) -> None:
        self.log_widget.append_line(f"{payload['pipeline']} → {payload['step']} [{payload['status']}]")

    def _on_finished(self, result) -> None:
        self.status_label.setText("Обучение завершено")
        self.log_widget.append_line("Обучение завершено")

    def _on_failed(self, message: str) -> None:
        self.status_label.setText("Ошибка")
        QMessageBox.critical(self, "Обучение", message)
        self.log_widget.append_line(f"Ошибка: {message}")
