from __future__ import annotations

import yaml
from PySide6.QtWidgets import (
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ..models.experiment_model import ExperimentTableModel
from ..services.dataset_service import DatasetService
from ..services.experiment_service import ExperimentItem, ExperimentPlan, ExperimentService
from ..widgets.config_editor import ConfigEditor


class ExperimentWindow(QWidget):
    """UI для пакетных экспериментов."""

    def __init__(
        self,
        dataset_service: DatasetService,
        experiment_service: ExperimentService,
        log_widget,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.dataset_service = dataset_service
        self.experiment_service = experiment_service
        self.log_widget = log_widget

        self.results_model = ExperimentTableModel()

        self._build_ui()
        self._load_template()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        splitter = QSplitter(self)

        self.plan_editor = ConfigEditor(self)
        splitter.addWidget(self.plan_editor)

        right_panel = QVBoxLayout()
        self.progress_bar = QProgressBar(self)
        right_panel.addWidget(self.progress_bar)

        self.results_view = QTableView(self)
        self.results_view.setModel(self.results_model)
        right_panel.addWidget(self.results_view)

        run_btn = QPushButton("Запустить эксперименты", self)
        run_btn.clicked.connect(self.run_experiments)
        right_panel.addWidget(run_btn)

        right_widget = QWidget(self)
        right_widget.setLayout(right_panel)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 400])

        layout.addWidget(splitter)

    def _load_template(self) -> None:
        datasets = self.dataset_service.list_datasets()[:2]
        dataset_items = [
            {
                "name": f"{meta.ticker}_{meta.timeframe}",
                "overrides": {
                    "data_path": f"artifacts/data/{meta.ticker}/{meta.timeframe}/dataset.parquet",
                    "dataset": {"id": str(meta.dataset_id)},
                },
            }
            for meta in datasets
        ]

        template = {
            "base_training_config": {
                "data_path": "",
                "model_type": "lightgbm",
                "trainer_config": {"experiment_name": "gui_experiments"},
            },
            "base_backtest_config": {
                "data_path": "",
                "model_path": "artifacts/models/latest.pkl",
                "strategy_type": "model_based",
            },
            "datasets": dataset_items,
            "feature_sets": [{"name": "default_features", "overrides": {}}],
            "labeling_setups": [{"name": "tb_default", "overrides": {}}],
            "models": [{"name": "lgbm_default", "overrides": {}}],
            "strategies": [{"name": "model_strategy", "overrides": {}}],
        }

        self.plan_editor.set_text(yaml.safe_dump(template, sort_keys=False, allow_unicode=True))

    def run_experiments(self) -> None:
        try:
            plan_dict = self.plan_editor.get_config()
            plan = self._parse_plan(plan_dict)
        except Exception as exc:
            QMessageBox.critical(self, "Эксперименты", f"Некорректный план: {exc}")
            return

        try:
            results = self.experiment_service.run_plan(
                plan,
                progress_callback=self._on_progress,
            )
        except Exception as exc:  # pragma: no cover - runtime errors
            QMessageBox.critical(self, "Эксперименты", str(exc))
            return

        self.results_model.set_results(results)
        QMessageBox.information(self, "Эксперименты", f"Завершено {len(results)} комбинаций")

    def _parse_plan(self, plan_dict: dict) -> ExperimentPlan:
        def _items(key: str) -> list[ExperimentItem]:
            items = plan_dict.get(key, [])
            return [ExperimentItem(name=item["name"], overrides=item.get("overrides", {})) for item in items]

        return ExperimentPlan(
            base_training_config=plan_dict["base_training_config"],
            base_backtest_config=plan_dict["base_backtest_config"],
            datasets=_items("datasets"),
            feature_sets=_items("feature_sets"),
            labeling_setups=_items("labeling_setups"),
            models=_items("models"),
            strategies=_items("strategies"),
        )

    def _on_progress(self, update) -> None:
        self.progress_bar.setMaximum(update.total)
        self.progress_bar.setValue(update.index)
        self.log_widget.append_line(f"Эксперимент {update.index}/{update.total}: {update.status}")
