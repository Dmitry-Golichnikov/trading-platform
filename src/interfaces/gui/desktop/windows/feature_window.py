from __future__ import annotations

import yaml
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ..models.feature_model import IndicatorTreeModel
from ..services.dataset_service import DatasetService
from ..services.feature_service import FeatureService
from ..widgets.config_editor import ConfigEditor
from ..widgets.table_widget import VirtualizedTableWidget


class FeatureWindow(QWidget):
    """Модуль конфигурирования и генерации признаков."""

    def __init__(
        self,
        dataset_service: DatasetService,
        feature_service: FeatureService,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.dataset_service = dataset_service
        self.feature_service = feature_service

        self._catalog_model = IndicatorTreeModel()

        self._build_ui()
        self._reload_datasets()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.dataset_combo = QComboBox(self)
        form.addRow("Датасет:", self.dataset_combo)

        refresh_btn = QPushButton("Обновить датасеты", self)
        refresh_btn.clicked.connect(self._reload_datasets)
        form.addWidget(refresh_btn)

        layout.addLayout(form)

        splitter = QSplitter(self)

        self.tree_view = QTreeView(self)
        self.tree_view.setModel(self._catalog_model)
        self.tree_view.doubleClicked.connect(self._handle_indicator_double_click)
        splitter.addWidget(self.tree_view)

        right_panel = QVBoxLayout()
        self.config_editor = ConfigEditor(self)
        self.config_editor.set_text("")
        right_panel.addWidget(self.config_editor)

        button_row = QHBoxLayout()
        generate_btn = QPushButton("Сгенерировать", self)
        generate_btn.clicked.connect(self.generate_features)
        preset_btn = QPushButton("Загрузить минимальный конфиг", self)
        preset_btn.clicked.connect(self._load_minimal_config)
        button_row.addWidget(generate_btn)
        button_row.addWidget(preset_btn)

        right_panel.addLayout(button_row)

        right_widget = QWidget(self)
        right_widget.setLayout(right_panel)

        splitter.addWidget(right_widget)
        splitter.setSizes([300, 500])

        layout.addWidget(splitter)

        self.table = VirtualizedTableWidget(self)
        layout.addWidget(self.table)

    def _reload_datasets(self) -> None:
        datasets = self.dataset_service.list_datasets()
        self.dataset_combo.clear()
        for meta in datasets:
            self.dataset_combo.addItem(f"{meta.ticker}/{meta.timeframe}", meta)

        catalog = self.feature_service.list_indicator_catalog()
        self._catalog_model.populate(catalog)

    def _handle_indicator_double_click(self, index) -> None:  # noqa: D401
        item = self._catalog_model.itemFromIndex(index)
        if item.hasChildren():
            return

        indicator_name = item.text()
        current_text = self.config_editor.editor.toPlainText()
        current_config = self.config_editor.get_config() if current_text.strip() else None

        indicator_config = self.feature_service.build_minimal_config([indicator_name])

        if current_config:
            current_config.setdefault("features", [])
            current_config["features"].extend(indicator_config["features"])
            config_to_dump = current_config
        else:
            config_to_dump = indicator_config

        dumped = yaml.safe_dump(config_to_dump, sort_keys=False, allow_unicode=True)
        self.config_editor.set_text(dumped)

    def _load_minimal_config(self) -> None:
        catalog = self.feature_service.list_indicator_catalog()
        first_category = next(iter(catalog.values()))
        config = self.feature_service.build_minimal_config(first_category[:2])
        self.config_editor.set_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    def generate_features(self) -> None:
        current_index = self.dataset_combo.currentIndex()
        if current_index < 0:
            QMessageBox.warning(self, "Признаки", "Выберите датасет.")
            return

        metadata = self.dataset_combo.currentData()
        data = self.dataset_service.load_preview(metadata, limit=50_000)

        try:
            config = self.config_editor.get_config()
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка конфига", str(exc))
            return

        try:
            result = self.feature_service.generate_features(data, config, dataset_id=str(metadata.dataset_id))
        except Exception as exc:  # pragma: no cover - UI
            QMessageBox.critical(self, "Ошибка генерации", str(exc))
            return

        self.table.set_dataframe(result.features.head(10_000))
        QMessageBox.information(self, "Готово", f"Сгенерировано {result.features.shape[1]} признаков")
