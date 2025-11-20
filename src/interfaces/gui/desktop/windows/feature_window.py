import json

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.features.config_parser import FeatureConfig, FeatureConfigItem
from src.interfaces.gui.desktop.services.feature_service import FeatureService


class FeatureWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.service = FeatureService()
        self.current_config = FeatureConfig(features=[])
        self.updating_props = False

        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Feature List
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.feature_list = QListWidget()
        self.feature_list.currentRowChanged.connect(self.on_feature_selected)

        left_layout.addWidget(QLabel("Selected Features"))
        left_layout.addWidget(self.feature_list)

        # Buttons
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add Indicator")
        self.add_btn.clicked.connect(self.add_indicator)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.remove_feature)

        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        left_layout.addLayout(btn_layout)

        # Middle: Properties
        self.props_group = QGroupBox("Properties")
        self.props_layout = QFormLayout()
        self.props_group.setLayout(self.props_layout)

        # Right: Actions
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("Actions"))
        self.save_btn = QPushButton("Save Config")
        # self.save_btn.clicked.connect(self.save_config)
        right_layout.addWidget(self.save_btn)
        right_layout.addStretch()

        splitter.addWidget(left_widget)
        splitter.addWidget(self.props_group)
        splitter.addWidget(right_widget)

        layout.addWidget(splitter)

    def add_indicator(self):
        indicators = self.service.get_available_indicators()
        item, ok = QInputDialog.getItem(self, "Select Indicator", "Indicator:", indicators, 0, False)
        if ok and item:
            feature_item = FeatureConfigItem(type="indicator", name=item, params={"window": 14})
            self.current_config.features.append(feature_item)
            self.feature_list.addItem(f"{item} (Indicator)")
            self.feature_list.setCurrentRow(len(self.current_config.features) - 1)

    def remove_feature(self):
        row = self.feature_list.currentRow()
        if row >= 0:
            self.feature_list.takeItem(row)
            self.current_config.features.pop(row)
            self.clear_props()

    def on_feature_selected(self, row):
        if row < 0 or row >= len(self.current_config.features):
            self.clear_props()
            return

        feature = self.current_config.features[row]
        self.show_props(feature)

    def clear_props(self):
        while self.props_layout.count():
            child = self.props_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def show_props(self, feature: FeatureConfigItem):
        self.updating_props = True
        self.clear_props()

        # Name (read only for now)
        self.props_layout.addRow("Name", QLabel(feature.name))

        # Params (JSON editor for simplicity)
        self.params_edit = QTextEdit()
        self.params_edit.setText(json.dumps(feature.params, indent=2))
        self.params_edit.textChanged.connect(self.on_params_changed)
        self.props_layout.addRow("Params (JSON)", self.params_edit)

        self.updating_props = False

    def on_params_changed(self):
        if self.updating_props:
            return

        row = self.feature_list.currentRow()
        if row < 0:
            return

        try:
            params = json.loads(self.params_edit.toPlainText())
            self.current_config.features[row].params = params
        except json.JSONDecodeError:
            pass  # Invalid JSON, ignore
