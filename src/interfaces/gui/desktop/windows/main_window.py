from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDockWidget,
    QLabel,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QToolBar,
)

from src.interfaces.gui.desktop.windows.backtest_window import BacktestWindow
from src.interfaces.gui.desktop.windows.dataset_window import DatasetWindow
from src.interfaces.gui.desktop.windows.experiment_window import ExperimentWindow
from src.interfaces.gui.desktop.windows.feature_window import FeatureWindow
from src.interfaces.gui.desktop.windows.labeling_window import LabelingWindow
from src.interfaces.gui.desktop.windows.training_window import TrainingWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Platform")
        self.resize(1400, 900)

        self.setup_ui()
        self.create_actions()
        self.create_menus()
        self.create_toolbar()

    def setup_ui(self):
        # Central Widget - Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Initialize Modules
        self.tabs.addTab(DatasetWindow(), "Datasets")
        self.tabs.addTab(FeatureWindow(), "Features")
        self.tabs.addTab(LabelingWindow(), "Labeling")
        self.tabs.addTab(TrainingWindow(), "Training")
        self.tabs.addTab(BacktestWindow(), "Backtesting")
        self.tabs.addTab(ExperimentWindow(), "Experiments")

        # Dock Widgets (Example)
        self.log_dock = QDockWidget("Logs", self)
        self.log_dock.setWidget(self.create_placeholder("Log Console"))
        self.log_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)

        # Status Bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def create_placeholder(self, text):
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 20px; color: #888;")
        return label

    def create_actions(self):
        self.exit_action = QAction("Exit", self)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)

        self.settings_action = QAction("Settings", self)
        self.settings_action.triggered.connect(lambda: print("Settings clicked"))

    def create_menus(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.exit_action)

        edit_menu = menu_bar.addMenu("&Edit")
        edit_menu.addAction(self.settings_action)

        view_menu = menu_bar.addMenu("&View")
        view_menu.addAction(self.log_dock.toggleViewAction())

    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        toolbar.addAction(self.settings_action)
