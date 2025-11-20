from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QDockWidget, QMainWindow, QTabWidget

from ..services import GUIServiceBundle
from ..widgets.log_widget import LogWidget
from .backtest_window import BacktestWindow
from .dataset_window import DatasetWindow
from .experiment_window import ExperimentWindow
from .feature_window import FeatureWindow
from .labeling_window import LabelingWindow
from .training_window import TrainingWindow


class MainWindow(QMainWindow):
    """Главное окно desktop-приложения."""

    def __init__(self, *, services: GUIServiceBundle, theme: str) -> None:
        super().__init__()
        self.services = services
        self.theme = theme

        self.setWindowTitle("Trading Platform GUI")
        self.resize(1600, 1000)

        self._tab_widget = QTabWidget(self)
        self.setCentralWidget(self._tab_widget)

        self.log_widget = LogWidget(self)
        self._init_docks()
        self._init_tabs()
        self._init_menus()
        self._init_toolbar()
        self.statusBar().showMessage("Готово")

    def _init_tabs(self) -> None:
        self.dataset_window = DatasetWindow(self.services.dataset, parent=self)
        self.feature_window = FeatureWindow(
            dataset_service=self.services.dataset,
            feature_service=self.services.feature,
            parent=self,
        )
        self.labeling_window = LabelingWindow(
            dataset_service=self.services.dataset,
            labeling_service=self.services.labeling,
            parent=self,
        )
        self.training_window = TrainingWindow(
            dataset_service=self.services.dataset,
            training_service=self.services.training,
            feature_service=self.services.feature,
            labeling_service=self.services.labeling,
            log_widget=self.log_widget,
            parent=self,
        )
        self.backtest_window = BacktestWindow(
            dataset_service=self.services.dataset,
            backtest_service=self.services.backtest,
            parent=self,
        )
        self.experiment_window = ExperimentWindow(
            dataset_service=self.services.dataset,
            experiment_service=self.services.experiment,
            log_widget=self.log_widget,
            parent=self,
        )

        self._tab_widget.addTab(self.dataset_window, "Datasets")
        self._tab_widget.addTab(self.feature_window, "Features")
        self._tab_widget.addTab(self.labeling_window, "Labeling")
        self._tab_widget.addTab(self.training_window, "Training")
        self._tab_widget.addTab(self.backtest_window, "Backtesting")
        self._tab_widget.addTab(self.experiment_window, "Experiments")

    def _init_docks(self) -> None:
        log_dock = QDockWidget("Логи", self)
        log_dock.setWidget(self.log_widget)
        log_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)

    def _init_menus(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&Файл")
        view_menu = menubar.addMenu("&Вид")
        help_menu = menubar.addMenu("&Справка")

        exit_action = QAction("Выход", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        theme_action = QAction("Переключить тему", self)
        theme_action.setShortcut("Ctrl+T")
        theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(theme_action)

        about_action = QAction("О программе", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _init_toolbar(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)

        actions = [
            ("Datasets", lambda: self._tab_widget.setCurrentWidget(self.dataset_window)),
            ("Features", lambda: self._tab_widget.setCurrentWidget(self.feature_window)),
            ("Labeling", lambda: self._tab_widget.setCurrentWidget(self.labeling_window)),
            ("Training", lambda: self._tab_widget.setCurrentWidget(self.training_window)),
            ("Backtest", lambda: self._tab_widget.setCurrentWidget(self.backtest_window)),
            ("Experiments", lambda: self._tab_widget.setCurrentWidget(self.experiment_window)),
        ]

        for name, handler in actions:
            action = QAction(name, self)
            action.triggered.connect(handler)
            toolbar.addAction(action)

    def _toggle_theme(self) -> None:
        self.theme = "light" if self.theme == "dark" else "dark"
        self.statusBar().showMessage(f"Тема переключена: {self.theme}", 2000)

    def _show_about(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.information(
            self,
            "О программе",
            "GUI торговой платформы.\nЭтап 15: Desktop приложение на Qt6.",
        )
