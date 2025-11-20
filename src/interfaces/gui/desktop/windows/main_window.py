"""
Главное окно приложения.
"""

from typing import Optional

try:
    from PyQt6.QtCore import QSize, Qt
    from PyQt6.QtGui import QAction, QKeySequence
    from PyQt6.QtWidgets import (
        QDockWidget,
        QMainWindow,
        QMessageBox,
        QStatusBar,
        QTabWidget,
        QTextEdit,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    raise ImportError("PyQt6 не установлен. Установите с помощью: pip install PyQt6 pyqtgraph qdarkstyle")


class MainWindow(QMainWindow):
    """
    Главное окно desktop приложения.

    Содержит:
    - Меню и панели инструментов
    - Вкладки для модулей (Data, Features, Labeling, Training, Backtesting, Experiments)
    - Панель логов
    - Строку состояния
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setWindowTitle("Trading Platform - Desktop GUI")
        self.setGeometry(100, 100, 1600, 900)

        # Создать центральный виджет с вкладками
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Создать меню
        self._create_menu_bar()

        # Создать панель инструментов
        self._create_toolbar()

        # Создать док-панели
        self._create_dock_widgets()

        # Создать строку состояния
        self._create_status_bar()

        # Создать вкладки модулей
        self._create_module_tabs()

        # Применить стиль
        self._apply_style()

    def _create_menu_bar(self) -> None:
        """Создать меню."""
        menubar = self.menuBar()

        # Файл
        file_menu = menubar.addMenu("&Файл")

        # Открыть конфигурацию
        open_config_action = QAction("&Открыть конфигурацию...", self)
        open_config_action.setShortcut(QKeySequence.StandardKey.Open)
        open_config_action.setStatusTip("Открыть конфигурационный файл")
        open_config_action.triggered.connect(self._open_config)
        file_menu.addAction(open_config_action)

        # Сохранить конфигурацию
        save_config_action = QAction("&Сохранить конфигурацию...", self)
        save_config_action.setShortcut(QKeySequence.StandardKey.Save)
        save_config_action.setStatusTip("Сохранить текущую конфигурацию")
        save_config_action.triggered.connect(self._save_config)
        file_menu.addAction(save_config_action)

        file_menu.addSeparator()

        # Выход
        exit_action = QAction("В&ыход", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip("Выйти из приложения")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Вид
        view_menu = menubar.addMenu("&Вид")

        # Темная тема
        dark_theme_action = QAction("Тёмная тема", self, checkable=True)
        dark_theme_action.setChecked(True)
        dark_theme_action.setStatusTip("Переключить тёмную/светлую тему")
        dark_theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(dark_theme_action)

        view_menu.addSeparator()

        # Панель логов
        logs_action = QAction("Панель &логов", self, checkable=True)
        logs_action.setChecked(True)
        logs_action.setStatusTip("Показать/скрыть панель логов")
        logs_action.triggered.connect(self._toggle_logs_panel)
        view_menu.addAction(logs_action)

        # Помощь
        help_menu = menubar.addMenu("&Помощь")

        # О программе
        about_action = QAction("&О программе", self)
        about_action.setStatusTip("Информация о приложении")
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        # Документация
        docs_action = QAction("&Документация", self)
        docs_action.setShortcut(QKeySequence("F1"))
        docs_action.setStatusTip("Открыть документацию")
        docs_action.triggered.connect(self._open_docs)
        help_menu.addAction(docs_action)

    def _create_toolbar(self) -> None:
        """Создать панель инструментов."""
        toolbar = QToolBar("Основная панель")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Кнопки быстрого доступа (пока без иконок)
        load_data_action = QAction("Загрузить данные", self)
        load_data_action.setStatusTip("Быстрая загрузка данных")
        load_data_action.triggered.connect(self._quick_load_data)
        toolbar.addAction(load_data_action)

        toolbar.addSeparator()

        train_model_action = QAction("Обучить модель", self)
        train_model_action.setStatusTip("Быстрое обучение модели")
        train_model_action.triggered.connect(self._quick_train_model)
        toolbar.addAction(train_model_action)

        toolbar.addSeparator()

        run_backtest_action = QAction("Запустить бэктест", self)
        run_backtest_action.setStatusTip("Быстрый бэктест")
        run_backtest_action.triggered.connect(self._quick_run_backtest)
        toolbar.addAction(run_backtest_action)

    def _create_dock_widgets(self) -> None:
        """Создать док-панели."""
        # Панель логов
        self.log_dock = QDockWidget("Логи", self)
        self.log_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumHeight(200)
        self.log_widget.setPlaceholderText("Здесь будут отображаться логи операций...")
        self.log_dock.setWidget(self.log_widget)

        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)

    def _create_status_bar(self) -> None:
        """Создать строку состояния."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов к работе")

    def _create_module_tabs(self) -> None:
        """Создать вкладки для модулей."""
        # Импортировать модули
        try:
            from src.interfaces.gui.desktop.windows.backtest_window import BacktestWindow
            from src.interfaces.gui.desktop.windows.dataset_window import DatasetWindow
            from src.interfaces.gui.desktop.windows.experiment_window import ExperimentWindow
            from src.interfaces.gui.desktop.windows.feature_window import FeatureWindow
            from src.interfaces.gui.desktop.windows.labeling_window import LabelingWindow
            from src.interfaces.gui.desktop.windows.training_window import TrainingWindow

            # 1. Данные
            self.data_window = DatasetWindow(self)
            self.tabs.addTab(self.data_window, "📊 Данные")

            # 2. Признаки
            self.feature_window = FeatureWindow(self)
            self.tabs.addTab(self.feature_window, "🔧 Признаки")

            # 3. Разметка
            self.labeling_window = LabelingWindow(self)
            self.tabs.addTab(self.labeling_window, "🎯 Разметка")

            # 4. Обучение
            self.training_window = TrainingWindow(self)
            self.tabs.addTab(self.training_window, "🤖 Обучение")

            # 5. Бэктестинг
            self.backtest_window = BacktestWindow(self)
            self.tabs.addTab(self.backtest_window, "📈 Бэктестинг")

            # 6. Эксперименты
            self.experiment_window = ExperimentWindow(self)
            self.tabs.addTab(self.experiment_window, "🔬 Эксперименты")

        except ImportError as e:
            # Если не удалось импортировать модули, показать заглушку
            error_widget = QWidget()
            error_layout = QVBoxLayout(error_widget)
            error_text = QTextEdit(
                "Ошибка загрузки модулей GUI:\n\n"
                f"{str(e)}\n\n"
                "Установите зависимости:\n"
                "pip install PyQt6 pyqtgraph qdarkstyle"
            )
            error_text.setReadOnly(True)
            error_layout.addWidget(error_text)
            self.tabs.addTab(error_widget, "❌ Ошибка")

    def _apply_style(self) -> None:
        """Применить стиль к приложению."""
        try:
            import qdarkstyle

            self.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt6"))
        except ImportError:
            # Если qdarkstyle не установлен, применить базовый тёмный стиль
            self.setStyleSheet(
                """
                QMainWindow {
                    background-color: #2b2b2b;
                }
                QTextEdit {
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                    border: 1px solid #3e3e3e;
                }
                QTabWidget::pane {
                    border: 1px solid #3e3e3e;
                }
                QTabBar::tab {
                    background-color: #2b2b2b;
                    color: #d4d4d4;
                    padding: 8px 16px;
                    border: 1px solid #3e3e3e;
                }
                QTabBar::tab:selected {
                    background-color: #1e1e1e;
                    border-bottom: 2px solid #0e639c;
                }
            """
            )

    # Обработчики действий меню

    def _open_config(self) -> None:
        """Открыть конфигурационный файл."""
        self.log_widget.append("📂 Открытие конфигурации...")
        # TODO: Реализовать

    def _save_config(self) -> None:
        """Сохранить конфигурацию."""
        self.log_widget.append("💾 Сохранение конфигурации...")
        # TODO: Реализовать

    def _toggle_theme(self, checked: bool) -> None:
        """Переключить тему."""
        if checked:
            self._apply_style()
            self.log_widget.append("🌙 Переключено на тёмную тему")
        else:
            self.setStyleSheet("")
            self.log_widget.append("☀️ Переключено на светлую тему")

    def _toggle_logs_panel(self, checked: bool) -> None:
        """Показать/скрыть панель логов."""
        self.log_dock.setVisible(checked)

    def _show_about(self) -> None:
        """Показать информацию о программе."""
        QMessageBox.about(
            self,
            "О программе",
            "<h2>Trading Platform</h2>"
            "<p>Версия 0.1.0</p>"
            "<p>Модульная платформа для разработки и тестирования торговых моделей.</p>"
            "<p>Desktop GUI на базе PyQt6</p>",
        )

    def _open_docs(self) -> None:
        """Открыть документацию."""
        self.log_widget.append("📖 Открытие документации...")
        # TODO: Открыть браузер с документацией

    # Быстрые действия панели инструментов

    def _quick_load_data(self) -> None:
        """Быстрая загрузка данных."""
        self.tabs.setCurrentIndex(0)  # Переключиться на вкладку данных
        self.log_widget.append("⚡ Быстрая загрузка данных...")

    def _quick_train_model(self) -> None:
        """Быстрое обучение модели."""
        self.tabs.setCurrentIndex(3)  # Переключиться на вкладку обучения
        self.log_widget.append("⚡ Быстрое обучение модели...")

    def _quick_run_backtest(self) -> None:
        """Быстрый бэктест."""
        self.tabs.setCurrentIndex(4)  # Переключиться на вкладку бэктестинга
        self.log_widget.append("⚡ Быстрый бэктест...")

    def log_message(self, message: str) -> None:
        """
        Добавить сообщение в лог.

        Args:
            message: Сообщение для логирования
        """
        self.log_widget.append(message)

    def update_status(self, message: str) -> None:
        """
        Обновить строку состояния.

        Args:
            message: Сообщение для статусной строки
        """
        self.status_bar.showMessage(message)
