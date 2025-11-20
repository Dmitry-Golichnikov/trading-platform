"""
Модуль управления данными.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QComboBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QListWidget,
        QMessageBox,
        QPushButton,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    raise ImportError("Требуется установка: pip install PyQt6")

from src.interfaces.gui.desktop.utils import log_to_parent
from src.interfaces.gui.desktop.widgets import ChartWidget, VirtualizedTableWidget


class DataLoadWorker(QThread):
    """
    Worker для загрузки данных в фоновом режиме.
    """

    finished = pyqtSignal(pd.DataFrame, str)  # data, message
    error = pyqtSignal(str)  # error message

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def run(self) -> None:
        """Загрузить данные."""
        try:
            file_path = Path(self.file_path)

            if file_path.suffix == ".csv":
                df = pd.read_csv(file_path)
            elif file_path.suffix == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                self.error.emit(f"Неподдерживаемый формат файла: {file_path.suffix}")
                return

            # Валидация базовых колонок
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                self.error.emit(f"Отсутствуют обязательные колонки: {missing_cols}")
                return

            # Преобразовать timestamp
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            message = f"Загружено {len(df)} баров из {file_path.name}"
            self.finished.emit(df, message)

        except Exception as e:
            self.error.emit(f"Ошибка загрузки данных: {str(e)}")


class DatasetWindow(QWidget):
    """
    Окно управления данными.

    Функции:
    - Список датасетов
    - Импорт данных из файлов
    - Просмотр OHLCV графиков
    - Просмотр таблицы данных
    - Quality reports
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Основной layout
        layout = QVBoxLayout(self)

        # Панель управления
        control_panel = QHBoxLayout()

        load_file_btn = QPushButton("📂 Загрузить из файла")
        load_file_btn.clicked.connect(self._load_from_file)
        control_panel.addWidget(load_file_btn)

        load_storage_btn = QPushButton("💾 Загрузить из хранилища")
        load_storage_btn.clicked.connect(self._load_from_storage)
        control_panel.addWidget(load_storage_btn)

        control_panel.addStretch()

        quality_report_btn = QPushButton("📊 Quality Report")
        quality_report_btn.clicked.connect(self._show_quality_report)
        control_panel.addWidget(quality_report_btn)

        layout.addLayout(control_panel)

        # Основной сплиттер (список датасетов + просмотр)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Левая панель - список датасетов
        datasets_group = QGroupBox("Датасеты")
        datasets_layout = QVBoxLayout(datasets_group)

        self.datasets_list = QListWidget()
        self.datasets_list.itemClicked.connect(self._on_dataset_selected)
        datasets_layout.addWidget(self.datasets_list)

        # Кнопки управления датасетами
        dataset_buttons = QHBoxLayout()

        refresh_btn = QPushButton("🔄")
        refresh_btn.setMaximumWidth(40)
        refresh_btn.setToolTip("Обновить список")
        refresh_btn.clicked.connect(self._refresh_datasets)
        dataset_buttons.addWidget(refresh_btn)

        delete_btn = QPushButton("🗑️")
        delete_btn.setMaximumWidth(40)
        delete_btn.setToolTip("Удалить датасет")
        delete_btn.clicked.connect(self._delete_dataset)
        dataset_buttons.addWidget(delete_btn)

        dataset_buttons.addStretch()

        datasets_layout.addLayout(dataset_buttons)

        main_splitter.addWidget(datasets_group)

        # Правая панель - просмотр данных
        view_group = QGroupBox("Просмотр данных")
        view_layout = QVBoxLayout(view_group)

        # Селектор режима просмотра
        view_selector = QHBoxLayout()
        view_selector.addWidget(QLabel("Режим:"))

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["График", "Таблица"])
        self.view_mode_combo.currentIndexChanged.connect(self._switch_view_mode)
        view_selector.addWidget(self.view_mode_combo)

        view_selector.addStretch()
        view_layout.addLayout(view_selector)

        # Виджеты просмотра
        self.chart_widget = ChartWidget()
        self.table_widget = VirtualizedTableWidget()

        view_layout.addWidget(self.chart_widget)
        view_layout.addWidget(self.table_widget)

        # По умолчанию показываем график
        self.table_widget.hide()

        main_splitter.addWidget(view_group)

        # Пропорции: список 20%, просмотр 80%
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 4)

        layout.addWidget(main_splitter)

        # Данные
        self.current_data: Optional[pd.DataFrame] = None
        self.datasets: dict[str, pd.DataFrame] = {}
        self.load_worker: Optional[DataLoadWorker] = None

        # Загрузить список датасетов
        self._refresh_datasets()

    def _load_from_file(self) -> None:
        """Загрузить данные из файла."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл с данными",
            "",
            "Data files (*.csv *.parquet);;All files (*.*)",
        )

        if file_path:
            # Запустить загрузку в отдельном потоке
            self.load_worker = DataLoadWorker(file_path)
            self.load_worker.finished.connect(self._on_data_loaded)
            self.load_worker.error.connect(self._on_load_error)
            self.load_worker.start()

            # Показать сообщение о загрузке
            log_to_parent(self, f"[INFO] Загрузка данных из {Path(file_path).name}...")

    def _on_data_loaded(self, data: pd.DataFrame, message: str) -> None:
        """
        Обработчик успешной загрузки данных.

        Args:
            data: Загруженные данные
            message: Сообщение о загрузке
        """
        # Запросить имя датасета
        dataset_name, ok = QInputDialog.getText(
            self,
            "Имя датасета",
            "Введите имя для датасета:",
            text=f"dataset_{len(self.datasets) + 1}",
        )

        if ok and dataset_name:
            # Сохранить датасет
            self.datasets[dataset_name] = data
            self.datasets_list.addItem(dataset_name)

            # Отобразить данные
            self.current_data = data
            self._display_data()

            # Логирование
            log_to_parent(self, f"[OK] {message}")

    def _on_load_error(self, error: str) -> None:
        """
        Обработчик ошибки загрузки.

        Args:
            error: Сообщение об ошибке
        """
        QMessageBox.critical(self, "Ошибка загрузки", error)
        log_to_parent(self, f"[ERROR] {error}")

    def _load_from_storage(self) -> None:
        """Загрузить данные из хранилища."""
        # TODO: Реализовать загрузку из ParquetStorage
        QMessageBox.information(
            self,
            "В разработке",
            "Загрузка из хранилища будет реализована в следующей версии.",
        )

    def _on_dataset_selected(self) -> None:
        """Обработчик выбора датасета из списка."""
        selected_items = self.datasets_list.selectedItems()
        if selected_items:
            dataset_name = selected_items[0].text()
            if dataset_name in self.datasets:
                self.current_data = self.datasets[dataset_name]
                self._display_data()

    def _display_data(self) -> None:
        """Отобразить текущие данные."""
        if self.current_data is None:
            return

        mode = self.view_mode_combo.currentText()

        if mode == "График":
            self.chart_widget.plot_candlesticks(self.current_data)
        elif mode == "Таблица":
            self.table_widget.set_data(self.current_data)

    def _switch_view_mode(self, index: int) -> None:
        """
        Переключить режим просмотра.

        Args:
            index: Индекс выбранного режима
        """
        if index == 0:  # График
            self.chart_widget.show()
            self.table_widget.hide()
        elif index == 1:  # Таблица
            self.chart_widget.hide()
            self.table_widget.show()

        # Обновить отображение
        self._display_data()

    def _refresh_datasets(self) -> None:
        """Обновить список датасетов."""
        # TODO: Загрузить список из ParquetStorage/Catalog
        self.datasets_list.clear()

        # Пока просто показываем текущие датасеты в памяти
        for name in self.datasets.keys():
            self.datasets_list.addItem(name)

    def _delete_dataset(self) -> None:
        """Удалить выбранный датасет."""
        selected_items = self.datasets_list.selectedItems()
        if selected_items:
            dataset_name = selected_items[0].text()

            reply = QMessageBox.question(
                self,
                "Подтверждение",
                f"Удалить датасет '{dataset_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                if dataset_name in self.datasets:
                    del self.datasets[dataset_name]

                self.datasets_list.takeItem(self.datasets_list.currentRow())

                log_to_parent(self, f"[INFO] Датасет '{dataset_name}' удалён")

    def _show_quality_report(self) -> None:
        """Показать quality report для текущего датасета."""
        if self.current_data is None:
            QMessageBox.warning(
                self,
                "Нет данных",
                "Сначала загрузите датасет.",
            )
            return

        # TODO: Интегрировать с src.data.quality.reports
        QMessageBox.information(
            self,
            "В разработке",
            "Quality reports будут реализованы в следующей версии.",
        )
