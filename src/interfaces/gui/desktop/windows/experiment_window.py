"""
Модуль комплексных экспериментов (Full Pipeline).
"""

from typing import Optional

import numpy as np
import pandas as pd

try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QCheckBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QMessageBox,
        QProgressDialog,
        QPushButton,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    raise ImportError("Требуется установка: pip install PyQt6")

from src.interfaces.gui.desktop.utils import log_to_parent
from src.interfaces.gui.desktop.widgets import VirtualizedTableWidget


class ExperimentWorker(QThread):
    """
    Worker для выполнения комплексного эксперимента.
    """

    progress = pyqtSignal(int, str)  # progress, message
    finished = pyqtSignal(pd.DataFrame, str)  # results, message
    error = pyqtSignal(str)  # error message

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self) -> None:
        """Выполнить эксперимент."""
        try:
            datasets = self.config.get("datasets", [])
            features = self.config.get("features", [])
            labels = self.config.get("labels", [])
            models = self.config.get("models", [])
            strategies = self.config.get("strategies", [])

            # Подсчитать общее количество комбинаций
            total = len(datasets) * len(features) * len(labels) * len(models) * len(strategies)

            if total == 0:
                self.error.emit("Не выбрано ни одной комбинации")
                return

            self.progress.emit(0, f"Запуск {total} экспериментов...")
            self.msleep(300)

            results = []
            current = 0

            # Симуляция экспериментов
            for dataset in datasets:
                for feature_set in features:
                    for label_set in labels:
                        for model in models:
                            for strategy in strategies:
                                current += 1
                                progress_pct = int((current / total) * 100)

                                self.progress.emit(progress_pct, f"Эксперимент {current}/{total}: {dataset} + {model}")

                                # Симуляция выполнения
                                self.msleep(200)

                                # Случайные метрики
                                result = {
                                    "experiment_id": f"exp_{current:04d}",
                                    "dataset": dataset,
                                    "features": feature_set,
                                    "labels": label_set,
                                    "model": model,
                                    "strategy": strategy,
                                    "accuracy": np.random.uniform(0.5, 0.9),
                                    "f1_score": np.random.uniform(0.4, 0.85),
                                    "sharpe_ratio": np.random.uniform(0.5, 2.5),
                                    "total_pnl": np.random.uniform(-100, 500),
                                    "win_rate": np.random.uniform(0.4, 0.7),
                                    "status": "completed",
                                }

                                results.append(result)

            # Создать DataFrame с результатами
            results_df = pd.DataFrame(results)

            self.progress.emit(100, "Завершено!")
            self.finished.emit(results_df, f"Успешно выполнено {total} экспериментов")

        except Exception as e:
            self.error.emit(f"Ошибка выполнения экспериментов: {str(e)}")


class ExperimentWindow(QWidget):
    """
    Окно комплексных экспериментов.

    Функции:
    - Конфигуратор полных пайплайнов
    - Множественный выбор (datasets × features × models × strategies)
    - Batch processing
    - Сводная таблица результатов
    - Экспорт результатов
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Основной layout
        layout = QVBoxLayout(self)

        # Панель управления
        control_panel = QHBoxLayout()

        run_experiments_btn = QPushButton("▶️ Запустить эксперименты")
        run_experiments_btn.clicked.connect(self._run_experiments)
        run_experiments_btn.setStyleSheet("QPushButton { background-color: #0e639c; color: white; font-weight: bold; }")
        control_panel.addWidget(run_experiments_btn)

        control_panel.addStretch()

        self.use_all_combinations = QCheckBox("Все комбинации")
        self.use_all_combinations.setChecked(True)
        self.use_all_combinations.setToolTip("Запустить все возможные комбинации выбранных элементов")
        control_panel.addWidget(self.use_all_combinations)

        layout.addLayout(control_panel)

        # Основной сплиттер
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Левая панель - конфигуратор
        config_group = QGroupBox("Конфигурация экспериментов")
        config_layout = QVBoxLayout(config_group)

        # Датасеты
        datasets_label = QLabel("<b>Датасеты</b>")
        config_layout.addWidget(datasets_label)

        self.datasets_list = QListWidget()
        self.datasets_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.datasets_list.addItems(["dataset_1", "dataset_2", "dataset_3"])
        self.datasets_list.setMaximumHeight(80)
        config_layout.addWidget(self.datasets_list)

        # Признаки
        features_label = QLabel("<b>Конфигурации признаков</b>")
        config_layout.addWidget(features_label)

        self.features_list = QListWidget()
        self.features_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.features_list.addItems(["features_v1", "features_v2", "features_v3"])
        self.features_list.setMaximumHeight(80)
        config_layout.addWidget(self.features_list)

        # Разметка
        labels_label = QLabel("<b>Методы разметки</b>")
        config_layout.addWidget(labels_label)

        self.labels_list = QListWidget()
        self.labels_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.labels_list.addItems(["triple_barrier_2_1", "horizon_10", "regression"])
        self.labels_list.setMaximumHeight(80)
        config_layout.addWidget(self.labels_list)

        # Модели
        models_label = QLabel("<b>Модели</b>")
        config_layout.addWidget(models_label)

        self.models_list = QListWidget()
        self.models_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.models_list.addItems(["LightGBM", "XGBoost", "CatBoost", "LSTM", "Transformer"])
        self.models_list.setMaximumHeight(100)
        config_layout.addWidget(self.models_list)

        # Стратегии
        strategies_label = QLabel("<b>Стратегии</b>")
        config_layout.addWidget(strategies_label)

        self.strategies_list = QListWidget()
        self.strategies_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.strategies_list.addItems(["strategy_a", "strategy_b", "strategy_c"])
        self.strategies_list.setMaximumHeight(80)
        config_layout.addWidget(self.strategies_list)

        # Кнопки выбора
        select_buttons = QHBoxLayout()

        select_all_btn = QPushButton("Выбрать все")
        select_all_btn.clicked.connect(self._select_all)
        select_buttons.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Снять всё")
        deselect_all_btn.clicked.connect(self._deselect_all)
        select_buttons.addWidget(deselect_all_btn)

        config_layout.addLayout(select_buttons)

        config_layout.addStretch()

        main_splitter.addWidget(config_group)

        # Правая панель - результаты
        tabs = QTabWidget()

        # Вкладка сводной таблицы
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        results_layout.setContentsMargins(0, 0, 0, 0)

        # Панель управления результатами
        results_control = QHBoxLayout()

        self.info_label = QLabel("Экспериментов: 0")
        results_control.addWidget(self.info_label)

        results_control.addStretch()

        export_btn = QPushButton("📥 Экспорт в CSV")
        export_btn.clicked.connect(self._export_results)
        results_control.addWidget(export_btn)

        filter_positive_btn = QPushButton("✨ Только прибыльные")
        filter_positive_btn.clicked.connect(self._filter_positive)
        results_control.addWidget(filter_positive_btn)

        results_layout.addLayout(results_control)

        self.results_table = VirtualizedTableWidget()
        results_layout.addWidget(self.results_table)

        tabs.addTab(results_tab, "📊 Результаты")

        # Вкладка логов
        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)

        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setPlaceholderText("Логи выполнения экспериментов...")
        logs_layout.addWidget(self.logs_text)

        tabs.addTab(logs_tab, "📝 Логи")

        # Вкладка визуализаций
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        viz_placeholder = QTextEdit(
            "Визуализации\n\n"
            "Здесь будут:\n"
            "- Parallel coordinates plot\n"
            "- Scatter matrix (модели vs метрики)\n"
            "- Heatmap корреляций\n"
            "- Сравнение equity curves"
        )
        viz_placeholder.setReadOnly(True)
        viz_layout.addWidget(viz_placeholder)

        tabs.addTab(viz_tab, "📈 Визуализации")

        main_splitter.addWidget(tabs)

        # Пропорции
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)

        layout.addWidget(main_splitter)

        # Worker
        self.worker: Optional[ExperimentWorker] = None
        self.current_results: Optional[pd.DataFrame] = None

    def _run_experiments(self) -> None:
        """Запустить эксперименты."""
        # Собрать выбранные элементы
        datasets = [item.text() for item in self.datasets_list.selectedItems()]
        features = [item.text() for item in self.features_list.selectedItems()]
        labels = [item.text() for item in self.labels_list.selectedItems()]
        models = [item.text() for item in self.models_list.selectedItems()]
        strategies = [item.text() for item in self.strategies_list.selectedItems()]

        # Проверить что хоть что-то выбрано
        if not all([datasets, features, labels, models, strategies]):
            QMessageBox.warning(
                self,
                "Недостаточно данных",
                "Выберите хотя бы один элемент в каждой категории.",
            )
            return

        # Посчитать количество экспериментов
        total = len(datasets) * len(features) * len(labels) * len(models) * len(strategies)

        reply = QMessageBox.question(
            self,
            "Подтверждение",
            f"Будет запущено {total} экспериментов.\n\nПродолжить?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Собрать конфигурацию
        config = {
            "datasets": datasets,
            "features": features,
            "labels": labels,
            "models": models,
            "strategies": strategies,
        }

        # Очистить предыдущие результаты
        self.logs_text.clear()

        # Создать progress dialog
        progress = QProgressDialog("Выполнение экспериментов...", "Отмена", 0, 100, self)
        progress.setWindowTitle("Эксперименты")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        # Запустить worker
        self.worker = ExperimentWorker(config)

        def on_progress(value: int, message: str) -> None:
            progress.setValue(value)
            progress.setLabelText(message)
            self.logs_text.append(message)
            log_to_parent(self, f"[PROGRESS] {message}")

        def on_finished(results: pd.DataFrame, message: str) -> None:
            progress.close()
            self.current_results = results
            self._display_results(results)
            QMessageBox.information(self, "Эксперименты завершены", message)
            log_to_parent(self, f"[OK] {message}")

        def on_error(error: str) -> None:
            progress.close()
            QMessageBox.critical(self, "Ошибка экспериментов", error)
            log_to_parent(self, f"[ERROR] {error}")

        self.worker.progress.connect(on_progress)
        self.worker.finished.connect(on_finished)
        self.worker.error.connect(on_error)
        progress.canceled.connect(self.worker.terminate)

        self.worker.start()

    def _display_results(self, results: pd.DataFrame) -> None:
        """
        Отобразить результаты экспериментов.

        Args:
            results: DataFrame с результатами
        """
        self.results_table.set_data(results)
        self.info_label.setText(f"Экспериментов: {len(results)}")

    def _select_all(self) -> None:
        """Выбрать все элементы."""
        self.datasets_list.selectAll()
        self.features_list.selectAll()
        self.labels_list.selectAll()
        self.models_list.selectAll()
        self.strategies_list.selectAll()

    def _deselect_all(self) -> None:
        """Снять выбор со всех элементов."""
        self.datasets_list.clearSelection()
        self.features_list.clearSelection()
        self.labels_list.clearSelection()
        self.models_list.clearSelection()
        self.strategies_list.clearSelection()

    def _export_results(self) -> None:
        """Экспортировать результаты."""
        if self.current_results is None or self.current_results.empty:
            QMessageBox.warning(
                self,
                "Нет результатов",
                "Сначала запустите эксперименты.",
            )
            return

        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт результатов",
            "experiments_results.csv",
            "CSV files (*.csv);;Excel files (*.xlsx)",
        )

        if file_path:
            try:
                if file_path.endswith(".xlsx"):
                    self.current_results.to_excel(file_path, index=False)
                else:
                    self.current_results.to_csv(file_path, index=False)

                QMessageBox.information(
                    self,
                    "Экспорт завершён",
                    f"Результаты экспортированы в:\n{file_path}",
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Ошибка экспорта",
                    f"Не удалось экспортировать результаты:\n{str(e)}",
                )

    def _filter_positive(self) -> None:
        """Отфильтровать только прибыльные эксперименты."""
        if self.current_results is None or self.current_results.empty:
            QMessageBox.warning(
                self,
                "Нет результатов",
                "Сначала запустите эксперименты.",
            )
            return

        # Фильтровать по PnL > 0
        positive = self.current_results[self.current_results["total_pnl"] > 0].copy()

        if positive.empty:
            QMessageBox.information(
                self,
                "Фильтр",
                "Нет прибыльных экспериментов.",
            )
        else:
            self.results_table.set_data(positive)
            self.info_label.setText(f"Прибыльных: {len(positive)} из {len(self.current_results)}")
