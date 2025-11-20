"""
Модуль обучения моделей.
"""

from typing import Optional

import numpy as np

try:
    import pyqtgraph as pg
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QSpacerItem,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    raise ImportError("Требуется установка: pip install PyQt6 pyqtgraph")

from src.interfaces.gui.desktop.utils import log_to_parent


class TrainingWorker(QThread):
    """
    Worker для обучения модели в фоновом режиме.
    """

    epoch_finished = pyqtSignal(int, dict)  # epoch, metrics
    log_message = pyqtSignal(str)  # log message
    finished = pyqtSignal(str)  # message
    error = pyqtSignal(str)  # error message

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self) -> None:
        """Обучить модель."""
        try:
            epochs = self.config.get("epochs", 10)

            self.log_message.emit(f"Начинается обучение модели {self.config['model']}...")
            self.log_message.emit(f"Эпох: {epochs}")

            # Симуляция обучения
            for epoch in range(1, epochs + 1):
                self.msleep(500)  # Имитация работы

                # Симулировать метрики (улучшение с эпохами)
                train_loss = 1.0 - (epoch / epochs) * 0.7 + np.random.uniform(-0.05, 0.05)
                val_loss = 1.0 - (epoch / epochs) * 0.6 + np.random.uniform(-0.05, 0.05)
                train_acc = (epoch / epochs) * 0.85 + np.random.uniform(-0.05, 0.05)
                val_acc = (epoch / epochs) * 0.8 + np.random.uniform(-0.05, 0.05)

                metrics = {
                    "epoch": epoch,
                    "train_loss": max(0.1, train_loss),
                    "val_loss": max(0.1, val_loss),
                    "train_accuracy": min(1.0, max(0.0, train_acc)),
                    "val_accuracy": min(1.0, max(0.0, val_acc)),
                }

                self.epoch_finished.emit(epoch, metrics)
                self.log_message.emit(
                    f"Эпоха {epoch}/{epochs} - "
                    f"Loss: {metrics['train_loss']:.4f}/{metrics['val_loss']:.4f} - "
                    f"Acc: {metrics['train_accuracy']:.4f}/{metrics['val_accuracy']:.4f}"
                )

                # Проверка остановки
                if self.isInterruptionRequested():
                    self.log_message.emit("Обучение прервано пользователем")
                    return

            self.finished.emit("Обучение успешно завершено!")

        except Exception as e:
            self.error.emit(f"Ошибка обучения: {str(e)}")


class TrainingMonitorWidget(QWidget):
    """
    Виджет для real-time мониторинга обучения.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # График loss
        self.loss_plot = pg.PlotWidget(title="Loss")
        self.loss_plot.setLabel("left", "Loss")
        self.loss_plot.setLabel("bottom", "Epoch")
        self.loss_plot.addLegend()
        self.loss_plot.showGrid(x=True, y=True, alpha=0.3)

        self.train_loss_curve = self.loss_plot.plot(pen=pg.mkPen(color="r", width=2), name="Train")
        self.val_loss_curve = self.loss_plot.plot(pen=pg.mkPen(color="b", width=2), name="Val")

        layout.addWidget(self.loss_plot)

        # График accuracy
        self.acc_plot = pg.PlotWidget(title="Accuracy")
        self.acc_plot.setLabel("left", "Accuracy")
        self.acc_plot.setLabel("bottom", "Epoch")
        self.acc_plot.addLegend()
        self.acc_plot.showGrid(x=True, y=True, alpha=0.3)

        self.train_acc_curve = self.acc_plot.plot(pen=pg.mkPen(color="r", width=2), name="Train")
        self.val_acc_curve = self.acc_plot.plot(pen=pg.mkPen(color="b", width=2), name="Val")

        layout.addWidget(self.acc_plot)

        # Данные для графиков
        self.epochs_data: list[int] = []
        self.train_loss_data: list[float] = []
        self.val_loss_data: list[float] = []
        self.train_acc_data: list[float] = []
        self.val_acc_data: list[float] = []

    def update_metrics(self, epoch: int, metrics: dict) -> None:
        """
        Обновить графики метриками.

        Args:
            epoch: Номер эпохи
            metrics: Словарь с метриками
        """
        self.epochs_data.append(epoch)
        self.train_loss_data.append(metrics["train_loss"])
        self.val_loss_data.append(metrics["val_loss"])
        self.train_acc_data.append(metrics["train_accuracy"])
        self.val_acc_data.append(metrics["val_accuracy"])

        # Обновить кривые
        self.train_loss_curve.setData(self.epochs_data, self.train_loss_data)
        self.val_loss_curve.setData(self.epochs_data, self.val_loss_data)
        self.train_acc_curve.setData(self.epochs_data, self.train_acc_data)
        self.val_acc_curve.setData(self.epochs_data, self.val_acc_data)

    def clear(self) -> None:
        """Очистить графики."""
        self.epochs_data.clear()
        self.train_loss_data.clear()
        self.val_loss_data.clear()
        self.train_acc_data.clear()
        self.val_acc_data.clear()

        self.train_loss_curve.setData([], [])
        self.val_loss_curve.setData([], [])
        self.train_acc_curve.setData([], [])
        self.val_acc_curve.setData([], [])


class TrainingWindow(QWidget):
    """
    Окно обучения моделей.

    Функции:
    - Создание экспериментов обучения
    - Real-time мониторинг
    - Hyperparameter search
    - Сравнение моделей
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Основной layout
        layout = QVBoxLayout(self)

        # Панель управления
        control_panel = QHBoxLayout()

        start_training_btn = QPushButton("▶️ Начать обучение")
        start_training_btn.clicked.connect(self._start_training)
        start_training_btn.setStyleSheet("QPushButton { background-color: #0e639c; color: white; font-weight: bold; }")
        control_panel.addWidget(start_training_btn)

        self.stop_training_btn = QPushButton("⏹️ Остановить")
        self.stop_training_btn.clicked.connect(self._stop_training)
        self.stop_training_btn.setEnabled(False)
        control_panel.addWidget(self.stop_training_btn)

        control_panel.addStretch()

        hyperopt_btn = QPushButton("🔍 Hyperparameter Search")
        hyperopt_btn.clicked.connect(self._start_hyperopt)
        control_panel.addWidget(hyperopt_btn)

        layout.addLayout(control_panel)

        # Основной сплиттер
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Левая панель - конфигуратор
        config_group = QGroupBox("Конфигурация обучения")
        config_layout = QFormLayout(config_group)

        # Модель
        self.model_combo = QComboBox()
        self.model_combo.addItems(
            [
                "LightGBM",
                "XGBoost",
                "CatBoost",
                "Random Forest",
                "LSTM",
                "GRU",
                "Transformer",
            ]
        )
        config_layout.addRow("Модель:", self.model_combo)

        # Датасет
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["dataset_1", "dataset_2", "dataset_3"])
        config_layout.addRow("Датасет:", self.dataset_combo)

        # Признаки
        self.features_combo = QComboBox()
        self.features_combo.addItems(["features_v1", "features_v2", "features_v3"])
        config_layout.addRow("Признаки:", self.features_combo)

        # Разметка
        self.labels_combo = QComboBox()
        self.labels_combo.addItems(["triple_barrier", "horizon", "regression"])
        config_layout.addRow("Разметка:", self.labels_combo)

        config_layout.addRow(QLabel(""))  # Разделитель

        # Гиперпараметры
        config_layout.addRow(QLabel("<b>Гиперпараметры</b>"))

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        config_layout.addRow("Эпохи:", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        config_layout.addRow("Learning Rate:", self.lr_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 10000)
        self.batch_size_spin.setValue(64)
        config_layout.addRow("Batch Size:", self.batch_size_spin)

        config_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        main_splitter.addWidget(config_group)

        # Правая панель - мониторинг
        tabs = QTabWidget()

        # Вкладка мониторинга
        monitor_tab = QWidget()
        monitor_layout = QVBoxLayout(monitor_tab)
        monitor_layout.setContentsMargins(0, 0, 0, 0)

        self.monitor_widget = TrainingMonitorWidget()
        monitor_layout.addWidget(self.monitor_widget)

        # Лог обучения
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumHeight(150)
        self.log_widget.setPlaceholderText("Логи обучения будут отображаться здесь...")
        monitor_layout.addWidget(self.log_widget)

        tabs.addTab(monitor_tab, "📊 Мониторинг")

        # Вкладка сравнения моделей
        comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(comparison_tab)
        comparison_placeholder = QTextEdit(
            "Сравнение моделей\n\n"
            "Здесь будет таблица со всеми обученными моделями:\n"
            "- ID эксперимента\n"
            "- Модель\n"
            "- Accuracy, Precision, Recall, F1\n"
            "- ROC-AUC\n"
            "- Время обучения\n"
            "\nВозможность сортировки и фильтрации"
        )
        comparison_placeholder.setReadOnly(True)
        comparison_layout.addWidget(comparison_placeholder)
        tabs.addTab(comparison_tab, "📈 Сравнение")

        main_splitter.addWidget(tabs)

        # Пропорции
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)

        layout.addWidget(main_splitter)

        # Worker
        self.worker: Optional[TrainingWorker] = None

    def _start_training(self) -> None:
        """Начать обучение модели."""
        # Собрать конфигурацию
        config = {
            "model": self.model_combo.currentText(),
            "dataset": self.dataset_combo.currentText(),
            "features": self.features_combo.currentText(),
            "labels": self.labels_combo.currentText(),
            "epochs": self.epochs_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "batch_size": self.batch_size_spin.value(),
        }

        # Очистить предыдущие результаты
        self.monitor_widget.clear()
        self.log_widget.clear()

        # Запустить worker
        self.worker = TrainingWorker(config)

        self.worker.epoch_finished.connect(self._on_epoch_finished)
        self.worker.log_message.connect(self._on_log_message)
        self.worker.finished.connect(self._on_training_finished)
        self.worker.error.connect(self._on_training_error)

        self.worker.start()

        # Обновить UI
        self.stop_training_btn.setEnabled(True)

        log_to_parent(self, f"[START] Начато обучение модели {config['model']}")

    def _stop_training(self) -> None:
        """Остановить обучение."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait()

            self.stop_training_btn.setEnabled(False)

            log_to_parent(self, "[STOP] Обучение остановлено")

    def _on_epoch_finished(self, epoch: int, metrics: dict) -> None:
        """
        Обработчик завершения эпохи.

        Args:
            epoch: Номер эпохи
            metrics: Метрики эпохи
        """
        self.monitor_widget.update_metrics(epoch, metrics)

    def _on_log_message(self, message: str) -> None:
        """
        Обработчик лог-сообщения.

        Args:
            message: Сообщение
        """
        self.log_widget.append(message)

    def _on_training_finished(self, message: str) -> None:
        """
        Обработчик завершения обучения.

        Args:
            message: Сообщение
        """
        self.stop_training_btn.setEnabled(False)
        QMessageBox.information(self, "Обучение завершено", message)

        log_to_parent(self, f"[OK] {message}")

    def _on_training_error(self, error: str) -> None:
        """
        Обработчик ошибки обучения.

        Args:
            error: Сообщение об ошибке
        """
        self.stop_training_btn.setEnabled(False)
        QMessageBox.critical(self, "Ошибка обучения", error)

        log_to_parent(self, f"[ERROR] {error}")

    def _start_hyperopt(self) -> None:
        """Запустить hyperparameter search."""
        # TODO: Реализовать
        QMessageBox.information(
            self,
            "В разработке",
            "Hyperparameter search будет реализован в следующей версии.",
        )
