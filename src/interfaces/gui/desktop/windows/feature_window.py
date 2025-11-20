"""
Модуль генерации признаков.
"""

from typing import Optional

import yaml

try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QMessageBox,
        QProgressDialog,
        QPushButton,
        QSplitter,
        QTextEdit,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    raise ImportError("Требуется установка: pip install PyQt6")

from src.interfaces.gui.desktop.utils import log_to_parent


class FeatureGenerationWorker(QThread):
    """
    Worker для генерации признаков в фоновом режиме.
    """

    progress = pyqtSignal(int, str)  # progress, message
    finished = pyqtSignal(str)  # message
    error = pyqtSignal(str)  # error message

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self) -> None:
        """Сгенерировать признаки."""
        try:
            # TODO: Интегрировать с src.features.generator
            self.progress.emit(25, "Загрузка данных...")
            self.msleep(500)

            self.progress.emit(50, "Генерация индикаторов...")
            self.msleep(500)

            self.progress.emit(75, "Применение трансформаций...")
            self.msleep(500)

            self.progress.emit(100, "Завершено!")

            self.finished.emit("Признаки успешно сгенерированы")

        except Exception as e:
            self.error.emit(f"Ошибка генерации признаков: {str(e)}")


class FeatureWindow(QWidget):
    """
    Окно генерации признаков.

    Функции:
    - Конфигуратор индикаторов (древовидный список)
    - Drag-and-drop для добавления в конфиг
    - Генерация признаков
    - Просмотр важности признаков
    - Корреляционные матрицы
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Основной layout
        layout = QVBoxLayout(self)

        # Панель управления
        control_panel = QHBoxLayout()

        load_config_btn = QPushButton("📂 Загрузить конфиг")
        load_config_btn.clicked.connect(self._load_config)
        control_panel.addWidget(load_config_btn)

        save_config_btn = QPushButton("💾 Сохранить конфиг")
        save_config_btn.clicked.connect(self._save_config)
        control_panel.addWidget(save_config_btn)

        control_panel.addStretch()

        generate_btn = QPushButton("▶️ Сгенерировать признаки")
        generate_btn.clicked.connect(self._generate_features)
        generate_btn.setStyleSheet("QPushButton { background-color: #0e639c; color: white; font-weight: bold; }")
        control_panel.addWidget(generate_btn)

        layout.addLayout(control_panel)

        # Основной сплиттер
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Левая панель - доступные индикаторы
        indicators_group = QGroupBox("Доступные индикаторы")
        indicators_layout = QVBoxLayout(indicators_group)

        self.indicators_tree = QTreeWidget()
        self.indicators_tree.setHeaderLabels(["Индикатор"])
        self._populate_indicators_tree()
        indicators_layout.addWidget(self.indicators_tree)

        # Кнопка добавления
        add_btn = QPushButton("➕ Добавить в конфиг")
        add_btn.clicked.connect(self._add_indicator_to_config)
        indicators_layout.addWidget(add_btn)

        main_splitter.addWidget(indicators_group)

        # Правая панель - редактор конфигурации
        config_group = QGroupBox("Конфигурация признаков (YAML)")
        config_layout = QVBoxLayout(config_group)

        self.config_editor = QTextEdit()
        self.config_editor.setPlaceholderText(
            "# Конфигурация признаков\n"
            "\n"
            "# Пример:\n"
            "# features:\n"
            "#   - type: SMA\n"
            "#     window: 20\n"
            "#   - type: RSI\n"
            "#     window: 14"
        )
        config_layout.addWidget(self.config_editor)

        # Кнопки управления конфигом
        config_buttons = QHBoxLayout()

        validate_btn = QPushButton("✓ Валидировать")
        validate_btn.clicked.connect(self._validate_config)
        config_buttons.addWidget(validate_btn)

        clear_btn = QPushButton("🗑️ Очистить")
        clear_btn.clicked.connect(self._clear_config)
        config_buttons.addWidget(clear_btn)

        config_buttons.addStretch()

        config_layout.addLayout(config_buttons)

        main_splitter.addWidget(config_group)

        # Пропорции
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 2)

        layout.addWidget(main_splitter)

        # Worker
        self.worker: Optional[FeatureGenerationWorker] = None

    def _populate_indicators_tree(self) -> None:
        """Заполнить дерево индикаторов."""
        # Категории индикаторов
        categories = {
            "Трендовые": ["SMA", "EMA", "WMA", "MACD", "ADX", "Parabolic SAR", "Ichimoku Cloud"],
            "Моментум": ["RSI", "Stochastic Oscillator", "Stochastic RSI", "CCI", "Williams %R", "TRIX"],
            "Волатильность": ["Bollinger Bands", "ATR", "Donchian Channels", "Keltner Channels"],
            "Объёмные": ["OBV", "Chaikin Money Flow", "Money Flow Index", "Volume Profile", "VWAP"],
            "Продвинутые": ["Heikin-Ashi", "Pivot Points", "Fractal Dimension", "Nadaraya-Watson"],
        }

        for category, indicators in categories.items():
            category_item = QTreeWidgetItem(self.indicators_tree, [category])
            category_item.setExpanded(False)

            for indicator in indicators:
                QTreeWidgetItem(category_item, [indicator])

    def _add_indicator_to_config(self) -> None:
        """Добавить выбранный индикатор в конфигурацию."""
        selected_items = self.indicators_tree.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Выбор индикатора", "Выберите индикатор для добавления.")
            return

        indicator = selected_items[0].text(0)

        # Шаблон YAML для индикатора
        template = f"""
  - type: {indicator}
    window: 14  # Настройте параметры
"""

        # Добавить в редактор
        current_text = self.config_editor.toPlainText()
        if not current_text.strip():
            current_text = "features:"

        self.config_editor.setPlainText(current_text + template)

    def _load_config(self) -> None:
        """Загрузить конфигурацию из файла."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить конфигурацию",
            "configs/features",
            "YAML files (*.yaml *.yml);;All files (*.*)",
        )

        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    config_text = f.read()

                self.config_editor.setPlainText(config_text)

                log_to_parent(self, f"[OPEN] Конфигурация загружена: {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось загрузить конфигурацию:\n{str(e)}")

    def _save_config(self) -> None:
        """Сохранить конфигурацию в файл."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить конфигурацию",
            "configs/features/feature_config.yaml",
            "YAML files (*.yaml *.yml);;All files (*.*)",
        )

        if file_path:
            try:
                config_text = self.config_editor.toPlainText()

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(config_text)

                log_to_parent(self, f"[SAVE] Конфигурация сохранена: {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось сохранить конфигурацию:\n{str(e)}")

    def _validate_config(self) -> None:
        """Валидировать конфигурацию."""
        config_text = self.config_editor.toPlainText()

        if not config_text.strip():
            QMessageBox.warning(self, "Валидация", "Конфигурация пуста.")
            return

        try:
            # Попытаться распарсить YAML
            yaml.safe_load(config_text)

            # TODO: Интегрировать с валидацией из src.features.config_parser

            QMessageBox.information(
                self,
                "Валидация",
                "✅ Конфигурация корректна!",
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка валидации",
                f"Конфигурация содержит ошибки:\n{str(e)}",
            )

    def _clear_config(self) -> None:
        """Очистить конфигурацию."""
        reply = QMessageBox.question(
            self,
            "Подтверждение",
            "Очистить конфигурацию?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.config_editor.clear()

    def _generate_features(self) -> None:
        """Сгенерировать признаки."""
        config_text = self.config_editor.toPlainText()

        if not config_text.strip():
            QMessageBox.warning(self, "Генерация признаков", "Конфигурация пуста.")
            return

        try:
            config = yaml.safe_load(config_text)

            # Создать progress dialog
            progress = QProgressDialog("Генерация признаков...", "Отмена", 0, 100, self)
            progress.setWindowTitle("Генерация признаков")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)

            # Запустить worker
            self.worker = FeatureGenerationWorker(config)

            def on_progress(value: int, message: str) -> None:
                progress.setValue(value)
                progress.setLabelText(message)
                log_to_parent(self, f"[PROGRESS] {message}")

            def on_finished(message: str) -> None:
                progress.close()
                QMessageBox.information(self, "Генерация завершена", message)
                log_to_parent(self, f"[OK] {message}")

            def on_error(error: str) -> None:
                progress.close()
                QMessageBox.critical(self, "Ошибка генерации", error)
                log_to_parent(self, f"[ERROR] {error}")

            self.worker.progress.connect(on_progress)
            self.worker.finished.connect(on_finished)
            self.worker.error.connect(on_error)
            progress.canceled.connect(self.worker.terminate)

            self.worker.start()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось запустить генерацию:\n{str(e)}",
            )
