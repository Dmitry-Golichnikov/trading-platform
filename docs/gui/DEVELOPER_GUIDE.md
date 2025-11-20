# Developer Guide - GUI Extension

Руководство по расширению и модификации GUI приложения.

## Архитектура

### Структура проекта

```
src/interfaces/gui/desktop/
├── main.py                 # Entry point (создаёт QApplication)
├── windows/                # Модули-окна
│   ├── main_window.py      # Главное окно (координатор)
│   ├── *_window.py         # Модули функциональности
├── widgets/                # Переиспользуемые виджеты
│   ├── chart_widget.py
│   └── table_widget.py
├── models/                 # Qt Models (для QTableView)
├── dialogs/                # Диалоговые окна
├── services/               # Бизнес-логика
├── workers/                # QThread workers
└── resources/              # Статические ресурсы
```

### Паттерны проектирования

1. **Model-View (MVC)**: Для таблиц используется `QAbstractTableModel`
2. **Worker Pattern**: Долгие операции в QThread workers
3. **Signals/Slots**: Для межкомпонентной коммуникации
4. **Service Layer**: Бизнес-логика отделена от UI

## Добавление нового модуля

### Шаг 1: Создать окно модуля

```python
# src/interfaces/gui/desktop/windows/my_module_window.py

from typing import Optional
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
)

class MyModuleWindow(QWidget):
    """
    Описание вашего модуля.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        # Добавить UI элементы
        button = QPushButton("Действие")
        button.clicked.connect(self._on_action)
        layout.addWidget(button)

    def _on_action(self) -> None:
        """Обработчик действия."""
        # Логировать в главное окно
        if self.parentWidget() is not None:
            self.parentWidget().log_message("✨ Действие выполнено")  # type: ignore
```

### Шаг 2: Добавить в главное окно

```python
# src/interfaces/gui/desktop/windows/main_window.py

def _create_module_tabs(self) -> None:
    # ... существующий код ...

    from src.interfaces.gui.desktop.windows.my_module_window import MyModuleWindow

    self.my_module_window = MyModuleWindow(self)
    self.tabs.addTab(self.my_module_window, "✨ Мой модуль")
```

### Шаг 3: Обновить exports

```python
# src/interfaces/gui/desktop/windows/__init__.py

from .my_module_window import MyModuleWindow

__all__ = [
    # ... существующие ...
    "MyModuleWindow",
]
```

## Создание асинхронных операций

### Шаг 1: Создать Worker

```python
# src/interfaces/gui/desktop/workers/my_worker.py

from PyQt6.QtCore import QThread, pyqtSignal

class MyWorker(QThread):
    """Worker для долгой операции."""

    # Сигналы
    progress = pyqtSignal(int, str)  # progress, message
    finished = pyqtSignal(object, str)  # result, message
    error = pyqtSignal(str)  # error message

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self) -> None:
        """Выполнить операцию."""
        try:
            # Имитация работы
            for i in range(100):
                self.progress.emit(i, f"Шаг {i}/100")
                self.msleep(50)  # Имитация

                # Проверка остановки
                if self.isInterruptionRequested():
                    return

            result = {"status": "success"}
            self.finished.emit(result, "Операция завершена")

        except Exception as e:
            self.error.emit(f"Ошибка: {str(e)}")
```

### Шаг 2: Использовать Worker в UI

```python
from PyQt6.QtWidgets import QProgressDialog
from src.interfaces.gui.desktop.workers.my_worker import MyWorker

def _start_operation(self) -> None:
    """Запустить операцию."""
    config = {"param": "value"}

    # Progress dialog
    progress = QProgressDialog("Выполнение...", "Отмена", 0, 100, self)
    progress.setWindowModality(Qt.WindowModality.WindowModal)

    # Worker
    self.worker = MyWorker(config)

    # Подключить сигналы
    self.worker.progress.connect(
        lambda value, msg: (progress.setValue(value), progress.setLabelText(msg))
    )
    self.worker.finished.connect(self._on_finished)
    self.worker.error.connect(self._on_error)
    progress.canceled.connect(self.worker.requestInterruption)

    # Запустить
    self.worker.start()

def _on_finished(self, result: dict, message: str) -> None:
    """Обработчик успешного завершения."""
    QMessageBox.information(self, "Завершено", message)
```

## Добавление нового виджета

### Шаг 1: Создать виджет

```python
# src/interfaces/gui/desktop/widgets/my_widget.py

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class MyWidget(QWidget):
    """Описание виджета."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        self.label = QLabel("Мой виджет")
        layout.addWidget(self.label)

    def set_data(self, data: str) -> None:
        """Установить данные."""
        self.label.setText(data)
```

### Шаг 2: Использовать виджет

```python
from src.interfaces.gui.desktop.widgets.my_widget import MyWidget

# В вашем окне
self.my_widget = MyWidget()
layout.addWidget(self.my_widget)

# Установить данные
self.my_widget.set_data("Новые данные")
```

## Интеграция с core-модулями

### Пример: Интеграция DataLoader

```python
# В DatasetWindow

from src.data.loaders.local_file import LocalFileLoader
from src.data.storage.parquet_storage import ParquetStorage

class DatasetWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Инициализировать core-модули
        self.loader = LocalFileLoader()
        self.storage = ParquetStorage()

    def _load_from_storage(self) -> None:
        """Загрузить из ParquetStorage."""
        try:
            df = self.storage.load_dataset(
                ticker="SBER",
                timeframe="1h",
            )
            self.current_data = df
            self._display_data()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
```

### Пример: Интеграция FeatureGenerator

```python
# В FeatureWindow

from src.features.generator import FeatureGenerator
from src.features.config_parser import parse_feature_config

def _generate_features(self) -> None:
    """Реальная генерация признаков."""
    config_text = self.config_editor.toPlainText()

    try:
        # Распарсить конфиг
        config = parse_feature_config(config_text)

        # Создать генератор
        generator = FeatureGenerator(config)

        # Сгенерировать (в worker!)
        features = generator.generate(self.current_data)

        # Отобразить
        self._display_features(features)

    except Exception as e:
        QMessageBox.critical(self, "Ошибка", str(e))
```

## Работа с графиками

### Добавление графика в окно

```python
from src.interfaces.gui.desktop.widgets import ChartWidget

# Создать
self.chart = ChartWidget(use_opengl=True)
layout.addWidget(self.chart)

# Отобразить свечи
self.chart.plot_candlesticks(df)

# Добавить индикатор
self.chart.add_indicator_overlay(
    indicator=df["sma_20"],
    name="SMA(20)",
    color="blue",
    width=2,
)

# Добавить метки
self.chart.mark_labels(labels_series)
```

### Создание custom графика

```python
import pyqtgraph as pg

# Создать PlotWidget
self.custom_plot = pg.PlotWidget(title="Custom Plot")
self.custom_plot.setLabel("left", "Y Axis")
self.custom_plot.setLabel("bottom", "X Axis")

# Добавить данные
x = [1, 2, 3, 4, 5]
y = [10, 20, 30, 40, 50]
self.custom_plot.plot(x, y, pen=pg.mkPen(color="r", width=2))
```

## Работа с таблицами

### Использование VirtualizedTableWidget

```python
from src.interfaces.gui.desktop.widgets import VirtualizedTableWidget

# Создать
self.table = VirtualizedTableWidget()
layout.addWidget(self.table)

# Установить данные
self.table.set_data(df)

# Получить данные
current_df = self.table.get_data()
```

### Создание custom модели таблицы

```python
from PyQt6.QtCore import QAbstractTableModel, Qt

class CustomTableModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._data.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            else:
                return str(section)
        return None
```

## Стилизация

### Применение темы

```python
import qdarkstyle

# В main.py или main_window.py
app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt6"))
```

### Custom стили

```python
# Inline стили
button.setStyleSheet("""
    QPushButton {
        background-color: #0e639c;
        color: white;
        font-weight: bold;
        padding: 5px 15px;
    }
    QPushButton:hover {
        background-color: #0d5a8a;
    }
""")

# Из файла
with open("resources/styles/custom.css") as f:
    widget.setStyleSheet(f.read())
```

## Горячие клавиши

```python
from PyQt6.QtGui import QAction, QKeySequence

# В MainWindow или любом окне
action = QAction("Действие", self)
action.setShortcut(QKeySequence("Ctrl+A"))
action.triggered.connect(self._on_action)

# Добавить в меню
menu.addAction(action)
```

## Логирование

### В главное окно

```python
# Из любого дочернего виджета
if self.parentWidget() is not None:
    self.parentWidget().log_message("📝 Сообщение")  # type: ignore
```

### Обновление статус-бара

```python
if self.parentWidget() is not None:
    self.parentWidget().update_status("Статус изменён")  # type: ignore
```

## Тестирование

### Unit тесты для виджетов

```python
import pytest
from PyQt6.QtWidgets import QApplication
from src.interfaces.gui.desktop.widgets.chart_widget import ChartWidget

@pytest.fixture
def app():
    """Создать QApplication."""
    return QApplication([])

def test_chart_widget(app):
    """Тест ChartWidget."""
    widget = ChartWidget()
    assert widget is not None

    # Тест отрисовки
    import pandas as pd
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=100),
        "open": [100] * 100,
        "high": [110] * 100,
        "low": [90] * 100,
        "close": [105] * 100,
        "volume": [1000] * 100,
    })

    widget.plot_candlesticks(df)
    assert widget.candlestick_item is not None
```

## Best Practices

### 1. Асинхронность
✅ DO: Используйте QThread для операций >100ms
❌ DON'T: Блокируйте главный поток

### 2. Сигналы
✅ DO: Используйте signals для межкомпонентной коммуникации
❌ DON'T: Обращайтесь напрямую к родительским виджетам

### 3. Модульность
✅ DO: Создавайте независимые, переиспользуемые виджеты
❌ DON'T: Дублируйте код между модулями

### 4. Error Handling
✅ DO: Обрабатывайте все исключения и показывайте QMessageBox
❌ DON'T: Позволяйте исключениям падать UI

### 5. Производительность
✅ DO: Используйте виртуализацию для больших данных
❌ DON'T: Загружайте все данные в UI сразу

## Debugging

### QtCreator Designer
Для визуального дизайна UI можно использовать Qt Designer:

```bash
# Установить
pip install pyqt6-tools

# Запустить
pyqt6-tools designer
```

### Логирование PyQt

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# В коде
logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

### Профилирование

```python
import cProfile

def profile_function():
    # Ваш код
    pass

cProfile.run('profile_function()')
```

## Ресурсы

- [PyQt6 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
- [pyqtgraph Documentation](https://pyqtgraph.readthedocs.io/)
- [Qt for Python](https://doc.qt.io/qtforpython/)
- [Real Python PyQt Tutorial](https://realpython.com/python-pyqt-gui-calculator/)

## Контакты

При возникновении вопросов создавайте issues в репозитории проекта.
