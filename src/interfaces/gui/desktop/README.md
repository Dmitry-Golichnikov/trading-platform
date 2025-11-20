# Desktop GUI

Нативное desktop-приложение для управления платформой, построенное на PyQt6.

## Возможности

### 📊 Данные (Data Management)
- Загрузка данных из файлов (CSV, Parquet)
- Загрузка из хранилища (ParquetStorage)
- Просмотр OHLCV графиков (свечи/линии)
- Таблица данных с виртуализацией
- Quality reports

### 🔧 Признаки (Feature Engineering)
- Древовидный список индикаторов (30+ встроенных)
- Drag-and-drop конфигуратор
- YAML редактор конфигурации
- Real-time генерация признаков
- Валидация конфигураций

### 🎯 Разметка (Labeling)
- Конфигуратор методов (Triple Barrier, Horizon, Regression)
- Визуализация меток на графиках
- Статистика распределения классов
- Балансировка классов

### 🤖 Обучение (Model Training)
- Создание экспериментов обучения
- Real-time мониторинг (loss/accuracy curves)
- Поддержка всех моделей (LightGBM, XGBoost, LSTM, Transformer, etc)
- Hyperparameter search (в разработке)
- Сравнение моделей

### 📈 Бэктестинг (Backtesting)
- Конфигуратор стратегий
- Equity curve с benchmark
- Таблица сделок
- Метрики (Sharpe, Sortino, Max DD, Win Rate, etc)
- Strategy optimization (в разработке)

### 🔬 Эксперименты (Full Pipeline)
- Множественный выбор (datasets × features × labels × models × strategies)
- Batch processing с прогресс-баром
- Сводная таблица всех результатов
- Фильтрация прибыльных экспериментов
- Экспорт в CSV/Excel

## Установка

### 1. Установить зависимости

```bash
# Базовые GUI зависимости
pip install PyQt6 pyqtgraph qdarkstyle qt-material

# Или из requirements
pip install -r requirements/gui.txt

# Или через pyproject.toml
pip install -e ".[gui]"
```

### 2. Запустить приложение

```bash
# Из корня проекта
python src/interfaces/gui/scripts/run_gui.py

# Или через entry point (если установлен пакет)
trading-gui

# Или напрямую
python -m src.interfaces.gui.desktop.main
```

## Архитектура

```
src/interfaces/gui/desktop/
├── main.py                     # Entry point
├── windows/
│   ├── main_window.py          # Главное окно
│   ├── dataset_window.py       # Модуль данных
│   ├── feature_window.py       # Модуль признаков
│   ├── labeling_window.py      # Модуль разметки
│   ├── training_window.py      # Модуль обучения
│   ├── backtest_window.py      # Модуль бэктестинга
│   └── experiment_window.py    # Модуль экспериментов
├── widgets/
│   ├── chart_widget.py         # Высокопроизводительные графики
│   └── table_widget.py         # Виртуализированные таблицы
├── models/                     # Qt Models (MVC)
├── dialogs/                    # Диалоги
├── services/                   # Бизнес-логика
├── workers/                    # QThread workers
└── resources/                  # Иконки, стили
```

## Ключевые компоненты

### ChartWidget
Высокопроизводительный виджет для графиков:
- OpenGL-акселерация (опционально)
- Поддержка 10,000+ свечей без лагов
- Zoom/Pan с помощью мыши
- Crosshair с координатами
- Overlay индикаторов
- Метки Long/Short сигналов

```python
from src.interfaces.gui.desktop.widgets import ChartWidget

chart = ChartWidget(use_opengl=True)
chart.plot_candlesticks(df)  # df с OHLCV данными
chart.add_indicator_overlay(sma, "SMA(20)", color="blue")
chart.mark_labels(labels)  # Long/Short метки
```

### VirtualizedTableWidget
Таблица с виртуализацией для больших датасетов:
- Отображаются только видимые строки
- Быстрая прокрутка через миллионы строк
- Сортировка по колонкам
- Экспорт в CSV

```python
from src.interfaces.gui.desktop.widgets import VirtualizedTableWidget

table = VirtualizedTableWidget()
table.set_data(df)  # Любой pandas DataFrame
```

### QThread Workers
Все долгие операции выполняются в фоновых потоках:
- DataLoadWorker - загрузка данных
- FeatureGenerationWorker - генерация признаков
- LabelingWorker - разметка
- TrainingWorker - обучение моделей
- BacktestWorker - бэктестинг
- ExperimentWorker - комплексные эксперименты

## Производительность

### Оптимизации
- **pyqtgraph** - использует OpenGL для рендеринга графиков
- **QTableView** - виртуализация таблиц (отображаются только видимые строки)
- **QThread** - все операции >100ms выполняются асинхронно
- **Downsampling** - автоматическое прореживание данных для графиков (если >100K баров)

### Тестирование производительности
- ✅ 10,000 свечей - отрисовка < 100ms
- ✅ 100,000 строк в таблице - прокрутка без лагов
- ✅ Real-time обновление графиков обучения (каждая эпоха)
- ✅ Batch experiments - параллельная обработка

## Темизация

Приложение поддерживает светлую и тёмную темы:

```python
# Переключение через меню: Вид → Тёмная тема

# Или программно (в main.py)
import qdarkstyle
app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt6"))
```

## Горячие клавиши

- `Ctrl+O` - Открыть конфигурацию
- `Ctrl+S` - Сохранить конфигурацию
- `Ctrl+Q` - Выход
- `F1` - Документация
- `F5` - Обновить (в соответствующих модулях)

## Интеграция с core-модулями

GUI интегрируется со всеми модулями платформы:

```python
# Данные
from src.data.loaders.local_file import LocalFileLoader
from src.data.storage.parquet_storage import ParquetStorage

# Признаки
from src.features.generator import FeatureGenerator

# Разметка
from src.labeling.methods import TripleBarrierLabeler

# Обучение
from src.modeling.trainer import ModelTrainer

# Бэктест
from src.backtesting.engine import BacktestEngine

# Эксперименты
from src.orchestration.experiment_manager import ExperimentManager
```

## Расширение

### Добавление нового модуля

1. Создать новое окно в `windows/`:

```python
# windows/my_module_window.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout

class MyModuleWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        # Ваш код
```

2. Добавить вкладку в `main_window.py`:

```python
from .my_module_window import MyModuleWindow

# В _create_module_tabs()
self.my_module_window = MyModuleWindow(self)
self.tabs.addTab(self.my_module_window, "🔥 Мой модуль")
```

### Добавление нового виджета

Создать виджет в `widgets/`:

```python
# widgets/my_widget.py
from PyQt6.QtWidgets import QWidget

class MyWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Ваш код
```

## Packaging

Создание standalone .exe (Windows):

```bash
pip install pyinstaller

pyinstaller --onefile --windowed \
    --name="TradingPlatform" \
    --icon="resources/icon.ico" \
    src/interfaces/gui/desktop/main.py
```

Для кроссплатформенной сборки используйте `cx_Freeze`.

## Troubleshooting

### Ошибка: "PyQt6 не установлен"
```bash
pip install PyQt6 pyqtgraph
```

### Графики отображаются медленно
```python
# Включить OpenGL-акселерацию
chart = ChartWidget(use_opengl=True)
```

### Приложение зависает при долгих операциях
Убедитесь что долгие операции выполняются в QThread workers, а не в главном потоке.

## TODO

- [ ] MLflow UI интеграция (QWebEngineView)
- [ ] Hyperparameter search UI
- [ ] Strategy optimization UI
- [ ] Parallel coordinates plot для экспериментов
- [ ] SHAP plots интеграция
- [ ] Настройка dockable panels
- [ ] Context menus (правый клик)
- [ ] Autocomplete в конфиг-редакторах

## Лицензия

См. основной LICENSE файл проекта.
