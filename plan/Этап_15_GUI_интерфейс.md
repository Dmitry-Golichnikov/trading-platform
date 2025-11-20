# Этап 15: GUI интерфейс (Desktop)

## Цель
Нативное desktop-приложение для управления платформой, визуализации и мониторинга с поддержкой больших объемов данных (графики с десятками тысяч баров/свечей, большие таблицы).

## Зависимости
Этапы 00-14

## Технологии

**Framework:**
- **PyQt6** или **PySide6** (Qt6) — мощный GUI framework с нативным рендерингом и отличной производительностью
  - Альтернативы: CustomTkinter (более простой), Dear PyGui (gaming-grade UI)

**Визуализация графиков:**
- **pyqtgraph** — высокопроизводительные графики для PyQt/PySide, оптимизированы для больших данных (OpenGL-акселерация)
  - Альтернативы: matplotlib (встраивание в Qt), mplfinance + FigureCanvas
  - Поддержка свечных графиков, индикаторов, zoom/pan, crosshair

**Таблицы:**
- **QTableView** + **QAbstractTableModel** — виртуализация для миллионов строк
- Lazy loading и pagination для больших датасетов

**Дополнительно:**
- **qdarkstyle** / **qt-material** — dark/light темы
- **pyqtgraph.dockarea** — настраиваемые панели
- Threading (QThread) для фоновых задач

## Структура

```
src/interfaces/gui/
├── desktop/
│   ├── main.py                    # Entry point (QApplication)
│   ├── windows/
│   │   ├── main_window.py         # Главное окно (QMainWindow)
│   │   ├── dataset_window.py
│   │   ├── feature_window.py
│   │   ├── labeling_window.py
│   │   ├── training_window.py
│   │   ├── backtest_window.py
│   │   └── experiment_window.py
│   ├── widgets/
│   │   ├── chart_widget.py        # pyqtgraph chart для свечей/баров
│   │   ├── table_widget.py        # Виртуализированная таблица
│   │   ├── metrics_widget.py
│   │   ├── log_widget.py
│   │   └── config_editor.py       # YAML/JSON editor
│   ├── models/                     # Qt Models (MVC)
│   │   ├── dataset_model.py
│   │   ├── feature_model.py
│   │   └── experiment_model.py
│   ├── dialogs/
│   │   ├── progress_dialog.py
│   │   ├── config_dialog.py
│   │   └── comparison_dialog.py
│   ├── services/                   # Business logic
│   │   ├── dataset_service.py
│   │   ├── feature_service.py
│   │   ├── labeling_service.py
│   │   ├── training_service.py
│   │   └── backtest_service.py
│   ├── workers/                    # QThread workers
│   │   ├── training_worker.py
│   │   ├── backtest_worker.py
│   │   └── optimization_worker.py
│   └── resources/
│       ├── icons/
│       ├── styles/
│       └── ui/                     # Qt Designer .ui files (optional)
└── scripts/
    └── run_gui.py                  # Launcher script
```

## Основные модули (полный пайплайн)

### 1. Наборы данных (Data Management)
- **Список датасетов** (ticker, timeframe, bars, date range)
- **Импорт данных** (загрузка CSV/Parquet, MOEX/Tinkoff API)
- **Просмотр данных**:
  - Свечной/линейный график (pyqtgraph) с zoom/pan
  - Таблица OHLCV (виртуализированная)
  - Quality report (пропуски, выбросы)
- **Экспорт/удаление датасетов**

### 2. Признаки (Feature Engineering)
- **Конфигуратор признаков**:
  - Список доступных индикаторов (древовидный QTreeView)
  - Drag-and-drop для добавления в конфиг
  - Параметры индикаторов (spin boxes, sliders)
- **Генерация признаков** (QProgressDialog с real-time логами)
- **Просмотр результатов**:
  - Таблица признаков (виртуализированная)
  - Корреляционная матрица (heatmap)
  - Feature importance (bar chart)
- **Сохранение/загрузка конфигов** (YAML)

### 3. Разметка данных (Labeling)
- **Конфигуратор методов разметки**:
  - Выбор метода (triple_barrier, horizon, regression)
  - Параметры (барьеры, временной горизонт)
  - Применение к датасетам (batch processing)
- **Просмотр результатов**:
  - График с метками (Long/Short/Hold) наложенными на свечи
  - Таблица меток (label, barrier_hit, holding_period, realized_return)
  - Статистика распределения классов
- **Фильтры и балансировка** (Majority Vote, SMOTE, undersampling)

### 4. Обучение моделей (Model Training & Optimization)
- **Создание эксперимента обучения**:
  - Выбор датасета + признаков + разметки
  - Выбор модели (LightGBM, XGBoost, CatBoost, LSTM, Transformer)
  - Конфигурация гиперпараметров
  - Train/Val/Test split
- **Запуск обучения** (в отдельном QThread):
  - Real-time логи (QTextEdit с auto-scroll)
  - Графики loss/metrics (обновляются по эпохам)
  - Progress bar
- **Hyperparameter Search**:
  - Optuna/Hyperopt интеграция
  - Таблица trials (№, params, score)
  - График оптимизации (best value vs trial)
- **Сравнение моделей**:
  - Таблица (model_id, accuracy, precision, recall, F1, ROC-AUC)
  - Сортировка/фильтрация
  - Calibration plots, Confusion matrix

### 5. Тестирование стратегий (Backtesting & Optimization)
- **Конфигуратор бэктеста**:
  - Выбор модели + датасета
  - Параметры стратегии (threshold, position_size, stop_loss, take_profit)
  - Комиссии, slippage
- **Запуск бэктеста** (в QThread):
  - Real-time equity curve
  - Логи сделок
- **Результаты**:
  - Equity curve (с drawdown overlay)
  - Таблица сделок (entry, exit, PnL, duration)
  - Метрики (Sharpe, Sortino, max DD, Win Rate)
- **Strategy Optimization**:
  - Grid search / Random search по параметрам стратегии
  - Таблица результатов (params → metrics)
  - Heatmap для 2D параметров
- **Сравнение стратегий**:
  - Наложение equity curves
  - Таблица метрик side-by-side

### 6. Эксперименты (Full Pipeline Experiments)
- **Конфигуратор комплексного эксперимента**:
  - Множественный выбор:
    - Датасетов (ABIO_1h, SBER_1d, ...)
    - Конфигов признаков (v1, v2, v3)
    - Методов разметки (tb_2_1, horizon_10, ...)
    - Моделей (lgbm_default, xgb_tuned, lstm_v2)
    - Стратегий (strategy_a, strategy_b)
  - Комбинаторика: все × все или custom grid
- **Запуск** (параллельная обработка):
  - Progress bar (общий + детали)
  - Таблица выполнения (experiment_id, status, time)
  - Логи по каждому эксперименту
- **Результаты**:
  - Сводная таблица (dataset, features, model, strategy → final metrics)
  - Фильтрация/сортировка по любому столбцу
  - Экспорт в CSV/Excel
  - Визуализация: parallel coordinates plot, scatter matrix
- **MLflow интеграция** (опционально):
  - Логирование всех экспериментов
  - Просмотр в встроенном MLflow UI (QWebEngineView)

## Ключевые компоненты

### ChartWidget (pyqtgraph)
```python
class ChartWidget(pg.PlotWidget):
    """Высокопроизводительный график для свечей/линий"""

    def plot_candlesticks(self, df: pd.DataFrame):
        # OpenGL-акселерированный рендеринг
        # Zoom/Pan с mouse/keyboard
        # Crosshair с координатами
        ...

    def add_indicator_overlay(self, indicator: pd.Series, name: str):
        # RSI, MACD, Bollinger Bands overlay
        ...

    def mark_labels(self, labels: pd.Series):
        # Long (↑), Short (↓), Hold markers
        ...
```

### VirtualizedTableWidget
```python
class VirtualizedTableWidget(QTableView):
    """Таблица с lazy loading для больших датасетов"""

    def __init__(self, model: QAbstractTableModel):
        # Виртуализация строк (отображаем только видимые)
        # Быстрая прокрутка через миллионы строк
        # Сортировка/фильтрация
        ...
```

### TrainingMonitorWidget
```python
class TrainingMonitorWidget(QWidget):
    """Real-time мониторинг обучения"""

    def __init__(self):
        self.loss_chart = ChartWidget()      # Loss curve
        self.metrics_chart = ChartWidget()   # Accuracy, F1, etc
        self.log_view = QTextEdit()          # Логи

    def update_epoch(self, epoch: int, metrics: dict):
        # Обновление графиков и логов
        ...
```

## Ключевые фичи

- **Нативная производительность** (C++ Qt backend)
- **OpenGL-акселерированные графики** (pyqtgraph)
- **Виртуализация таблиц** (QTableView)
- **Многопоточность** (QThread для долгих операций)
- **Preset management** (сохранение/загрузка конфигов YAML)
- **Dark/Light theme** (qdarkstyle)
- **Dockable panels** (перетаскиваемые панели)
- **Горячие клавиши** (QKeySequence)

## Визуализации

### 1. Price Charts (pyqtgraph)
- **Свечи/линии** (переключение режимов)
- **Индикаторы** (RSI, MACD, Bollinger overlay)
- **Метки разметки** (Long/Short markers)
- **Zoom/Pan** (mouse wheel, drag)
- **Crosshair** (координаты + OHLC tooltip)
- **Time axis** (умное форматирование дат)

### 2. Equity Curves
- **Множественные кривые** (сравнение стратегий)
- **Drawdown overlay** (заливка просадок)
- **Benchmark comparison** (vs Buy & Hold)

### 3. Feature Importance
- **Bar charts** (top-N признаков)
- **SHAP plots** (встраивание matplotlib canvas)
- **Correlation heatmap**

### 4. Training Progress
- **Loss curves** (train/val)
- **Metrics curves** (accuracy, F1, ROC-AUC)
- **Real-time updates** (QThread → signal → UI update)

### 5. Optimization Results
- **Parallel coordinates plot** (hyperopt trials)
- **Scatter matrix** (param1 vs param2 vs score)
- **Contour plots** (для 2D grid search)

## Критерии готовности

### Базовая функциональность
- [x] Главное окно с меню и панелями запускается
- [x] Все 6 основных модулей реализованы (Datasets, Features, Labeling, Training, Backtesting, Experiments)
- [x] Интеграция со всеми core-модулями платформы (заглушки готовы, требуется подключение)

### Графики и таблицы
- [x] pyqtgraph chart отображает свечи/бары (10000+ баров без лагов)
- [x] Zoom/Pan/Crosshair работают
- [x] Виртуализированные таблицы (QTableView) для больших датасетов
- [x] Индикаторы накладываются на график

### Асинхронность и многопоточность
- [x] Долгие операции (обучение, бэктест) не блокируют UI (QThread)
- [x] Real-time обновление логов и графиков
- [x] Progress dialogs с возможностью отмены

### Эксперименты и оптимизация
- [x] Batch experiments (множественные комбинации)
- [ ] Hyperparameter search интегрирован (TODO)
- [ ] Strategy optimization интегрирован (TODO)
- [x] Сравнение результатов (таблицы, графики)

### UX и удобство
- [x] Dark/Light theme переключение
- [x] Preset management (save/load configs)
- [x] Горячие клавиши для основных действий
- [x] Tooltips и status bar подсказки

## Статус реализации

### ✅ Выполнено
- Базовая структура приложения
- MainWindow с меню, панелями и вкладками
- Все 6 модулей-окон (Data, Features, Labeling, Training, Backtesting, Experiments)
- ChartWidget с поддержкой свечей, индикаторов и меток
- VirtualizedTableWidget для больших таблиц
- QThread workers для всех асинхронных операций
- Темизация (qdarkstyle)
- Launcher script и entry point
- Документация (README.md)
- Зависимости добавлены в pyproject.toml

### 🔄 В разработке
- Полная интеграция с core-модулями (сейчас используются заглушки)
- Hyperparameter search UI
- Strategy optimization UI
- MLflow UI интеграция
- Визуализации (parallel coords, SHAP plots)

### 📋 Следующие шаги
1. Подключить реальные модули вместо заглушек
2. Добавить тесты для GUI компонентов
3. Реализовать hyperopt и strategy optimization UI
4. Добавить продвинутые визуализации
5. Создать packaging для .exe/.app

## Промпты

```
Фаза 1 - Базовая структура:
Создай главное окно PyQt6 приложения в src/interfaces/gui/desktop/:
1. MainWindow с QMenuBar, QToolBar, QDockWidget панелями
2. Центральный виджет с QTabWidget для модулей
3. Настройка тем (qdarkstyle)
4. Базовая навигация и статус бар

Фаза 2 - Модуль данных:
Реализуй DatasetWindow и связанные виджеты:
1. Список датасетов (QTableView + model)
2. ChartWidget (pyqtgraph) для свечей
3. Импорт данных (QFileDialog, workers)
4. Интеграция с src/data/

Фаза 3 - Модуль признаков:
Реализуй FeatureWindow:
1. Конфигуратор (QTreeView для индикаторов)
2. Drag-and-drop для добавления в конфиг
3. QThread worker для генерации
4. Просмотр результатов (таблица + корреляция)

Фаза 4 - Модуль разметки:
Реализуй LabelingWindow:
1. Конфигуратор методов разметки
2. Просмотр графика с метками (ChartWidget + markers)
3. Таблица результатов разметки
4. Интеграция с src/labeling/

Фаза 5 - Модуль обучения:
Реализуй TrainingWindow:
1. Конфигуратор эксперимента
2. TrainingMonitorWidget (real-time графики + логи)
3. QThread worker для обучения
4. Hyperparameter search UI (таблица trials)
5. Сравнение моделей

Фаза 6 - Модуль бэктестинга:
Реализуй BacktestWindow:
1. Конфигуратор бэктеста
2. Equity curve (pyqtgraph)
3. Таблица сделок
4. Strategy optimization UI
5. Сравнение стратегий

Фаза 7 - Модуль экспериментов:
Реализуй ExperimentWindow:
1. Конфигуратор комплексных экспериментов (множественный выбор)
2. Batch processing (QThreadPool)
3. Сводная таблица результатов
4. Экспорт результатов
5. Визуализации (parallel coords, scatter)

Фаза 8 - Полировка:
1. Preset management (save/load YAML/JSON)
2. Горячие клавиши (Ctrl+N, Ctrl+S, F5, etc)
3. Tooltips и help dialogs
4. Error handling и user-friendly сообщения
5. Packaging (PyInstaller для .exe)
```

## Важные замечания

**Производительность:**
- **pyqtgraph** использует OpenGL для рендеринга (включить `useOpenGL=True`)
- **QTableView** с виртуализацией: отображаются только видимые строки
- **QThread** для всех долгих операций (>100ms)
- **Signals/Slots** для межпоточной коммуникации
- Downsampling данных для графиков (если >100K баров)

**Архитектура:**
- **Model-View** паттерн (QAbstractTableModel для таблиц)
- **Service layer** для бизнес-логики (не в UI коде)
- **Workers** (QRunnable) для параллельных задач
- **Signals** для уведомлений между компонентами

**UX Best Practices:**
- **QProgressDialog** для долгих операций с кнопкой отмены
- **QMessageBox** для подтверждений и ошибок
- **Status bar** для подсказок о текущем действии
- **Context menus** (правый клик) для быстрых действий
- **Keyboard shortcuts** (QAction + QKeySequence)

**Packaging:**
```bash
# PyInstaller для создания standalone .exe
pyinstaller --onefile --windowed --name="TradingPlatform" src/interfaces/gui/desktop/main.py

# Или cx_Freeze для кроссплатформенности
```

## Следующий этап
[Этап 16: Интеграция Tinkoff API](Этап_16_Интеграция_Tinkoff_API.md)
