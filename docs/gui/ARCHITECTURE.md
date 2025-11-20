# GUI Desktop Архитектура

Документ описывает реализацию **Этапа 15** плана (`plan/Этап_15_GUI_интерфейс.md`) с учетом требований из `technical_spec.md` (раздел 6. Признаки) и сопутствующих документов (`SUMMARY.md`, `README.md`, `QUICK_START.md`).

## Технологический стек

- **Qt6** через `PySide6`
- **pyqtgraph** для высокопроизводительных графиков (свечи, equity, индикаторы)
- **QTableView + кастомные модели** для виртуализированных таблиц
- **qdarkstyle/qdarktheme** для темизации
- **QThread** воркеры для обучения, бэктестов и оптимизаций
- Сервисы оборачивают существующие модули: `src/data`, `src/features`, `src/labeling`, `src/pipelines`, `src/backtesting`

## Структура каталогов

```
src/interfaces/gui/
├── desktop/
│   ├── main.py                  # GUIApplication
│   ├── windows/                 # Окна модулей (datasets, features, labeling, ...)
│   ├── widgets/                 # Общие виджеты (ChartWidget, Table, Log, ConfigEditor)
│   ├── models/                  # Qt модели (DatasetTableModel, IndicatorTreeModel, ...)
│   ├── dialogs/                 # Диалоги (progress, config, comparison)
│   ├── services/                # Связка бизнес-логики и UI
│   └── workers/                 # QThread воркеры (training, backtest, optimization)
└── scripts/run_gui.py           # Точка входа
```

## Основные окна

| Окно | Назначение | Ключевые компоненты |
| --- | --- | --- |
| `DatasetWindow` | Управление датасетами (список, импорт, превью) | `DatasetTableModel`, `ChartWidget`, `DatasetService` |
| `FeatureWindow` | Конфигуратор признаков + генерация | `IndicatorTreeModel`, `ConfigEditor`, `FeatureService` |
| `LabelingWindow` | Настройки разметки и визуализация | `ConfigEditor`, `LabelingService`, `ChartWidget` |
| `TrainingWindow` | Настройка экспериментов обучения | `ConfigEditor`, `MetricsWidget`, `TrainingWorker` |
| `BacktestWindow` | Конфигурация стратегий и просмотр результатов | `ChartWidget`, `VirtualizedTableWidget`, `BacktestWorker` |
| `ExperimentWindow` | Batch-эксперименты (datasets × features × models × strategies) | `ConfigEditor`, `ExperimentTableModel`, `ExperimentService` |

## Сервисы и интеграции

- `DatasetService` — `DatasetCatalog` + `ParquetStorage`, импорт/экспорт, превью 10k+ баров.
- `FeatureService` — обертка над `FeatureGenerator`, каталог индикаторов из ТЗ (SMA/EMA/.../Nadaraya-Watson).
- `LabelingService` — пайплайн разметки (`LabelingPipeline`, triple barrier / horizon / regression / custom).
- `TrainingService` — пайплайн обучения (`TrainingPipeline`), поддержка callbacks для realtime обновлений.
- `BacktestService` — пайплайн бэктестов (`BacktestPipeline`), загрузка equity/trades.
- `ExperimentService` — генерация комбинаций и последовательный запуск training/backtest для каждой связки.

Все сервисы используют общий `run_pipeline_with_callback`, дублирующий базовую логику `BasePipeline` и добавляющий события прогресса для UI.

## Воркеры

- `TrainingWorker` — запускает `TrainingService` в потоке, пробрасывает прогресс, метрики и историю в `TrainingMonitorWidget`.
- `BacktestWorker` — запускает `BacktestService`, обновляет статус, equity и таблицу сделок.
- `OptimizationWorker` — универсальный враппер для длительных задач (hyperopt/strategy grid search).

## Темы и запуск

- В `GUIApplication` поддерживаются темы `dark` и `light`, используется `qdarkstyle` (fallback на `qdarktheme`).
- Скрипт запуска: `python scripts/run_gui.py`.

## Связь с документами плана

- Соответствие функциональности разделам `Этап_15_GUI_интерфейс.md`:
  - 6 модулей (Datasets/Features/Labeling/Training/Backtesting/Experiments) реализованы отдельными окнами.
  - pyqtgraph графики, виртуализированные таблицы, QThread воркеры, пресеты YAML, dockable панели и горячие клавиши (через toolbar/меню).
- Использованы требования `technical_spec.md#6 Признаки` — список индикаторов и конфигуратор фич.
- В `SUMMARY.md` и `README.md` Stage 15 обозначен как часть фазы интерфейсов; GUI документ покрывает критерии готовности (графики, таблицы, асинхронность, эксперименты).
- `QUICK_START.md` теперь дополняется скриптом `scripts/run_gui.py` для быстрой проверки GUI.

## Дальнейшие шаги

1. Расширить интеграцию с пайплайнами (передача реальных data_path/feature_path).
2. Добавить сохранение UI-пресетов и историю запусков (модуль `artifacts/gui`).
3. Встроить MLflow UI (см. план Этапа 15, раздел “MLflow интеграция”).
