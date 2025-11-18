# Backend реализация модуля Labeling

## Обзор

Реализован полноценный backend для модуля разметки таргетов с поддержкой асинхронного выполнения задач, различных методов разметки и управления жизненным циклом задач.

## Архитектура

### Компоненты

```
src/interfaces/gui/backend/
├── api/
│   ├── models.py                        # Добавлены модели Labeling
│   └── routers/
│       └── labeling.py                  # ✨ НОВЫЙ: API endpoints
└── services/
    ├── labeling_service.py              # ✨ НОВЫЙ: Бизнес-логика
    └── labeling_task_service.py         # ✨ НОВЫЙ: Менеджер задач
```

## API Endpoints

### Управление наборами разметки

#### `GET /api/labeling`
Получить список всех наборов разметки
- Query параметры:
  - `dataset_id` (optional): фильтр по датасету
  - `method` (optional): фильтр по методу разметки

#### `GET /api/labeling/{labeling_set_id}`
Получить информацию о конкретном наборе разметки

#### `GET /api/labeling/{labeling_set_id}/data`
Получить данные разметки
- Query параметры:
  - `limit` (default: 1000): максимальное количество строк

#### `DELETE /api/labeling/{labeling_set_id}`
Удалить набор разметки

#### `GET /api/labeling/{labeling_set_id}/visualize`
Получить данные для визуализации
- Query параметры:
  - `chart_type` (default: "distribution"): тип графика

### Управление задачами разметки

#### `GET /api/labeling/tasks`
Получить список задач разметки
- Query параметры:
  - `status` (optional): фильтр по статусу
  - `dataset_id` (optional): фильтр по датасету

#### `POST /api/labeling/tasks`
Создать задачи разметки
- Body: `LabelingTaskCreateRequest`

```json
{
  "name": "Long стратегия 2%/1%",
  "dataset_ids": ["SBER__1h", "GAZP__1h"],
  "feature_set_id": null,
  "apply_to_all": false,
  "method": "triple_barrier",
  "config": {
    "upper_barrier": {"type": "percentage", "value": 0.02},
    "lower_barrier": {"type": "percentage", "value": 0.01},
    "time_barrier": 20,
    "direction": "long",
    "commission_rate": 0.0005
  },
  "description": "Conservative long strategy",
  "auto_start": true
}
```

#### `GET /api/labeling/tasks/{task_id}`
Получить информацию о задаче

#### `POST /api/labeling/tasks/{task_id}/pause`
Приостановить задачу

#### `POST /api/labeling/tasks/{task_id}/resume`
Возобновить задачу

#### `POST /api/labeling/tasks/{task_id}/cancel`
Отменить задачу

#### `POST /api/labeling/tasks/{task_id}/restart`
Перезапустить задачу

## Модели данных

### LabelingSetInfo
```python
{
    "id": str,                           # ID набора разметки
    "name": str,                         # Название
    "dataset_id": str,                   # ID датасета
    "method": LabelingMethod,            # Метод разметки
    "config": dict,                      # Конфигурация
    "num_samples": int,                  # Количество сэмплов
    "class_distribution": dict,          # Распределение классов
    "created_at": datetime,              # Дата создания
    "updated_at": datetime,              # Дата обновления
    "status": str,                       # Статус
    "description": str,                  # Описание
    "feature_set_id": str                # ID набора признаков
}
```

### LabelingTaskInfo
```python
{
    "id": str,                           # ID задачи
    "name": str,                         # Название
    "dataset_id": str,                   # ID датасета
    "labeling_set_id": str,              # ID набора разметки
    "status": TaskStatus,                # Статус задачи
    "progress": float,                   # Прогресс (0.0 - 1.0)
    "processed_rows": int,               # Обработано строк
    "total_rows": int,                   # Всего строк
    "config": dict,                      # Конфигурация
    "message": str,                      # Сообщение
    "created_at": datetime,              # Дата создания
    "updated_at": datetime,              # Дата обновления
    "started_at": datetime,              # Дата начала
    "finished_at": datetime,             # Дата завершения
    "method": str,                       # Метод разметки
    "class_distribution": dict,          # Распределение классов
    "feature_set_id": str                # ID набора признаков
}
```

## Сервисы

### LabelingService

Базовый сервис для работы с наборами разметки:

**Основные методы:**
- `list_labeling_sets()` - получить список наборов
- `get_labeling_set()` - получить набор по ID
- `get_labeling_data()` - получить данные разметки
- `delete_labeling_set()` - удалить набор
- `save_labeling_result()` - сохранить результат разметки
- `update_metadata_status()` - обновить статус

**Хранение данных:**
```
artifacts/labeling/
├── {dataset_id}__{labeling_name}/
│   ├── metadata.json              # Метаданные
│   └── labels.parquet             # Данные разметки
└── tasks/
    └── {task_id}.json             # Состояние задач
```

### LabelingTaskManager

Менеджер асинхронных задач разметки:

**Возможности:**
- Выполнение задач в фоновом режиме
- Pause/Resume/Cancel для управления задачами
- Прогресс в реальном времени через WebSocket
- Персистентное хранение состояния
- Автоматическое восстановление после рестарта

**Поддерживаемые методы разметки:**
1. **Horizon** - фиксированный и адаптивный горизонт
2. **Triple Barrier** - тройной барьер (TP/SL/Time)
3. **Regression** - регрессионные таргеты

**Интеграция с модулем `src/labeling/`:**
- Использует `HorizonLabeler`, `TripleBarrierLabeler`, `RegressionTargetsLabeler`
- Применяет постфильтры из `src/labeling/filters/`
- Поддерживает все параметры конфигурации

## Конфигурация методов

### Horizon
```python
config = {
    "horizon": 20,                       # Горизонт в барах
    "adaptive": False,                   # Адаптивный горизонт
    "threshold_pct": 0.01,               # Порог для классификации
    "direction": "long+short"            # Направление торговли
}
```

### Triple Barrier
```python
config = {
    "upper_barrier": {
        "type": "percentage",            # или "atr", "volatility"
        "value": 0.02                    # 2%
    },
    "lower_barrier": {
        "type": "percentage",
        "value": 0.01                    # 1%
    },
    "time_barrier": 20,                  # Временной барьер
    "direction": "long",                 # Направление
    "min_return": 0.0,                   # Минимальный return
    "commission_rate": 0.0005            # Комиссия (0.05%)
}
```

### Regression
```python
config = {
    "target": "future_return",           # или "mfe", "mae", "sharpe"
    "horizon": 20                        # Горизонт
}
```

### Постфильтры
```python
config = {
    "filters": [
        {
            "type": "smoothing",
            "params": {
                "window": 3,
                "method": "median"       # или "mean", "exponential"
            }
        },
        {
            "type": "sequence",
            "params": {
                "min_length": 2          # Минимальная длина последовательности
            }
        },
        {
            "type": "majority_vote",
            "params": {
                "window": 5              # Окно голосования
            }
        },
        {
            "type": "danger_zones",
            "params": {
                "high_volatility_threshold": 3.0  # Порог волатильности
            }
        }
    ]
}
```

## Жизненный цикл задачи

1. **Queued** - задача создана и ожидает выполнения
2. **Running** - задача выполняется
   - Загрузка датасета
   - Создание labeler
   - Выполнение разметки
   - Применение фильтров
   - Сохранение результата
3. **Paused** - задача приостановлена пользователем
4. **Completed** - задача успешно завершена
5. **Failed** - произошла ошибка
6. **Cancelled** - задача отменена пользователем

## WebSocket обновления

Задачи отправляют обновления в реальном времени через WebSocket:

```json
{
    "type": "labeling_task",
    "data": {
        "id": "task-123",
        "status": "running",
        "progress": 0.45,
        "message": "Labeling data...",
        "processed_rows": 4500,
        "total_rows": 10000
    }
}
```

## Обработка ошибок

**Типичные ошибки:**
- `404` - Набор разметки или задача не найдены
- `400` - Некорректные параметры запроса
- `500` - Внутренняя ошибка сервера

**Обработка в задачах:**
- Ошибки логируются в `message` поле задачи
- Статус меняется на `failed`
- Сохраняется traceback для отладки

## Тестирование

### Запуск backend
```bash
cd src/interfaces/gui/backend
python -m uvicorn main:app --reload --port 8000
```

### Проверка API
```bash
# Health check
curl http://localhost:8000/health

# API docs
open http://localhost:8000/api/docs

# Список наборов разметки
curl http://localhost:8000/api/labeling

# Создание задачи
curl -X POST http://localhost:8000/api/labeling/tasks \
  -H "Content-Type: application/json" \
  -d @labeling_task.json
```

## Интеграция с frontend

Frontend использует API через `labelingAPI` клиент:

```typescript
// Получить список наборов
const sets = await labelingAPI.list();

// Создать задачи
const tasks = await labelingAPI.createTasks({
    name: "Long strategy",
    method: "triple_barrier",
    config: { ... },
    auto_start: true
});

// Получить данные разметки
const data = await labelingAPI.getData(labelingSetId);
```

## Производительность

### Оптимизации
- Асинхронное выполнение задач в ThreadPoolExecutor
- Chunked обработка больших датасетов (через pandas)
- Персистентное хранение в Parquet для быстрого чтения
- Кеширование метаданных в памяти

### Лимиты
- Максимум 2 параллельных задачи (настраивается)
- Таймаут задачи: без ограничений (управляется вручную)
- Размер данных: без ограничений (зависит от памяти)

## Мониторинг

### Логирование
Все операции логируются:
- Создание/удаление задач
- Прогресс выполнения
- Ошибки и исключения

### Метрики
Доступны через API:
- Количество активных задач
- Статистика выполнения
- Распределение классов

## Будущие улучшения

- [ ] Поддержка пользовательских labelers
- [ ] Batch обработка для больших датасетов
- [ ] Кеширование промежуточных результатов
- [ ] Экспорт результатов в различные форматы
- [ ] A/B тестирование разных методов разметки
- [ ] Автоматическая оптимизация параметров

## Зависимости

Модуль использует:
- `src/labeling/` - методы и фильтры разметки
- `src/data/` - работа с датасетами
- `fastapi` - REST API
- `pandas` - обработка данных
- `pyarrow` - хранение в Parquet

## См. также

- [LABELING_GUIDE.md](../LABELING_GUIDE.md) - руководство пользователя
- [QUICKSTART.md](../QUICKSTART.md) - быстрый старт GUI
- [Этап 05: Разметка таргетов](../../../../plan/Этап_05_Разметка_таргетов.md)

