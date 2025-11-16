# Feature Generation в GUI

## Обзор

Модуль генерации признаков позволяет создавать наборы признаков (feature sets) из существующих датасетов прямо через веб-интерфейс. Это упрощает процесс подготовки данных для обучения моделей.

## Основные возможности

### 1. Создание наборов признаков
- Выбор исходного датасета
- Именование набора признаков
- Описание для документации
- Выбор конфигурации признаков

### 2. Типы конфигураций

#### Дефолтная конфигурация
Включает стандартный набор технических индикаторов:
- **SMA** (Simple Moving Average, период 20)
- **EMA** (Exponential Moving Average, период 20)
- **RSI** (Relative Strength Index, период 14)
- **MACD** (Moving Average Convergence Divergence)
- **Bollinger Bands** (период 20, стандартное отклонение 2)

И трансформации:
- **Лаги**: 1, 2, 3, 5, 10 периодов
- **Скользящие окна**: mean и std для окон 5, 10, 20 периодов

#### Кастомная конфигурация
Позволяет задать собственную конфигурацию через YAML файл или inline JSON.

### 3. Просмотр наборов признаков
- Таблица со всеми созданными наборами
- Информация: название, датасет, количество признаков, дата создания, статус
- Фильтрация по датасету
- Сортировка по дате

### 4. Управление наборами
- Просмотр деталей набора признаков
- Просмотр данных (с ограничением на количество строк)
- Удаление устаревших наборов
- Фильтрация по датасету, статусу, конфигурации

### 5. Управление задачами генерации ✨
- Запуск задач по одному или нескольким датасетам
- Опция *“Run for all datasets”* — автоматический прогон по всему каталогу
- Инкрементальное обновление: достраиваем признаки только на новых данных
- Настройка размера чанка и шага сохранения прогресса
- Пауза, возобновление, отмена и перезапуск задач
- Прогресс-бар и Live‑статус через таблицу задач
- WebSocket-нотификации (через `/ws/task/{task_id}`)

## Использование через UI

### Шаг 1: Открыть страницу Features
Перейдите в раздел "Features" в боковом меню (иконка звезды ✨).

### Шаг 2: Перейти на вкладку *Tasks*
Слева располагается конфигуратор признаков, справа — список задач.

### Шаг 3: Настроить конфигурацию
1. Выберите индикаторы (трендовые/моментум/волатильность/объём)
2. Отметьте дополнительные блоки: календарь, one-hot тикеров, higher timeframe, лаги, rolling-статистики, разности
3. Укажите название набора, описание, chunk size, режим incremental

### Шаг 4: Выбрать датасеты
- По умолчанию переключатель "Для всех датасетов" **выключен**, чтобы можно было сразу выбрать конкретные id
- Включите переключатель, если хотите прогнать конфигурацию по всему каталогу
- Если каталога данных ещё нет, появится предупреждение и кнопка запуска будет заблокирована

### Шаг 5: Запустить задачи
Нажмите **“Запустить генерацию”**. Каждому датасету соответствует отдельная задача (с batch_id).

### Шаг 6: Следить за прогрессом
- Список задач обновляется каждые 4 секунды
- Можно ставить на паузу, возобновлять, отменять или перезапускать задачу из карточки
- После завершения новый набор автоматически появляется на вкладке *Sets*

## Использование через API

### Создание набора признаков

#### С дефолтной конфигурацией
```bash
curl -X POST http://localhost:8000/api/features/generate \
  -H "Content-Type: application/json" \
  -d '{
    "name": "SBER Technical Indicators",
    "dataset_id": "SBER_1h",
    "description": "Default technical indicators for SBER 1h data"
  }'
```

#### С кастомной конфигурацией
```bash
curl -X POST http://localhost:8000/api/features/generate \
  -H "Content-Type: application/json" \
  -d '{
    "name": "SBER Custom Features",
    "dataset_id": "SBER_1h",
    "config": {
      "indicators": [
        {"name": "SMA", "params": {"period": 10}},
        {"name": "SMA", "params": {"period": 50}},
        {"name": "RSI", "params": {"period": 7}},
        {"name": "ATR", "params": {"period": 14}}
      ],
      "transformers": [
        {"type": "lags", "params": {"lags": [1, 5, 10, 20]}}
      ]
    }
  }'
```

#### С конфигурацией из файла
```bash
curl -X POST http://localhost:8000/api/features/generate \
  -H "Content-Type: application/json" \
  -d '{
    "name": "SBER Advanced Features",
    "dataset_id": "SBER_1h",
    "config_path": "features/advanced_config.yml"
  }'
```

### Получение списка наборов признаков

```bash
# Все наборы
curl http://localhost:8000/api/features

# Фильтрация по датасету
curl http://localhost:8000/api/features?dataset_id=SBER_1h
```

### Получение информации о наборе

```bash
curl http://localhost:8000/api/features/{feature_set_id}
```

### Получение данных признаков

```bash
# Первые 1000 строк
curl http://localhost:8000/api/features/{feature_set_id}/data?limit=1000

# Выбранные колонки
curl "http://localhost:8000/api/features/{feature_set_id}/data?columns=SMA_20&columns=RSI_14&limit=500"
```

### Удаление набора признаков

```bash
curl -X DELETE http://localhost:8000/api/features/{feature_set_id}
```

## Структура данных

### Метаданные набора признаков
```json
{
  "id": "features_20251116_143022_SBER_1h",
  "name": "SBER Technical Indicators",
  "dataset_id": "SBER_1h",
  "config": {
    "indicators": [...],
    "transformers": [...]
  },
  "num_features": 42,
  "num_rows": 10000,
  "created_at": "2025-11-16T14:30:22.123456",
  "status": "completed",
  "description": "Default technical indicators",
  "columns": ["SMA_20", "EMA_20", "RSI_14", ...]
}
```

### Формат хранения
- **Данные**: `artifacts/features/{feature_set_id}.parquet`
- **Метаданные**: `artifacts/features/{feature_set_id}_metadata.json`

## Примеры конфигураций

### Простая конфигурация с индикаторами
```yaml
indicators:
  - name: SMA
    params:
      period: 20
  - name: EMA
    params:
      period: 20
  - name: RSI
    params:
      period: 14
```

### Расширенная конфигурация
```yaml
indicators:
  # Трендовые индикаторы
  - name: SMA
    params:
      period: 10
  - name: SMA
    params:
      period: 50
  - name: SMA
    params:
      period: 200
  - name: EMA
    params:
      period: 12
  - name: EMA
    params:
      period: 26

  # Моментум индикаторы
  - name: RSI
    params:
      period: 14
  - name: Stochastic
    params:
      k_period: 14
      d_period: 3

  # Волатильность
  - name: BollingerBands
    params:
      period: 20
      std_dev: 2
  - name: ATR
    params:
      period: 14

  # Объемные
  - name: OBV
    params: {}
  - name: VWAP
    params: {}

transformers:
  # Лаги
  - type: lags
    params:
      lags: [1, 2, 3, 5, 10, 20]

  # Скользящие окна
  - type: rolling
    params:
      windows: [5, 10, 20]
      functions: [mean, std, min, max]

  # Разности
  - type: differences
    params:
      periods: [1, 5, 10]

  # Соотношения
  - type: ratios
    params:
      columns:
        - [close, SMA_20]
        - [high, low]
```

### Конфигурация для multi-timeframe признаков
```yaml
indicators:
  - name: SMA
    params:
      period: 20
  - name: RSI
    params:
      period: 14

higher_timeframes:
  - timeframe: 4h
    indicators:
      - name: SMA
        params:
          period: 50
      - name: MACD
        params:
          fast: 12
          slow: 26
          signal: 9

transformers:
  - type: lags
    params:
      lags: [1, 3, 5]
```

## Best Practices

### 1. Именование
- Используйте осмысленные названия: `{ticker}_{timeframe}_{feature_type}`
- Например: `SBER_1h_technical_indicators`, `GAZP_15m_volume_features`

### 2. Документация
- Всегда заполняйте описание (description)
- Указывайте цель набора признаков
- Документируйте нестандартные параметры

### 3. Версионирование
- При изменении конфигурации создавайте новый набор
- Не перезаписывайте существующие наборы
- Используйте описание для указания версии

### 4. Производительность
- Начинайте с дефолтной конфигурации
- Постепенно добавляйте признаки
- Используйте feature selection для отбора важных признаков

### 5. Организация
- Группируйте наборы по типам (technical, volume, calendar)
- Удаляйте неиспользуемые наборы
- Регулярно проверяйте качество признаков

## Troubleshooting

### Ошибка "Dataset not found"
**Проблема**: Датасет не найден в системе.

**Решение**:
- Проверьте, что датасет существует: `ls artifacts/data/{ticker}/{timeframe}/`
- Убедитесь, что файл `data.parquet` присутствует
- Проверьте правильность dataset_id

### Ошибка "Failed to generate features"
**Проблема**: Ошибка во время генерации признаков.

**Решение**:
- Проверьте логи backend для деталей
- Убедитесь, что конфигурация корректна
- Проверьте, что датасет содержит все необходимые колонки (open, high, low, close, volume)

### Набор признаков со статусом "failed"
**Проблема**: Генерация завершилась с ошибкой.

**Решение**:
- Откройте файл `artifacts/features/{feature_set_id}_metadata.json`
- Посмотрите поле `error` для деталей ошибки
- Исправьте конфигурацию и попробуйте снова

### Медленная генерация
**Проблема**: Процесс занимает слишком много времени.

**Решение**:
- Уменьшите количество индикаторов
- Уменьшите количество лагов и окон
- Используйте меньший датасет для тестирования
- Проверьте загрузку CPU/RAM

## Интеграция с другими модулями

### Использование в экспериментах
```python
# В конфигурации эксперимента
config = {
    "feature_set_id": "features_20251116_143022_SBER_1h",
    "model_type": "LightGBM",
    # ... остальные параметры
}
```

### Использование в пайплайнах
```python
from src.features.generator import FeatureGenerator

# Загрузка сохраненной конфигурации
import json
with open("artifacts/features/{feature_set_id}_metadata.json") as f:
    metadata = json.load(f)
    config = metadata["config"]

# Применение к новым данным
generator = FeatureGenerator(config)
features = generator.generate(new_data)
```

## Roadmap

### Ближайшие улучшения
- [ ] Предпросмотр признаков в UI
- [ ] Визуализация корреляций
- [ ] Автоматический feature selection
- [ ] Шаблоны конфигураций
- [ ] Сравнение наборов признаков
- [ ] Экспорт в другие форматы (CSV, Feather)

### Долгосрочные планы
- [ ] Feature store интеграция
- [ ] Инкрементальная генерация (для новых данных)
- [ ] Параллельная генерация для нескольких датасетов
- [ ] Feature engineering notebooks
- [ ] Автоматическая генерация признаков (AutoFE)
- [ ] Feature importance tracking
