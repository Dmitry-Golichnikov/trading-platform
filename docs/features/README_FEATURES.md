# Модуль генерации признаков

## Обзор

Модуль `src.features` предоставляет декларативную систему генерации признаков для машинного обучения с поддержкой:

- Ценовых, объёмных, календарных признаков
- Технических индикаторов (30+)
- Rolling статистик, лагов, разностей, соотношений
- Признаков из старших таймфреймов
- Feature selection (8 методов)
- Кэширования с версионированием

## Быстрый старт

### 1. Создание конфигурации

Создайте YAML файл с описанием признаков:

```yaml
# my_features.yaml
version: "1.0"
cache_enabled: true

features:
  # Ценовые признаки
  - type: price
    features:
      - returns
      - high_low_ratio

  # Индикаторы
  - type: indicator
    name: SMA
    params:
      window: 20

  # Rolling статистики
  - type: rolling
    window: 10
    functions: [mean, std]
    columns: [close, volume]

# Feature selection (опционально)
selection:
  enabled: true
  method: correlation
  top_k: 30
```

### 2. Генерация признаков

```python
from src.features import FeatureGenerator
import pandas as pd

# Загрузка данных
data = pd.read_parquet("SBER_1m.parquet")

# Инициализация генератора
generator = FeatureGenerator("my_features.yaml")

# Генерация
features = generator.generate(
    data=data,
    dataset_id="SBER_1m",
    target=target_series  # опционально для selection
)

# Результат
print(f"Сгенерировано {features.shape[1]} признаков")
```

### 3. CLI использование

```bash
# Генерация признаков
python -m src.interfaces.cli features generate \
    -c configs/features/default.yaml \
    -d SBER_1m \
    -o features.parquet

# Список кэшированных признаков
python -m src.interfaces.cli features list

# Очистка кэша
python -m src.interfaces.cli features clear-cache -d SBER_1m

# Валидация конфигурации
python -m src.interfaces.cli features validate-config my_features.yaml
```

## Типы признаков

### Price Features (Ценовые)

```yaml
- type: price
  features:
    - returns           # Простые доходности
    - log_returns       # Логарифмические доходности
    - high_low_ratio    # High/Low
    - close_open_ratio  # Close/Open
    - body_size         # Размер тела свечи
    - upper_wick        # Верхний фитиль
    - lower_wick        # Нижний фитиль
    - price_position    # Позиция цены в диапазоне
```

### Volume Features (Объёмные)

```yaml
- type: volume
  features:
    - volume_change      # Изменение объёма
    - volume_ma_ratio    # Отношение к MA
    - money_volume       # Денежный объём
    - relative_volume    # Z-score объёма
    - volume_volatility  # Волатильность объёма
```

### Calendar Features (Календарные)

```yaml
- type: calendar
  features:
    - hour                  # Час (0-23)
    - day_of_week          # День недели (0-6)
    - month                # Месяц (1-12)
    - is_month_start       # Начало месяца
    - is_month_end         # Конец месяца
    - trading_day_of_month # Торговый день месяца
    - time_since_open      # Время с открытия
    - time_to_close        # Время до закрытия
```

### Indicator Features (Индикаторы)

```yaml
- type: indicator
  name: SMA  # или EMA, RSI, MACD, ATR, ...
  params:
    window: 20
  prefix: "short"  # опционально
```

Доступные индикаторы: SMA, EMA, WMA, MACD, RSI, Stochastic, ATR, BollingerBands, ADX, VWAP, OBV и др.

### Rolling Statistics

```yaml
- type: rolling
  window: 20
  functions:
    - mean
    - std
    - min
    - max
    - skew
    - kurt
  columns: [close, volume]
  min_periods: 1  # опционально
```

### Lags (Лаги)

```yaml
- type: lags
  lags: [1, 2, 5, 10]
  columns: [close, volume]
```

### Differences (Разности)

```yaml
- type: differences
  periods: [1, 5]
  method: pct_change  # или diff
  columns: [close]
```

### Ratios (Соотношения)

```yaml
- type: ratios
  pairs:
    - [close, open]
    - [high, low]
```

### Higher Timeframe Features

```yaml
- type: higher_timeframe
  source_tf: "1h"  # или 4h, 1d
  indicators:
    - SMA_20
    - RSI_14
  alignment: forward_fill  # каузальное
```

## Feature Selection

### Variance Threshold

```yaml
selection:
  enabled: true
  method: variance_threshold
  params:
    threshold: 0.01
```

### Correlation

```yaml
selection:
  enabled: true
  method: correlation
  params:
    threshold: 0.05
  top_k: 50
```

### Mutual Information

```yaml
selection:
  enabled: true
  method: mutual_info
  params:
    random_state: 42
  top_k: 30
```

### Tree-based Importance

```yaml
selection:
  enabled: true
  method: tree_importance
  params:
    n_estimators: 100
    max_depth: 10
  top_k: 50
```

### SHAP Values

```yaml
selection:
  enabled: true
  method: shap
  top_k: 30
```

## Кэширование

Кэш автоматически хранит сгенерированные признаки:

```python
from src.features import FeatureCache

# Инициализация
cache = FeatureCache("artifacts/features")

# Статистика
stats = cache.get_stats()
print(f"Размер кэша: {stats['size_mb']} MB")
print(f"Записей: {stats['total_entries']}")

# Список кэшированных
cached = cache.list_cached()
print(cached)

# Инвалидация
cache.invalidate(dataset_id="SBER_1m")
```

## Drift Detection

```python
from src.features.selectors import DriftDetector

# PSI
detector = DriftDetector(method="psi", threshold=0.2)
scores, drifted = detector.detect(X_train, X_test)

print(f"Дрифт обнаружен в {len(drifted)} признаках")
print(drifted)
```

## Производительность

- 10,000 баров: < 10 секунд (все признаки)
- Кэширование: 10-100x ускорение
- Векторизованные операции

## Важные замечания

### Каузальность

Все признаки **каузальные** - не используют будущую информацию:
- ✅ Лаги: `shift(n)` где n > 0
- ✅ Rolling: только прошлые значения
- ✅ Higher TF: forward fill
- ❌ backward fill, interpolate - не каузальные!

### NaN значения

Признаки могут генерировать NaN:
- Лаги: первые N строк
- Rolling: первые window-1 строк (если min_periods не указан)
- Индикаторы: зависит от периода

### Масштабирование

Масштабирование (normalization) **не выполняется** на этом этапе.
Используйте `sklearn.preprocessing` или аналоги в пайплайне обучения.

## Примеры конфигураций

### Минимальная (для тестов)

См. `configs/features/minimal.yaml`

### Дефолтная (сбалансированная)

См. `configs/features/default.yaml`

### Полная (максимум признаков)

См. `configs/features/full.yaml`

## API Reference

### FeatureGenerator

```python
generator = FeatureGenerator(
    config,              # Path | dict | FeatureConfig
    cache_enabled=True,  # Включить кэш
    cache_dir=None       # Директория кэша
)

features = generator.generate(
    data,                # DataFrame с OHLC
    dataset_id=None,     # ID для кэша
    target=None,         # Series для selection
    use_cache=True       # Использовать кэш
)
```

### FeatureCache

```python
cache = FeatureCache(cache_dir)

# Получить из кэша
features = cache.get(dataset_id, config)

# Сохранить в кэш
cache.save(dataset_id, config, features, metadata)

# Инвалидировать
cache.invalidate(dataset_id)

# Статистика
stats = cache.get_stats()
```

## См. также

- [Технические индикаторы](../indicators/)
- [Этап 04 - План](../../plan/Этап_04_Генерация_признаков.md)
- [Technical Spec](../../technical_spec.md)
