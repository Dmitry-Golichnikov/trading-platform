# Конфигурации разметки таргетов

Эта директория содержит готовые конфигурации для различных методов разметки.

## Доступные конфигурации

### Classification (Long/Short)

1. **long_only.yaml** - Long-only стратегия
   - Triple Barrier метод
   - Асимметричные барьеры (TP 2%, SL 1%)
   - Только long позиции
   - Строгая фильтрация

2. **long_short.yaml** - Long+Short стратегия
   - Triple Barrier метод
   - Симметричные барьеры (2% с обеих сторон)
   - Majority vote фильтр
   - Effective samples балансировка

3. **triple_barrier_atr.yaml** - Адаптивные ATR барьеры
   - Барьеры на основе волатильности (ATR)
   - Типичная цена (H+L+C)/3
   - Comprehensive фильтрация
   - Long+Short

### Horizon Methods

4. **horizon_fixed.yaml** - Фиксированный горизонт
   - Простой fixed horizon метод
   - Горизонт 20 баров
   - Порог 1%
   - Быстрая разметка

5. **horizon_adaptive.yaml** - Адаптивный горизонт
   - Горизонт адаптируется к волатильности
   - ATR-based адаптация
   - Диапазон 5-50 баров

### Regression Targets

6. **regression_returns.yaml** - Future Returns
   - Предсказание будущих returns
   - Горизонт 20 баров
   - Z-score нормализация

7. **regression_mfe_mae.yaml** - Max Excursions
   - MFE (Max Favorable Excursion)
   - MAE (Max Adverse Excursion)
   - Горизонт 30 баров

## Использование

### Через CLI

```bash
# Long-only стратегия
python -m src.interfaces.cli labels label-dataset \
    --data-path data/SBER_1h.parquet \
    --config configs/labeling/long_only.yaml \
    --dataset-id SBER_1h

# Адаптивный горизонт
python -m src.interfaces.cli labels label-dataset \
    --data-path data/BTCUSDT_15m.parquet \
    --config configs/labeling/horizon_adaptive.yaml \
    --visualize

# Regression таргеты
python -m src.interfaces.cli labels label-dataset \
    --data-path data/EURUSD_1h.parquet \
    --config configs/labeling/regression_returns.yaml
```

### Через Python API

```python
from pathlib import Path
import pandas as pd
import yaml

from src.labeling.pipeline import LabelingPipeline

# Загрузка данных
data = pd.read_parquet('data/SBER_1h.parquet')

# Загрузка конфигурации
with open('configs/labeling/long_only.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['dataset_id'] = 'SBER_1h'

# Создание и запуск пайплайна
pipeline = LabelingPipeline.from_config(config, data)
labeled_data, metadata = pipeline.run(data)

print(metadata.get_summary())
```

## Параметры конфигурации

### Общая структура

```yaml
method: <метод_разметки>  # horizon, triple_barrier, regression, custom

params:
  # Параметры метода разметки
  ...

filters:
  # Список постфильтров
  - type: <тип_фильтра>
    params:
      ...

balancing:
  # Параметры балансировки классов
  method: class_weights
  strategy: balanced
```

### Методы разметки

- **horizon**: Fixed или adaptive горизонт
- **triple_barrier**: Triple Barrier метод (Lopez de Prado)
- **regression**: Regression таргеты (returns, MFE, MAE, Sharpe)
- **custom**: Кастомные правила

### Постфильтры

- **smoothing**: Сглаживание меток (moving_average, exponential, median)
- **sequence**: Фильтрация коротких последовательностей
- **majority_vote**: Мажоритарное голосование
- **danger_zones**: Фильтрация опасных зон (высокая волатильность, gap'ы)

### Балансировка

- **balanced**: sklearn balanced weights
- **effective_samples**: Effective Number of Samples (Cui et al., 2019)
- **custom**: Кастомные веса

## Создание собственной конфигурации

1. Скопируйте один из примеров
2. Измените параметры под свою стратегию
3. Протестируйте на небольшом датасете
4. Проанализируйте результаты через `analyze-labels`

## Рекомендации

### Для краткосрочной торговли (< 1 день)
- Используйте **horizon_fixed.yaml** или **long_short.yaml**
- Короткие горизонты (5-20 баров)
- Жёсткие фильтры (sequence_filter с min_length=2+)

### Для среднесрочной торговли (1-7 дней)
- Используйте **triple_barrier_atr.yaml**
- Адаптивные барьеры на основе ATR
- Горизонты 20-50 баров

### Для долгосрочной торговли (> 7 дней)
- Используйте **horizon_adaptive.yaml** или **long_only.yaml**
- Большие горизонты (50+ баров)
- Менее агрессивные фильтры

### Для ML моделей
- **Classification**: Используйте triple_barrier или horizon с фильтрами
- **Regression**: Используйте regression_returns или regression_mfe_mae

## Troubleshooting

### Слишком мало примеров одного класса

Попробуйте:
1. Уменьшить порог threshold_pct
2. Увеличить горизонт
3. Ослабить фильтры (убрать sequence_filter)

### Слишком много neutral меток

Попробуйте:
1. Уменьшить min_return
2. Увеличить барьеры (для triple_barrier)
3. Проверить danger_zones фильтр

### Несбалансированные классы

Попробуйте:
1. Изменить balancing strategy на effective_samples
2. Использовать sequence-aware weighting
3. Adjust upper/lower barriers для симметрии

## Дополнительная информация

См. документацию:
- `plan/Этап_05_Разметка_таргетов.md` - Детальное описание этапа
- `src/labeling/` - Исходный код модуля
- `docs/labeling/` - Дополнительная документация (если есть)
