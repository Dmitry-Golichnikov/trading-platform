# Модуль данных (Data Ingestion)

Модуль для загрузки, хранения, валидации и каталогизации исторических данных.

## Компоненты

### Схемы (`schemas.py`)
- `OHLCVBar` - схема для одного бара с валидацией OHLC соотношений
- `DatasetMetadata` - метаданные датасета с хэшем и временными метками
- `DatasetConfig` - конфигурация загрузки с поддержкой backfill и обновлений

### Загрузчики (`loaders/`)
- `LocalFileLoader` - загрузка из CSV/Parquet, включая формат Tinkoff getHistory
- `CachedDataLoader` - кэширование данных на диске и в памяти
- `TinkoffDataLoader` - заглушка для будущей интеграции с API (Этап 16)

### Хранилище (`storage/`)
- `ParquetStorage` - сохранение/загрузка датасетов по годам
- `DatasetCatalog` - SQLite каталог для быстрого поиска
- `DataVersioning` - версионирование через SHA256 хэширование

### Препроцессоры (`preprocessors/`)
- `timezone` - конвертация в UTC, валидация timezone-aware
- `TimeframeResampler` - агрегация 1m → 5m/15m/1h/4h/1d

### Валидаторы (`validators/`)
- `SchemaValidator` - проверка типов и структуры
- `IntegrityValidator` - дубликаты, пропуски, монотонность
- `QualityValidator` - аномалии цен, проблемы с объёмами

## Примеры использования

### Загрузка данных из локального файла

```python
from src.data.loaders.local_file import LocalFileLoader
from src.data.schemas import DatasetConfig
from datetime import date

# Создать загрузчик
loader = LocalFileLoader(base_path='.')

# Загрузить данные
data = loader.load(
    ticker='SBER',
    from_date=date(2020, 1, 1),
    to_date=date(2023, 12, 31),
    timeframe='1m',
    file_path='cbdf1d32-5758-490e-a2b1-780eaa79bdf7_20200103.csv'
)

print(f"Loaded {len(data)} bars")
```

### Сохранение в хранилище

```python
from src.data.storage.parquet_storage import ParquetStorage

storage = ParquetStorage()

# Сохранить датасет (автоматически разбивается по годам)
metadata = storage.save_dataset(
    data=data,
    ticker='SBER',
    timeframe='1m',
    source='local'
)

print(f"Saved dataset: {metadata.dataset_id}")
print(f"Hash: {metadata.hash}")
```

### Ресэмплинг в старшие таймфреймы

```python
from src.data.preprocessors.resampler import TimeframeResampler

resampler = TimeframeResampler()

# Ресэмплировать 1m → 5m
data_5m = resampler.resample(data, from_tf='1m', to_tf='5m')

# Или сразу во все таймфреймы
all_timeframes = resampler.resample_multiple_timeframes(
    data,
    source_tf='1m',
    target_tfs=['5m', '15m', '1h', '4h', '1d']
)

# Сохранить все таймфреймы
for tf, tf_data in all_timeframes.items():
    storage.save_dataset(tf_data, ticker='SBER', timeframe=tf)
```

### Валидация данных

```python
from src.data.validators import IntegrityValidator, SchemaValidator, QualityValidator

# Проверка схемы
schema_validator = SchemaValidator()
result = schema_validator.validate_all(data)

if not result.is_valid:
    print("Errors:", result.errors)
    print("Warnings:", result.warnings)

# Проверка целостности
integrity_validator = IntegrityValidator()
result = integrity_validator.validate_all(data, timeframe='1m')

print(f"Missing bars: {result.statistics.get('missing_bars', 0)}")
print(f"Duplicates: {result.statistics.get('duplicates', 0)}")

# Проверка качества
quality_validator = QualityValidator()
report = quality_validator.generate_quality_report(data)

print(f"Price anomalies: {report['anomalies']['anomalies']}")
print(f"Zero volumes: {report['volume_issues']['zero_volumes']}")
```

### Асинхронный пайплайн

```python
import asyncio
from datetime import date

from src.data.schemas import DatasetConfig
from src.pipelines import DataPreparationPipeline

config = DatasetConfig(
    ticker=['SBER', 'GAZP'],
    timeframe='1m',
    from_date=date(2020, 1, 1),
    to_date=date(2023, 12, 31),
    source_type='api',
)

async def main() -> None:
    pipeline = DataPreparationPipeline(config, concurrency=4)
    results = await pipeline.run()
    for result in results:
        print(result.ticker, [m.timeframe for m in result.metadata])

asyncio.run(main())
```

### CLI команды

```bash
# Запуск со значениями по умолчанию:
#   - тикеры читаются из tickers_all.txt
#   - базовый таймфрейм 1m, остальное пересчитывается автоматически
#   - диапазон от года листинга (на основе данных Tinkoff) до текущего года
#   - токен берётся из TINKOFF_API_TOKEN / TINKOFF_INVEST_TOKEN
python -m src.interfaces.cli load-data

# Загрузить данные для списка тикеров
python -m src.interfaces.cli load-data \
  --tickers SBER \
  --tickers GAZP \
  --timeframe 1m \
  --from-date 2020-01-01 \
  --to-date 2023-12-31 \
  --source-type api \
  --token $TINKOFF_INVEST_TOKEN \
  --update-latest \
  --backfill-missing \
  --concurrency 1

# Посмотреть список датасетов
python -m src.interfaces.cli list-datasets --ticker SBER

# Валидировать датасет
python -m src.interfaces.cli validate-dataset --ticker SBER --timeframe 1m

# Ресэмплировать вручную
python -m src.interfaces.cli resample-dataset \
  --ticker SBER \
  --source-timeframe 1m \
  --target-timeframe 5m
```

- Токен доступа можно передать через флаг `--token` или переменную окружения `TINKOFF_INVEST_TOKEN`.
- Токен также может быть задан в `.env` (переменные `TINKOFF_API_TOKEN` или `TINKOFF_INVEST_TOKEN`), CLI автоматически его подхватит.
- REST-энпойнт `getHistory` возвращает минутные свечи. Старшие таймфреймы строятся пайплайном автоматически.

### Кэширование

```python
from src.data.loaders.cached import CachedDataLoader

# Обернуть загрузчик в кэш
cached_loader = CachedDataLoader(
    loader=LocalFileLoader(),
    cache_dir='artifacts/cache',
    ttl=86400  # 24 часа
)

# Первый запрос - загрузка из источника
data = cached_loader.load('SBER', from_date, to_date, '1m')

# Второй запрос - из кэша (быстро!)
data = cached_loader.load('SBER', from_date, to_date, '1m')

# Статистика кэша
print(cached_loader.get_stats())
```

### Каталог датасетов

```python
from src.data.storage.catalog import DatasetCatalog

catalog = DatasetCatalog()

# Добавить датасет в каталог
catalog.add_dataset(metadata)

# Поиск датасетов
datasets = catalog.search(
    ticker='SBER',
    timeframe='1m',
    from_date=date(2020, 1, 1)
)

for ds in datasets:
    print(f"{ds.ticker}/{ds.timeframe}: {ds.total_bars} bars")
```

### Версионирование

```python
from src.data.storage.versioning import DataVersioning

versioning = DataVersioning()

# Сохранить версию
version_hash = versioning.save_version(
    data=data,
    ticker='SBER',
    timeframe='1m',
    description='Initial load from Tinkoff API'
)

# Получить историю версий
history = versioning.get_history('SBER', '1m')
for version in history:
    print(f"{version['hash'][:8]}: {version['created_at']}")

# Загрузить конкретную версию
old_data = versioning.load_version('SBER', '1m', version_hash)
```

## Структура хранения

```
artifacts/
├── raw/
│   ├── downloads/
│   │   └── {ticker}/
│   │       └── {year}.zip          # Скачанные архивы
│   └── extracted/
│       └── {ticker}/
│           └── {year}/
│               └── *.csv            # Распакованные дневные файлы
├── data/
│   └── {ticker}/
│       └── {timeframe}/
│           ├── {year}.parquet       # Данные по годам
│           └── metadata.json        # Метаданные
├── cache/                           # Кэш загрузчика
│   └── *.parquet
├── versions/                        # Версионирование
│   └── {ticker}/
│       └── {timeframe}/
│           ├── {hash}.parquet
│           └── {hash}.json
└── db/
    └── catalog.db                   # SQLite каталог
```

## Формат CSV из Tinkoff getHistory

Пример строки:
```
cbdf1d32-5758-490e-a2b1-780eaa79bdf7;2020-01-03T07:04:00Z;13.37;13.37;13.37;13.37;2;
```

Формат: `figi;timestamp;open;high;low;close;volume;`

LocalFileLoader автоматически распознаёт этот формат и парсит его корректно.

## Следующие шаги

- Этап 02: Обработка и валидация данных
- Этап 03: Библиотека индикаторов
- Этап 16: Полная интеграция с Tinkoff Invest API
