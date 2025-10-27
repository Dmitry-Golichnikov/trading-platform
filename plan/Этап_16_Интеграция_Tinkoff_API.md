# Этап 16: Интеграция с Tinkoff Investments API

## Цель
Реализовать полноценную интеграцию с Tinkoff Investments API для загрузки данных.

## Зависимости
Этап 01 (обновление), Этап 17 (rate limiting)

## Компоненты

```
src/data/loaders/tinkoff/
├── __init__.py
├── client.py                     # TinkoffAPIClient
├── adapter.py                    # TinkoffAdapter
├── rate_limiter.py               # Rate limiting
├── instruments.py                # Поиск инструментов
├── market_data.py                # Market data
└── utils.py                      # Утилиты
```

## Основные классы

### 1. TinkoffAPIClient
```python
class TinkoffAPIClient:
    """Клиент Tinkoff API"""

    def __init__(self, token: str, sandbox: bool = False):
        self.token = token
        self.client = InvestAPI(token, sandbox=sandbox)
        self.rate_limiter = RateLimiter(max_rps=5)

    def get_candles(
        self,
        figi: str,
        from_: datetime,
        to: datetime,
        interval: CandleInterval
    ) -> list[Candle]:
        """Получить свечи с rate limiting"""
```

### 2. TinkoffDataLoader
```python
@DataLoaderRegistry.register('tinkoff')
class TinkoffDataLoader(BaseDataLoader):
    """Загрузчик данных из Tinkoff API"""

    def load(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime,
        timeframe: str = '1m'
    ) -> pd.DataFrame:
        """
        Загрузить данные с:
        - Пагинацией (годовые архивы для 1m)
        - Rate limiting
        - Retry logic
        - Кэшированием
        """
```

### 3. InstrumentSearch
```python
class InstrumentSearch:
    """Поиск инструментов"""

    def search_by_ticker(self, ticker: str) -> list[Instrument]:
        """Поиск по тикеру"""

    def get_figi(self, ticker: str) -> str:
        """Получить FIGI по тикеру"""

    def get_all_shares(self) -> list[Instrument]:
        """Все акции"""
```

### 4. Rate Limiter
```python
class RateLimiter:
    """Rate limiting (Token Bucket)"""

    def __init__(
        self,
        max_rps: int = 5,  # 300 req/min = 5 req/sec
        burst: int = 10
    ):
        self.max_rps = max_rps
        self.burst = burst
        self.tokens = burst

    async def acquire(self):
        """Ждать доступности токена"""
```

## Особенности API

### Лимиты (docs/system/API_Rate_Limits_and_Quotas.md)
- 300 requests/minute
- 5 requests/second
- Пагинация данных
- Минутные данные: только через годовые архивы

### Загрузка исторических данных
```python
async def download_historical_1m(
    ticker: str,
    from_year: int,
    to_year: int
) -> pd.DataFrame:
    """
    Загрузить минутные данные по годам
    Для каждого года - отдельный запрос (GetCandles)
    """

    data_frames = []
    for year in range(from_year, to_year + 1):
        year_data = await client.get_candles(
            figi=get_figi(ticker),
            from_=datetime(year, 1, 1),
            to=datetime(year, 12, 31, 23, 59, 59),
            interval=CandleInterval.CANDLE_INTERVAL_1_MIN
        )
        data_frames.append(convert_to_df(year_data))

    return pd.concat(data_frames, ignore_index=True)
```

### Resample higher timeframes
После загрузки 1m → resample в 5m, 15m, 1h, 4h, 1d

## Конфигурация

**configs/data/tinkoff.yaml:**
```yaml
source: tinkoff
api:
  token: ${TINKOFF_API_TOKEN}  # Из env
  sandbox: false

rate_limits:
  max_rps: 5
  burst: 10
  retry_attempts: 3
  retry_delay: 5

download:
  chunk_size: year  # Для 1m данных
  cache_dir: artifacts/data/tinkoff_cache

instruments:
  auto_update: true
  cache_ttl: 86400  # 1 день
```

## CLI команды

```bash
# Поиск инструмента
trading-cli data tinkoff search --ticker SBER

# Загрузка данных
trading-cli data tinkoff download \
  --ticker SBER \
  --from 2020-01-01 \
  --to 2023-12-31 \
  --timeframe 1m \
  --output artifacts/data/SBER/

# Обновление списка инструментов
trading-cli data tinkoff instruments update

# Список доступных инструментов
trading-cli data tinkoff instruments list --filter "type:share"
```

## Критерии готовности

- [ ] TinkoffAPIClient работает
- [ ] TinkoffDataLoader реализован
- [ ] Rate limiting работает
- [ ] Retry logic при ошибках
- [ ] Загрузка исторических 1m данных
- [ ] Автоматический resample
- [ ] Поиск инструментов
- [ ] Кэширование
- [ ] CLI команды
- [ ] Обработка ошибок API

## Промпты

```
Реализуй Tinkoff integration в src/data/loaders/tinkoff/:

1. client.py - TinkoffAPIClient с rate limiting
2. adapter.py - TinkoffDataLoader (реализация из Этапа 01)
3. rate_limiter.py - RateLimiter (Token Bucket algorithm)
4. instruments.py - InstrumentSearch
5. market_data.py - Вспомогательные функции для market data

Требования:
- Async/await для API calls
- Retry logic (exponential backoff)
- Rate limiting (300 req/min)
- Пагинация для больших диапазонов
- Конвертация Quotation → float
- Обработка timezone (UTC)
- Кэширование результатов

CLI команды в src/interfaces/cli/data_commands.py

Используй спецификации из docs/system/API_Rate_Limits_and_Quotas.md
```

## Важные замечания

**Token Bucket:**
Алгоритм для rate limiting, позволяет burst requests.

**Годовые архивы:**
Для 1m данных Tinkoff требует загрузку по годам.

**Quotation:**
Tinkoff использует Quotation (units + nano), нужна конвертация в float.

**Sandbox:**
Есть sandbox режим для тестирования.

**Async:**
Используй async/await для эффективных сетевых запросов.

## Следующий этап
[Этап 17: Мониторинг и логирование](Этап_17_Мониторинг_и_логирование.md)
