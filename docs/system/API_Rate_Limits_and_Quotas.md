# Ограничения и квоты API

## 1. Tinkoff Investments API Limits

### 1.1 Официальные лимиты (актуализировать по документации)

#### Rate Limits
- **Общий лимит**: 300 запросов в минуту на токен
- **По методам**:
  - `GetCandles` (исторические данные): 120 запросов/мин
  - `GetOrderBook` (стакан): 300 запросов/мин
  - `GetTradingStatus`: 60 запросов/мин
  - `GetInstruments`: 30 запросов/мин

#### Квоты
- **Исторические данные**: До 1 года за один запрос
- **Размер ответа**: Максимум 10,000 свечей на запрос
- **Concurrent connections**: До 5 одновременных WebSocket подключений

#### Timeout'ы
- **Connection timeout**: 10 секунд
- **Read timeout**: 30 секунд
- **Total request timeout**: 60 секунд

### 1.2 Неофициальные ограничения
- Возможное throttling при интенсивной загрузке (soft limit)
- Блокировка на 1-5 минут при превышении hard limit
- Приоритизация запросов для активных торговых клиентов

## 2. Стратегия управления Rate Limits

### 2.1 Token Bucket Algorithm

```python
class RateLimiter:
    """Реализация алгоритма Token Bucket"""

    def __init__(self, rate: int, per: float = 60.0):
        """
        Args:
            rate: Количество токенов (requests)
            per: Период в секундах
        """
        self.rate = rate
        self.per = per
        self.tokens = rate
        self.last_update = time.time()
        self.lock = threading.Lock()

    def _refill(self):
        """Пополнить токены"""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(
            self.rate,
            self.tokens + elapsed * (self.rate / self.per)
        )
        self.last_update = now

    def acquire(self, tokens: int = 1, blocking: bool = True) -> bool:
        """
        Получить токены для выполнения запроса

        Args:
            tokens: Количество требуемых токенов
            blocking: Ждать если токенов недостаточно

        Returns:
            True если токены получены
        """
        with self.lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            if not blocking:
                return False

            # Вычислить время ожидания
            wait_time = (tokens - self.tokens) / (self.rate / self.per)

        time.sleep(wait_time)
        return self.acquire(tokens, blocking=False)

    def get_available_tokens(self) -> float:
        """Получить количество доступных токенов"""
        with self.lock:
            self._refill()
            return self.tokens
```

### 2.2 Иерархия лимитеров

```python
class TinkoffAPIRateLimiter:
    """Многоуровневая система лимитов для Tinkoff API"""

    def __init__(self):
        # Глобальный лимит
        self.global_limiter = RateLimiter(rate=300, per=60)

        # Лимиты по методам
        self.method_limiters = {
            'GetCandles': RateLimiter(rate=120, per=60),
            'GetOrderBook': RateLimiter(rate=300, per=60),
            'GetTradingStatus': RateLimiter(rate=60, per=60),
            'GetInstruments': RateLimiter(rate=30, per=60),
        }

        # Счётчики для мониторинга
        self.request_counts = defaultdict(int)
        self.last_reset = time.time()

    def acquire(self, method: str) -> bool:
        """Получить разрешение на выполнение запроса"""
        # Проверить глобальный лимит
        if not self.global_limiter.acquire():
            return False

        # Проверить лимит метода
        method_limiter = self.method_limiters.get(method)
        if method_limiter and not method_limiter.acquire():
            # Вернуть глобальный токен
            self.global_limiter.tokens += 1
            return False

        # Инкремент счётчика
        self.request_counts[method] += 1
        return True

    def wait_if_needed(self, method: str):
        """Ждать если лимит превышен"""
        while not self.acquire(method):
            time.sleep(0.1)
```

### 2.3 Adaptive Rate Limiting

```python
class AdaptiveRateLimiter:
    """Адаптивный лимитер, снижающий rate при ошибках"""

    def __init__(self, base_rate: int, per: float = 60.0):
        self.base_rate = base_rate
        self.current_rate = base_rate
        self.per = per
        self.limiter = RateLimiter(base_rate, per)
        self.consecutive_errors = 0
        self.consecutive_successes = 0

    def on_success(self):
        """Вызывается после успешного запроса"""
        self.consecutive_errors = 0
        self.consecutive_successes += 1

        # Постепенно повышать rate обратно к базовому
        if self.consecutive_successes >= 20 and self.current_rate < self.base_rate:
            self.current_rate = min(self.base_rate, self.current_rate * 1.1)
            self._update_limiter()
            self.consecutive_successes = 0

    def on_rate_limit_error(self):
        """Вызывается при получении 429 Too Many Requests"""
        self.consecutive_successes = 0
        self.consecutive_errors += 1

        # Снизить rate на 50%
        self.current_rate = max(10, self.current_rate * 0.5)
        self._update_limiter()

        logger.warning(
            f"Rate limit hit, reducing to {self.current_rate} req/min"
        )

    def _update_limiter(self):
        """Обновить внутренний лимитер"""
        self.limiter = RateLimiter(int(self.current_rate), self.per)
```

## 3. Оптимизация запросов

### 3.1 Batching

```python
class CandleDataFetcher:
    """Оптимизированная загрузка исторических данных"""

    def __init__(self, api_client, rate_limiter):
        self.api = api_client
        self.rate_limiter = rate_limiter

    def fetch_range(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime,
        interval: str = "1min"
    ) -> pd.DataFrame:
        """Загрузить данные с оптимальным batching"""

        # Разбить диапазон на чанки (max 10k свечей на запрос)
        chunks = self._calculate_optimal_chunks(from_date, to_date, interval)

        all_candles = []
        for chunk_start, chunk_end in chunks:
            # Ожидание доступности токенов
            self.rate_limiter.wait_if_needed('GetCandles')

            try:
                candles = self.api.get_candles(
                    ticker=ticker,
                    from_=chunk_start,
                    to=chunk_end,
                    interval=interval
                )
                all_candles.extend(candles)
                self.rate_limiter.on_success()

            except RateLimitError:
                self.rate_limiter.on_rate_limit_error()
                # Retry с backoff
                time.sleep(60)
                return self.fetch_range(ticker, chunk_start, to_date, interval)

        return pd.DataFrame(all_candles)

    def _calculate_optimal_chunks(
        self,
        from_date: datetime,
        to_date: datetime,
        interval: str
    ) -> list:
        """Рассчитать оптимальные чанки для минимизации запросов"""
        # Количество свечей на один запрос
        max_candles_per_request = 10000

        # Вычислить длительность одного чанка
        interval_minutes = self._parse_interval(interval)
        chunk_duration = timedelta(minutes=interval_minutes * max_candles_per_request)

        chunks = []
        current = from_date
        while current < to_date:
            chunk_end = min(current + chunk_duration, to_date)
            chunks.append((current, chunk_end))
            current = chunk_end

        return chunks
```

### 3.2 Кэширование

```python
class CachedAPIClient:
    """API клиент с кэшированием для минимизации запросов"""

    def __init__(self, api_client, cache_dir: Path):
        self.api = api_client
        self.cache = diskcache.Cache(cache_dir)
        self.cache_ttl = {
            'candles': 86400,        # 1 день
            'instruments': 3600,     # 1 час
            'trading_status': 300,   # 5 минут
        }

    def get_candles(self, ticker: str, from_: datetime, to: datetime, interval: str):
        """Получить свечи с кэшированием"""
        cache_key = f"candles:{ticker}:{interval}:{from_.date()}:{to.date()}"

        # Проверить кэш
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {cache_key}")
            return cached

        # Запрос к API
        logger.debug(f"Cache miss for {cache_key}, fetching from API")
        data = self.api.get_candles(ticker, from_, to, interval)

        # Сохранить в кэш
        self.cache.set(cache_key, data, expire=self.cache_ttl['candles'])
        return data

    def invalidate_cache(self, pattern: str = None):
        """Инвалидировать кэш"""
        if pattern:
            keys = [k for k in self.cache.iterkeys() if pattern in k]
            for key in keys:
                del self.cache[key]
        else:
            self.cache.clear()
```

### 3.3 Parallel Requests с ограничением concurrency

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class ParallelDataFetcher:
    """Параллельная загрузка с учётом rate limits"""

    def __init__(self, api_client, rate_limiter, max_workers: int = 3):
        self.api = api_client
        self.rate_limiter = rate_limiter
        self.max_workers = max_workers

    def fetch_multiple_tickers(
        self,
        tickers: list,
        from_date: datetime,
        to_date: datetime,
        interval: str = "1min"
    ) -> dict:
        """Загрузить данные для нескольких тикеров параллельно"""

        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Создать задачи
            future_to_ticker = {
                executor.submit(
                    self._fetch_ticker_data,
                    ticker,
                    from_date,
                    to_date,
                    interval
                ): ticker
                for ticker in tickers
            }

            # Собрать результаты
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    results[ticker] = data
                    logger.info(f"Fetched {len(data)} candles for {ticker}")
                except Exception as e:
                    logger.error(f"Failed to fetch {ticker}: {e}")
                    results[ticker] = None

        return results

    def _fetch_ticker_data(self, ticker, from_date, to_date, interval):
        """Загрузить данные для одного тикера"""
        fetcher = CandleDataFetcher(self.api, self.rate_limiter)
        return fetcher.fetch_range(ticker, from_date, to_date, interval)
```

## 4. Мониторинг использования квот

### 4.1 Quota Tracker

```python
class QuotaTracker:
    """Отслеживание использования квот"""

    def __init__(self):
        self.stats = {
            'total_requests': 0,
            'requests_by_method': defaultdict(int),
            'rate_limit_hits': 0,
            'data_fetched_mb': 0,
            'start_time': time.time()
        }
        self.lock = threading.Lock()

    def record_request(self, method: str, response_size_bytes: int):
        """Записать выполненный запрос"""
        with self.lock:
            self.stats['total_requests'] += 1
            self.stats['requests_by_method'][method] += 1
            self.stats['data_fetched_mb'] += response_size_bytes / (1024 * 1024)

    def record_rate_limit_hit(self):
        """Записать превышение лимита"""
        with self.lock:
            self.stats['rate_limit_hits'] += 1

    def get_summary(self) -> dict:
        """Получить статистику использования"""
        with self.lock:
            elapsed = time.time() - self.stats['start_time']
            return {
                'total_requests': self.stats['total_requests'],
                'requests_per_minute': self.stats['total_requests'] / (elapsed / 60),
                'by_method': dict(self.stats['requests_by_method']),
                'rate_limit_hits': self.stats['rate_limit_hits'],
                'data_fetched_mb': round(self.stats['data_fetched_mb'], 2),
                'uptime_minutes': round(elapsed / 60, 2)
            }

    def log_summary(self):
        """Залогировать статистику"""
        summary = self.get_summary()
        logger.info(f"API Usage Summary: {json.dumps(summary, indent=2)}")
```

### 4.2 Dashboard метрики

Метрики для Grafana/Streamlit dashboard:
- Requests per minute (текущий / лимит)
- Квоты по методам (использовано / доступно)
- Rate limit hit rate (количество превышений за период)
- Response latency (P50, P95, P99)
- Error rate по типам (network, rate limit, server error)
- Размер загруженных данных (MB/час)

### 4.3 Алерты

```python
class QuotaAlerter:
    """Система алертов при превышении квот"""

    def __init__(self, quota_tracker, alert_handler):
        self.tracker = quota_tracker
        self.alert_handler = alert_handler
        self.thresholds = {
            'rate_limit_hits_per_hour': 10,
            'quota_usage_percent': 80,
            'consecutive_errors': 5
        }

    def check_and_alert(self):
        """Проверить условия алертов"""
        summary = self.tracker.get_summary()

        # Слишком много rate limit hits
        if summary['rate_limit_hits'] > self.thresholds['rate_limit_hits_per_hour']:
            self.alert_handler.send_alert(
                level='warning',
                message=f"Rate limit hit {summary['rate_limit_hits']} times in last hour"
            )

        # Приближение к квоте
        usage_percent = (summary['requests_per_minute'] / 300) * 100
        if usage_percent > self.thresholds['quota_usage_percent']:
            self.alert_handler.send_alert(
                level='warning',
                message=f"API quota usage at {usage_percent:.1f}%"
            )
```

## 5. Обработка исчерпания квот

### 5.1 Backoff стратегии

```python
def exponential_backoff_with_jitter(attempt: int, base_delay: float = 1.0) -> float:
    """Экспоненциальная задержка с jitter"""
    delay = min(base_delay * (2 ** attempt), 300)  # Max 5 минут
    jitter = random.uniform(0, delay * 0.1)  # 10% jitter
    return delay + jitter

def api_call_with_retry(func, *args, max_retries: int = 5, **kwargs):
    """Выполнить API вызов с retry при rate limit"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)

        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise

            # Извлечь retry-after из заголовков
            retry_after = getattr(e, 'retry_after', None)
            if retry_after:
                delay = float(retry_after)
            else:
                delay = exponential_backoff_with_jitter(attempt)

            logger.warning(
                f"Rate limit hit, retrying after {delay:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(delay)

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
```

### 5.2 Queue-based подход

```python
class APIRequestQueue:
    """Очередь запросов с автоматическим управлением rate limit"""

    def __init__(self, rate_limiter, max_queue_size: int = 1000):
        self.rate_limiter = rate_limiter
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.results = {}
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def enqueue(self, request_id: str, method: str, func, *args, **kwargs) -> str:
        """Добавить запрос в очередь"""
        self.queue.put({
            'request_id': request_id,
            'method': method,
            'func': func,
            'args': args,
            'kwargs': kwargs
        })
        return request_id

    def get_result(self, request_id: str, timeout: float = None):
        """Получить результат запроса"""
        start = time.time()
        while True:
            if request_id in self.results:
                result = self.results.pop(request_id)
                if isinstance(result, Exception):
                    raise result
                return result

            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"Request {request_id} timed out")

            time.sleep(0.1)

    def _process_queue(self):
        """Worker поток для обработки очереди"""
        while True:
            try:
                request = self.queue.get()
                request_id = request['request_id']
                method = request['method']

                # Ожидание доступности токенов
                self.rate_limiter.wait_if_needed(method)

                # Выполнение запроса
                try:
                    result = request['func'](*request['args'], **request['kwargs'])
                    self.results[request_id] = result
                except Exception as e:
                    self.results[request_id] = e

                self.queue.task_done()

            except Exception as e:
                logger.error(f"Queue processing error: {e}")
```

## 6. Конфигурация

### 6.1 Файл конфигурации

```yaml
# configs/api_limits.yaml
tinkoff_api:
  rate_limits:
    global:
      rate: 300
      per_seconds: 60

    by_method:
      GetCandles:
        rate: 120
        per_seconds: 60
      GetOrderBook:
        rate: 300
        per_seconds: 60
      GetTradingStatus:
        rate: 60
        per_seconds: 60

  quotas:
    max_candles_per_request: 10000
    max_date_range_days: 365
    max_concurrent_connections: 5

  retry:
    max_attempts: 5
    base_delay_seconds: 1.0
    max_delay_seconds: 300
    exponential_backoff: true
    jitter_factor: 0.1

  timeouts:
    connect_seconds: 10
    read_seconds: 30
    total_seconds: 60

  caching:
    enabled: true
    ttl_seconds:
      candles: 86400
      instruments: 3600
      trading_status: 300

  monitoring:
    alert_thresholds:
      rate_limit_hits_per_hour: 10
      quota_usage_percent: 80
      error_rate_percent: 5
```

### 6.2 Загрузка конфигурации

```python
class APIConfig:
    """Конфигурация API лимитов"""

    @classmethod
    def from_yaml(cls, path: Path):
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config['tinkoff_api'])

    def __init__(self, config: dict):
        self.rate_limits = config['rate_limits']
        self.quotas = config['quotas']
        self.retry = config['retry']
        self.timeouts = config['timeouts']
        self.caching = config['caching']
        self.monitoring = config['monitoring']
```

## 7. Best Practices

### 7.1 Рекомендации
- ✅ Всегда использовать rate limiter для API вызовов
- ✅ Кэшировать данные, которые редко меняются
- ✅ Использовать batch запросы где возможно
- ✅ Мониторить использование квот
- ✅ Настроить алерты при приближении к лимитам
- ✅ Реализовать graceful degradation при исчерпании квот

### 7.2 Антипаттерны
- ❌ Игнорирование rate limits и retry без задержки
- ❌ Запросы в бесконечном цикле без backoff
- ❌ Повторная загрузка одних и тех же данных
- ❌ Отсутствие мониторинга использования API
- ❌ Хранение credentials в коде
