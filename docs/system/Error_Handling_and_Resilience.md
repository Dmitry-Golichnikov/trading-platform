# Обработка ошибок и отказоустойчивость

## 1. Общие принципы

### 1.1 Философия обработки ошибок
- **Fail Fast**: Раннее обнаружение ошибок на этапе валидации входных данных
- **Fail Safe**: Graceful degradation при некритичных сбоях
- **Transparency**: Подробное логирование причин ошибок для диагностики
- **Recoverability**: Возможность восстановления без потери прогресса

### 1.2 Классификация ошибок
- **Критичные**: Требуют немедленной остановки (повреждение данных, OOM)
- **Восстанавливаемые**: Можно повторить операцию (сетевые ошибки, временные сбои)
- **Предупреждения**: Не блокируют выполнение, но требуют внимания (missing values, drift)
- **Информационные**: Ожидаемые события (пустые результаты фильтров)

## 2. Стратегии обработки по модулям

### 2.1 Data Ingestion (`src/data/`)

#### Ошибки API Tinkoff
```python
class TinkoffAPIError(Exception):
    """Базовая ошибка API"""
    pass

class RateLimitError(TinkoffAPIError):
    """Превышен лимит запросов"""
    pass

class DataUnavailableError(TinkoffAPIError):
    """Данные недоступны для запрошенного периода"""
    pass
```

**Стратегия retry**:
- **Network errors**: 3 попытки с exponential backoff (1s, 2s, 4s)
- **Rate limits**: Ожидание указанного времени + jitter
- **Server errors (5xx)**: 5 попыток с увеличенным интервалом
- **Client errors (4xx)**: Без retry, немедленный fail

**Fallback**:
- Использование кэшированных данных, если доступны
- Переключение на локальные файлы при длительной недоступности API
- Уведомление пользователя о режиме degraded mode

#### Ошибки чтения файлов
- **Повреждённые файлы**: Попытка восстановления с использованием backup
- **Отсутствующие файлы**: Проверка альтернативных источников
- **Несовместимый формат**: Конвертация через промежуточные форматы
- **Encoding issues**: Автодетект кодировки с fallback на UTF-8

#### Проблемы целостности данных
- **Дубликаты**: Удаление с логированием (по timestamp + ticker)
- **Пропуски**: Заполнение forward-fill для OHLCV (с лимитом пропусков)
- **Несоответствие OHLC**: Коррекция или отбраковка бара
- **Отрицательные объёмы**: Отбраковка с алертом

### 2.2 Feature Engineering (`src/features/`)

#### Вычислительные ошибки
- **Division by zero**: Заполнение NaN с предупреждением
- **Invalid indicator parameters**: Валидация перед расчётом, fail fast
- **Insufficient data**: Skip индикатора с минимальной длиной окна
- **Numerical overflow**: Clipping с логированием

#### Кэширование
- **Cache corruption**: Автоматический пересчёт с удалением повреждённого кэша
- **Cache version mismatch**: Инвалидация и пересчёт
- **Disk full**: Очистка старых кэшей (LRU policy)

### 2.3 Labeling (`src/labeling/`)

#### Ошибки разметки
- **Insufficient bars for horizon**: Skip с предупреждением
- **Ambiguous labels**: Логирование для ручного анализа
- **Class imbalance critical**: Предупреждение + рекомендации по балансировке

### 2.4 Model Training (`src/modeling/`)

#### Ошибки обучения
- **CUDA OOM**: Автоматическое уменьшение batch size или fallback на CPU
- **NaN in loss**: Остановка с детальной диагностикой (градиенты, веса)
- **No improvement**: Early stopping с сохранением лучшего чекпоинта
- **Exploding gradients**: Gradient clipping с логированием

**Recovery strategy**:
1. Сохранение чекпоинта каждые N эпох
2. При сбое: загрузка последнего валидного чекпоинта
3. Продолжение с уменьшенным learning rate
4. После 3 сбоев: переход на стабильную конфигурацию

#### Ошибки hyperparameter search
- **Trial failure**: Продолжение поиска, маркировка failed trial
- **Budget exhausted**: Возврат лучшей найденной конфигурации
- **Optuna study corruption**: Экспорт/импорт из backup
- **No feasible parameters**: Расширение диапазонов поиска

### 2.5 Backtesting (`src/backtesting/`)

#### Ошибки симуляции
- **Lookahead detected**: Немедленная остановка, детальный отчёт
- **Negative balance**: Остановка с указанием причины (комиссии? плечо?)
- **Invalid order**: Skip с логированием, продолжение симуляции
- **Price spike anomaly**: Опциональная остановка или корректировка

### 2.6 GUI (`src/interfaces/gui/`)

#### Frontend ошибки
- **Backend unavailable**: Показ кэшированных данных + offline mode banner
- **Invalid user input**: Client-side validation перед отправкой
- **Long-running operation timeout**: Показ прогресса, возможность отмены
- **WebSocket disconnection**: Автоматическое переподключение с backoff

#### Rendering errors
- **Large dataset visualization**: Pagination или downsampling
- **Chart rendering failure**: Fallback на табличное представление
- **Memory leak**: Автоматическая очистка компонентов при размонтировании

## 3. Централизованная обработка

### 3.1 Exception Handler
```python
class PlatformExceptionHandler:
    """Централизованный обработчик исключений"""

    @staticmethod
    def handle(exception, context, retry_policy=None):
        # 1. Классификация ошибки
        # 2. Логирование с полным контекстом
        # 3. Применение retry policy
        # 4. Уведомление через alerts
        # 5. Graceful degradation или fail
        pass
```

### 3.2 Retry Decorator
```python
@retry(
    exceptions=(NetworkError, TemporaryError),
    tries=3,
    delay=1,
    backoff=2,
    jitter=(0, 1),
    on_retry=log_retry_attempt
)
def fetch_data(...):
    pass
```

### 3.3 Circuit Breaker
Для защиты от каскадных сбоев при обращении к внешним сервисам:
- **Closed**: Нормальная работа
- **Open**: Быстрый fail без запросов (после N последовательных сбоев)
- **Half-Open**: Тестовые запросы для проверки восстановления

Параметры:
- Failure threshold: 5 последовательных ошибок
- Timeout: 60 секунд в open state
- Success threshold: 2 успешных запроса для закрытия

## 4. Graceful Degradation

### 4.1 Режимы работы

#### Full Mode
Все компоненты доступны, полная функциональность.

#### Degraded Mode
- API недоступен → работа с локальными данными
- GPU недоступен → fallback на CPU
- Feature store недоступен → on-the-fly расчёт признаков
- MLflow недоступен → локальное логирование

#### Offline Mode
- Нет сетевого подключения
- Только локальные операции
- Синхронизация при восстановлении соединения

### 4.2 Приоритизация задач
При нехватке ресурсов:
1. **High**: Критичные операции (завершение обучения, сохранение результатов)
2. **Medium**: Плановые задачи (scheduled backtests)
3. **Low**: Вспомогательные операции (чистка кэша, агрегация логов)

## 5. Мониторинг и алертинг ошибок

### 5.1 Метрики
- Количество ошибок по типам (за час/день)
- Success rate операций
- Средняя длительность recovery
- Частота срабатывания circuit breaker

### 5.2 Алерты
- **Critical**: Сбой обучения, потеря данных, OOM
- **Warning**: Частые retry, degraded mode > 1 час
- **Info**: Успешное восстановление, переключение режимов

### 5.3 Error Budget
Допустимые уровни ошибок:
- Data ingestion: 99.9% success rate
- Model training: 95% success rate (с учётом экспериментов)
- Backtesting: 99% success rate

## 6. Обработка прерываний

### 6.1 Graceful Shutdown
При получении SIGTERM/SIGINT:
1. Завершение текущей итерации/батча
2. Сохранение чекпоинта
3. Flush логов и метрик
4. Закрытие соединений
5. Cleanup временных файлов

Timeout: 30 секунд, после чего SIGKILL.

### 6.2 Автоматическое восстановление
При перезапуске после сбоя:
1. Проверка integrity последних чекпоинтов
2. Загрузка состояния из последнего валидного
3. Определение незавершённых операций
4. Опциональное продолжение или skip

## 7. Диагностика и отладка

### 7.1 Error Context
Каждая ошибка должна содержать:
- Timestamp
- Exception type и message
- Full stack trace
- Execution context (функция, модуль, параметры)
- System state (память, GPU, диск)
- Correlation ID для трассировки

### 7.2 Post-Mortem Analysis
Автоматическая генерация отчёта при критичных ошибках:
- Хронология событий перед сбоем
- Состояние системы
- Логи за последние N минут
- Дамп переменных (если доступен)

### 7.3 Sentry Integration
Опциональная интеграция с Sentry для:
- Автоматического сбора исключений
- Группировки похожих ошибок
- Release tracking
- Performance monitoring

## 8. Тестирование отказоустойчивости

### 8.1 Chaos Engineering
Периодические тесты с инжекцией ошибок:
- Случайное отключение API
- Симуляция network latency/timeouts
- Ограничение памяти/CPU
- Corruption данных в чекпоинтах

### 8.2 Regression Tests
Набор тестов для проверки recovery сценариев:
- Восстановление после OOM
- Продолжение обучения с чекпоинта
- Работа в degraded mode
- Корректность retry logic

## 9. Документация ошибок

### 9.1 Error Catalog
`docs/errors/catalog.md` — справочник всех типов ошибок:
- Код ошибки
- Описание
- Причины возникновения
- Способы устранения
- Примеры

### 9.2 Troubleshooting Guide
`docs/errors/troubleshooting.md` — пошаговые инструкции для типовых проблем.

### 9.3 Known Issues
`docs/errors/known_issues.md` — список известных проблем с workaround'ами.
