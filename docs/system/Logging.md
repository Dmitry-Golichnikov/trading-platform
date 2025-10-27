# Логирование

## 1. Принципы логирования

### 1.1 Цели
- **Debugging**: Помощь в отладке и диагностике проблем
- **Auditing**: Отслеживание действий пользователей и системы
- **Monitoring**: Данные для мониторинга и алертов
- **Analytics**: Анализ поведения системы и оптимизация

### 1.2 Философия
- **Structured Logging**: Структурированные логи (JSON) для автоматической обработки
- **Context-Rich**: Максимум контекста для понимания ситуации
- **Performance-Aware**: Минимальное влияние на производительность
- **Secure**: Не логировать чувствительные данные (пароли, токены)

## 2. Уровни логирования

### 2.1 Standard Levels

```python
import logging

# DEBUG: Детальная информация для диагностики
logger.debug("Calculating SMA with window=20, data shape=%s", data.shape)

# INFO: Подтверждение нормальной работы
logger.info("Data ingestion completed: %d rows loaded", len(df))

# WARNING: Неожиданное событие, но система работает
logger.warning("Missing values detected: %d rows affected", missing_count)

# ERROR: Ошибка, некоторая функциональность не работает
logger.error("Failed to load model from %s: %s", model_path, exc)

# CRITICAL: Серьёзная ошибка, система может быть неработоспособна
logger.critical("Out of memory, terminating process")
```

### 2.2 Когда использовать

| Уровень | Использование | Примеры |
|---------|--------------|---------|
| DEBUG | Детальная диагностика разработки | Значения переменных, промежуточные результаты |
| INFO | Важные бизнес-события | Запуск пайплайна, завершение обучения |
| WARNING | Неожиданные, но обрабатываемые ситуации | Missing values, API retry, кэш промах |
| ERROR | Ошибки, требующие внимания | Failed API call, файл не найден |
| CRITICAL | Системные сбои | OOM, диск заполнен, corrupt data |

## 3. Структура логов

### 3.1 Structured Logging (JSON)

```python
import structlog

# Конфигурация structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Использование
logger.info(
    "model_training_completed",
    model_type="lightgbm",
    dataset_size=50000,
    epochs=100,
    final_loss=0.234,
    duration_seconds=3600
)
```

Результат (JSON):
```json
{
  "timestamp": "2025-01-26T14:30:22.123456Z",
  "level": "info",
  "logger": "src.modeling.trainer",
  "event": "model_training_completed",
  "model_type": "lightgbm",
  "dataset_size": 50000,
  "epochs": 100,
  "final_loss": 0.234,
  "duration_seconds": 3600,
  "pid": 12345,
  "thread": "MainThread"
}
```

### 3.2 Обязательные поля

```python
# Каждый лог должен содержать:
log_entry = {
    "timestamp": "2025-01-26T14:30:22.123456Z",  # ISO 8601
    "level": "info",                              # debug/info/warning/error/critical
    "logger": "module.name",                      # Источник лога
    "event": "event_name",                        # Название события
    "message": "Human readable message",          # Читаемое сообщение

    # Опционально но рекомендуется:
    "correlation_id": "abc-123",                  # Для трассировки
    "user_id": "user@example.com",                # Если применимо
    "session_id": "sess_xyz",                     # Сессия
    "request_id": "req_456",                      # ID запроса
    "pid": 12345,                                 # Process ID
    "thread": "Thread-1",                         # Thread name
    "hostname": "workstation-01",                 # Хост
    "environment": "development",                 # dev/staging/production
}
```

### 3.3 Context Management

```python
from contextvars import ContextVar

# Контекстные переменные для трассировки
correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default=None)
request_id_var: ContextVar[str] = ContextVar('request_id', default=None)

class LogContext:
    """Менеджер контекста для логирования"""

    def __init__(self, **context):
        self.context = context
        self.logger = structlog.get_logger()

    def __enter__(self):
        # Добавить контекст к логгеру
        self.logger = self.logger.bind(**self.context)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Очистить контекст
        pass

# Использование
with LogContext(pipeline_id="pipe_001", stage="feature_engineering"):
    logger.info("Started feature calculation")
    # Все логи в этом блоке будут иметь pipeline_id и stage
    calculate_features()
    logger.info("Completed feature calculation")
```

## 4. Конфигурация

### 4.1 Logging Config

```yaml
# configs/logging.yaml
version: 1
disable_existing_loggers: false

formatters:
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: "%(timestamp)s %(level)s %(name)s %(message)s"

  simple:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

  detailed:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: simple
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: json
    filename: logs/application.log
    maxBytes: 104857600  # 100MB
    backupCount: 10

  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: logs/errors.log
    maxBytes: 104857600
    backupCount: 10

  syslog:
    class: logging.handlers.SysLogHandler
    level: WARNING
    formatter: json
    address: /dev/log  # или ('localhost', 514)

loggers:
  # Корневой логгер
  root:
    level: INFO
    handlers: [console, file, error_file]

  # Модули платформы
  src.data:
    level: DEBUG
    handlers: [file]
    propagate: false

  src.modeling:
    level: DEBUG
    handlers: [file]
    propagate: false

  src.backtesting:
    level: INFO
    handlers: [file]
    propagate: false

  # Сторонние библиотеки (снизить шум)
  urllib3:
    level: WARNING

  matplotlib:
    level: WARNING

  numba:
    level: WARNING

# Специальные параметры
log_level_env_var: LOG_LEVEL  # Переменная окружения для override
log_dir: logs/
max_log_file_size_mb: 100
backup_count: 10
```

### 4.2 Python Setup

```python
# src/common/logging.py
import logging
import logging.config
import yaml
from pathlib import Path

def setup_logging(config_path: Path = None, default_level: str = "INFO"):
    """
    Настроить логирование для приложения

    Args:
        config_path: Путь к YAML конфигу
        default_level: Уровень по умолчанию если конфиг не найден
    """
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Создать директорию для логов
        log_dir = Path(config.get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)

        # Применить конфигурацию
        logging.config.dictConfig(config)
    else:
        # Fallback: базовая конфигурация
        logging.basicConfig(
            level=getattr(logging, default_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/application.log')
            ]
        )

    # Override уровня через env var
    env_level = os.getenv('LOG_LEVEL')
    if env_level:
        logging.getLogger().setLevel(getattr(logging, env_level.upper()))

    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")
```

## 5. Rotation и Archival

### 5.1 Rotation Strategies

#### Time-based Rotation
```python
from logging.handlers import TimedRotatingFileHandler

handler = TimedRotatingFileHandler(
    filename='logs/application.log',
    when='midnight',      # 'S', 'M', 'H', 'D', 'midnight', 'W0'-'W6'
    interval=1,           # Каждые N единиц
    backupCount=30,       # Хранить 30 дней
    encoding='utf-8'
)
```

#### Size-based Rotation
```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    filename='logs/application.log',
    maxBytes=100 * 1024 * 1024,  # 100MB
    backupCount=10,               # Хранить 10 файлов
    encoding='utf-8'
)
```

### 5.2 Compression

```python
import gzip
import shutil
from pathlib import Path

class CompressingRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler с автоматическим сжатием"""

    def doRollover(self):
        """Override rollover для добавления сжатия"""
        super().doRollover()

        # Сжать предыдущий лог
        for i in range(self.backupCount, 0, -1):
            sfn = f"{self.baseFilename}.{i}"
            if Path(sfn).exists() and not sfn.endswith('.gz'):
                self._compress_file(sfn)

    def _compress_file(self, filepath: str):
        """Сжать лог-файл"""
        with open(filepath, 'rb') as f_in:
            with gzip.open(f"{filepath}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Удалить оригинал
        Path(filepath).unlink()
```

### 5.3 Archival Policy

```yaml
# configs/log_retention.yaml
retention_policy:
  # Локальное хранение
  local:
    debug_logs:
      days: 7
      compression: true

    info_logs:
      days: 30
      compression: true

    error_logs:
      days: 90
      compression: true

  # Архив (S3/MinIO)
  archive:
    enabled: true
    bucket: "logs-archive"
    path_template: "logs/{year}/{month}/{day}/"

    # Переместить в архив после N дней
    archive_after_days:
      debug: 7
      info: 30
      error: never  # Ошибки хранить локально

    # Удалить из архива после N дней
    delete_after_days:
      debug: 90
      info: 365
      error: 730  # 2 года
```

```python
class LogArchiver:
    """Архивирование старых логов"""

    def __init__(self, config: dict, s3_client):
        self.config = config
        self.s3 = s3_client

    def archive_old_logs(self):
        """Архивировать логи согласно политике"""
        log_dir = Path('logs/')

        for log_file in log_dir.glob('*.log.*'):
            # Определить возраст файла
            age_days = (datetime.now() - datetime.fromtimestamp(
                log_file.stat().st_mtime
            )).days

            # Определить тип лога (debug/info/error)
            log_level = self._extract_log_level(log_file)

            archive_threshold = self.config['archive']['archive_after_days'].get(
                log_level, 30
            )

            if age_days >= archive_threshold:
                self._archive_to_s3(log_file)
                log_file.unlink()  # Удалить локально

    def _archive_to_s3(self, log_file: Path):
        """Загрузить лог в S3"""
        date = datetime.fromtimestamp(log_file.stat().st_mtime)
        s3_path = self.config['archive']['path_template'].format(
            year=date.year,
            month=date.month,
            day=date.day
        ) + log_file.name

        self.s3.upload_file(
            str(log_file),
            self.config['archive']['bucket'],
            s3_path
        )
        logger.info(f"Archived {log_file} to s3://{s3_path}")
```

## 6. Специальные случаи

### 6.1 Performance Logging

```python
import functools
import time

def log_execution_time(func):
    """Декоратор для логирования времени выполнения"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start = time.time()

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start

            logger.info(
                "function_executed",
                function=func.__name__,
                duration_seconds=duration,
                status="success"
            )

            return result

        except Exception as e:
            duration = time.time() - start
            logger.error(
                "function_failed",
                function=func.__name__,
                duration_seconds=duration,
                error=str(e),
                status="failed"
            )
            raise

    return wrapper

# Использование
@log_execution_time
def expensive_operation():
    # ... долгая операция
    pass
```

### 6.2 Exception Logging

```python
def log_exception(logger, exc: Exception, context: dict = None):
    """
    Детальное логирование исключения

    Args:
        logger: Logger instance
        exc: Исключение
        context: Дополнительный контекст
    """
    import traceback
    import sys

    exc_info = {
        'exception_type': type(exc).__name__,
        'exception_message': str(exc),
        'exception_args': exc.args,
        'traceback': traceback.format_exc(),
        'stack_trace': traceback.extract_tb(sys.exc_info()[2])
    }

    if context:
        exc_info['context'] = context

    logger.error(
        "exception_occurred",
        **exc_info
    )

# Использование
try:
    risky_operation()
except Exception as e:
    log_exception(
        logger,
        e,
        context={
            'operation': 'data_ingestion',
            'ticker': 'SBER',
            'date': '2025-01-26'
        }
    )
    raise
```

### 6.3 Security & PII

```python
import re

class SensitiveDataFilter(logging.Filter):
    """Фильтр для маскирования чувствительных данных"""

    PATTERNS = [
        # API tokens
        (re.compile(r't\.[A-Za-z0-9_-]{50,}'), 't.***REDACTED***'),

        # Пароли
        (re.compile(r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.I), 'password=***'),

        # Email
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '***@***.***'),

        # Номера карт
        (re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'), '****-****-****-****'),
    ]

    def filter(self, record):
        """Замаскировать чувствительные данные в сообщении"""
        if isinstance(record.msg, str):
            for pattern, replacement in self.PATTERNS:
                record.msg = pattern.sub(replacement, record.msg)

        # Также обработать args
        if record.args:
            record.args = tuple(
                self._redact_value(arg) for arg in record.args
            )

        return True

    def _redact_value(self, value):
        """Замаскировать значение"""
        if isinstance(value, str):
            for pattern, replacement in self.PATTERNS:
                value = pattern.sub(replacement, value)
        return value

# Добавить фильтр ко всем handlers
for handler in logging.root.handlers:
    handler.addFilter(SensitiveDataFilter())
```

### 6.4 Audit Logging

```python
class AuditLogger:
    """Специализированный логгер для аудита действий"""

    def __init__(self):
        self.logger = logging.getLogger('audit')

        # Отдельный handler для аудит-логов
        handler = RotatingFileHandler(
            'logs/audit.log',
            maxBytes=100*1024*1024,
            backupCount=100  # Хранить дольше
        )
        handler.setFormatter(
            logging.Formatter('%(message)s')  # Только JSON
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_action(
        self,
        action: str,
        user: str = None,
        resource: str = None,
        result: str = "success",
        **kwargs
    ):
        """
        Логировать действие пользователя/системы

        Args:
            action: Название действия
            user: Пользователь (или 'system')
            resource: Затронутый ресурс
            result: success/failure
            **kwargs: Дополнительные параметры
        """
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'user': user or 'system',
            'resource': resource,
            'result': result,
            **kwargs
        }

        self.logger.info(json.dumps(audit_entry))

# Использование
audit_logger = AuditLogger()

audit_logger.log_action(
    action='model_trained',
    user='system',
    resource='model_lightgbm_v1',
    model_type='lightgbm',
    dataset_size=50000,
    final_accuracy=0.856
)

audit_logger.log_action(
    action='config_changed',
    user='admin',
    resource='configs/training.yaml',
    changes={'learning_rate': '0.001 -> 0.0005'}
)
```

## 7. Monitoring Logs

### 7.1 Log Analysis

```python
import json
from collections import Counter, defaultdict

class LogAnalyzer:
    """Анализ логов для метрик и статистики"""

    def __init__(self, log_file: Path):
        self.log_file = log_file

    def analyze_errors(self, hours: int = 24) -> dict:
        """Проанализировать ошибки за последние N часов"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        error_counts = Counter()
        error_examples = defaultdict(list)

        with open(self.log_file) as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    timestamp = datetime.fromisoformat(log_entry['timestamp'])

                    if timestamp < cutoff_time:
                        continue

                    if log_entry['level'] in ['error', 'critical']:
                        error_type = log_entry.get('event', 'unknown')
                        error_counts[error_type] += 1

                        if len(error_examples[error_type]) < 5:
                            error_examples[error_type].append({
                                'timestamp': log_entry['timestamp'],
                                'message': log_entry.get('message', '')
                            })

                except json.JSONDecodeError:
                    continue

        return {
            'total_errors': sum(error_counts.values()),
            'error_counts': dict(error_counts.most_common(10)),
            'examples': dict(error_examples)
        }

    def get_performance_metrics(self) -> dict:
        """Извлечь метрики производительности из логов"""
        durations = defaultdict(list)

        with open(self.log_file) as f:
            for line in f:
                try:
                    log_entry = json.loads(line)

                    if 'duration_seconds' in log_entry:
                        function = log_entry.get('function', 'unknown')
                        durations[function].append(log_entry['duration_seconds'])

                except json.JSONDecodeError:
                    continue

        # Рассчитать статистики
        metrics = {}
        for function, times in durations.items():
            metrics[function] = {
                'count': len(times),
                'mean': np.mean(times),
                'median': np.median(times),
                'p95': np.percentile(times, 95),
                'p99': np.percentile(times, 99),
                'max': max(times)
            }

        return metrics
```

### 7.2 ELK Stack Integration

```yaml
# filebeat.yml - для отправки логов в Elasticsearch
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /app/logs/*.log
    json.keys_under_root: true
    json.add_error_key: true

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "trading-platform-%{+yyyy.MM.dd}"

# Kibana dashboard потом можно настроить для визуализации
```

## 8. Best Practices

### 8.1 Рекомендации
- ✅ Использовать структурированное логирование (JSON)
- ✅ Добавлять контекст к каждому логу (correlation ID, user, etc)
- ✅ Логировать на правильном уровне
- ✅ Не логировать чувствительные данные
- ✅ Использовать rotation для предотвращения переполнения диска
- ✅ Регулярно анализировать логи на предмет паттернов ошибок
- ✅ Настроить централизованный сбор логов (ELK, Loki)

### 8.2 Антипаттерны
- ❌ Чрезмерное логирование на DEBUG в production
- ❌ Логирование без контекста ("Error occurred")
- ❌ Игнорирование ротации (диск заполняется)
- ❌ Логирование паролей, токенов, PII
- ❌ Использование print() вместо logger
- ❌ Блокирующие операции в логировании

### 8.3 Performance Tips
```python
# ❌ Плохо: форматирование всегда выполняется
logger.debug(f"Processing {len(data)} rows: {data.head()}")

# ✅ Хорошо: форматирование только если уровень активен
if logger.isEnabledFor(logging.DEBUG):
    logger.debug("Processing %d rows: %s", len(data), data.head())

# ✅ Ещё лучше: lazy evaluation
logger.debug("Processing rows: %s", lambda: expensive_format(data))
```
