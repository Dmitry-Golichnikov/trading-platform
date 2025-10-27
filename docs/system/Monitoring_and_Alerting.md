# Мониторинг и алертинг

## 1. Цели системы мониторинга

### 1.1 Основные задачи
- **Observability**: Понимание состояния системы в любой момент времени
- **Early Detection**: Раннее обнаружение проблем до их критического влияния
- **Performance Tracking**: Отслеживание производительности и оптимизация узких мест
- **Resource Management**: Контроль использования ресурсов (CPU, GPU, RAM, Disk)

### 1.2 Уровни мониторинга
1. **Infrastructure**: Серверы, контейнеры, сеть
2. **Application**: Пайплайны, модели, процессы
3. **Business**: Качество моделей, метрики стратегий, PnL

## 2. Метрики

### 2.1 System Metrics (Infrastructure)

#### Compute Resources
```python
# Примеры метрик
system_metrics = {
    # CPU
    'cpu_usage_percent': 45.2,
    'cpu_usage_per_core': [40.1, 45.2, 50.3, 43.8],
    'cpu_load_average_1m': 2.5,
    'cpu_load_average_5m': 2.3,
    'cpu_load_average_15m': 2.1,

    # Memory
    'memory_total_gb': 32.0,
    'memory_used_gb': 18.5,
    'memory_available_gb': 13.5,
    'memory_usage_percent': 57.8,
    'memory_swap_used_gb': 0.5,

    # GPU (если доступен)
    'gpu_usage_percent': 75.3,
    'gpu_memory_used_gb': 6.2,
    'gpu_memory_total_gb': 8.0,
    'gpu_temperature_celsius': 72,
    'gpu_power_draw_watts': 180,

    # Disk
    'disk_usage_percent': 68.4,
    'disk_free_gb': 250.0,
    'disk_io_read_mb_per_sec': 45.2,
    'disk_io_write_mb_per_sec': 12.3,

    # Network
    'network_bytes_sent_per_sec': 1024000,
    'network_bytes_recv_per_sec': 2048000,
    'network_connections_active': 15
}
```

#### Мониторинг ресурсов
```python
import psutil
import GPUtil

class SystemMonitor:
    """Мониторинг системных ресурсов"""

    def __init__(self):
        self.start_time = time.time()

    def collect_metrics(self) -> dict:
        """Собрать текущие метрики системы"""
        metrics = {
            'timestamp': datetime.now(),
            'uptime_seconds': time.time() - self.start_time
        }

        # CPU
        cpu_percent = psutil.cpu_percent(interval=1, percpu=False)
        cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
        load_avg = psutil.getloadavg()

        metrics.update({
            'cpu_usage_percent': cpu_percent,
            'cpu_usage_per_core': cpu_per_core,
            'cpu_load_average_1m': load_avg[0],
            'cpu_load_average_5m': load_avg[1],
            'cpu_load_average_15m': load_avg[2],
        })

        # Memory
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        metrics.update({
            'memory_total_gb': mem.total / (1024**3),
            'memory_used_gb': mem.used / (1024**3),
            'memory_available_gb': mem.available / (1024**3),
            'memory_usage_percent': mem.percent,
            'memory_swap_used_gb': swap.used / (1024**3),
        })

        # GPU
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Первая GPU
                metrics.update({
                    'gpu_usage_percent': gpu.load * 100,
                    'gpu_memory_used_gb': gpu.memoryUsed / 1024,
                    'gpu_memory_total_gb': gpu.memoryTotal / 1024,
                    'gpu_temperature_celsius': gpu.temperature,
                })
        except Exception as e:
            logger.debug(f"GPU metrics unavailable: {e}")

        # Disk
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()

        metrics.update({
            'disk_usage_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3),
            'disk_io_read_bytes': disk_io.read_bytes,
            'disk_io_write_bytes': disk_io.write_bytes,
        })

        # Network
        net_io = psutil.net_io_counters()
        connections = len(psutil.net_connections())

        metrics.update({
            'network_bytes_sent': net_io.bytes_sent,
            'network_bytes_recv': net_io.bytes_recv,
            'network_connections_active': connections,
        })

        return metrics
```

### 2.2 Application Metrics

#### Pipeline Metrics
```python
pipeline_metrics = {
    # Data Ingestion
    'data_ingestion_rows_per_second': 1250.5,
    'data_ingestion_api_calls_total': 1543,
    'data_ingestion_api_errors_total': 3,
    'data_ingestion_cache_hit_rate': 0.85,

    # Feature Engineering
    'feature_calculation_duration_seconds': 45.2,
    'feature_cache_size_mb': 1250.0,
    'features_computed_total': 156,

    # Model Training
    'training_epoch_duration_seconds': 120.5,
    'training_loss': 0.234,
    'training_batch_size': 64,
    'training_learning_rate': 0.001,
    'training_gradient_norm': 2.34,

    # Backtesting
    'backtest_bars_processed_per_second': 5000,
    'backtest_trades_total': 1523,
    'backtest_pnl': 25432.15,

    # Hyperparameter Search
    'hparam_search_trials_completed': 15,
    'hparam_search_trials_total': 50,
    'hparam_search_best_metric': 0.876,
}
```

#### Pipeline Monitor
```python
class PipelineMonitor:
    """Мониторинг пайплайнов"""

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.metrics = {}
        self.start_time = None

    def start(self):
        """Начать мониторинг"""
        self.start_time = time.time()
        self.metrics = {
            'pipeline_name': self.pipeline_name,
            'status': 'running',
            'start_time': datetime.now(),
        }

    def update(self, **kwargs):
        """Обновить метрики"""
        self.metrics.update(kwargs)
        self.metrics['duration_seconds'] = time.time() - self.start_time

    def complete(self, status: str = 'success'):
        """Завершить мониторинг"""
        self.metrics['status'] = status
        self.metrics['end_time'] = datetime.now()
        self.metrics['duration_seconds'] = time.time() - self.start_time

        # Отправить метрики
        self._send_metrics()

    def _send_metrics(self):
        """Отправить метрики в систему мониторинга"""
        # MLflow, Prometheus, custom backend
        pass
```

### 2.3 Business Metrics

#### Model Performance
```python
model_metrics = {
    # Classification
    'model_accuracy': 0.856,
    'model_precision': 0.832,
    'model_recall': 0.845,
    'model_f1_score': 0.838,
    'model_roc_auc': 0.912,
    'model_log_loss': 0.345,

    # Calibration
    'model_calibration_error': 0.023,
    'model_brier_score': 0.156,

    # Stability
    'model_prediction_drift_psi': 0.045,  # PSI < 0.1 - стабильно
    'model_feature_drift_count': 2,

    # Inference
    'model_inference_latency_ms': 12.5,
    'model_predictions_per_second': 1200,
}
```

#### Strategy Performance
```python
strategy_metrics = {
    # Returns
    'strategy_total_return': 0.2534,
    'strategy_annual_return': 0.1567,
    'strategy_daily_return_mean': 0.0012,
    'strategy_daily_return_std': 0.0156,

    # Risk
    'strategy_sharpe_ratio': 1.45,
    'strategy_sortino_ratio': 2.13,
    'strategy_max_drawdown': -0.1245,
    'strategy_var_95': -0.0234,
    'strategy_cvar_95': -0.0345,

    # Trading
    'strategy_trades_total': 1523,
    'strategy_win_rate': 0.567,
    'strategy_profit_factor': 1.89,
    'strategy_avg_trade_return': 0.0045,
    'strategy_avg_win_size': 0.0234,
    'strategy_avg_loss_size': -0.0156,

    # Costs
    'strategy_commissions_total': 1250.45,
    'strategy_slippage_impact': 0.0023,
}
```

## 3. Alerting System

### 3.1 Alert Levels

```python
class AlertLevel(Enum):
    """Уровни важности алертов"""
    INFO = "info"           # Информационные
    WARNING = "warning"     # Предупреждения
    CRITICAL = "critical"   # Критичные
    EMERGENCY = "emergency" # Требуют немедленного внимания
```

### 3.2 Alert Rules

```yaml
# configs/monitoring/alert_rules.yaml
alert_rules:
  # System Resources
  - name: "high_cpu_usage"
    condition: "cpu_usage_percent > 90"
    duration: 300  # 5 минут
    level: "warning"
    message: "CPU usage above 90% for 5 minutes"
    actions:
      - "send_notification"
      - "log_alert"

  - name: "high_memory_usage"
    condition: "memory_usage_percent > 90"
    duration: 300
    level: "critical"
    message: "Memory usage above 90%"
    actions:
      - "send_notification"
      - "trigger_garbage_collection"

  - name: "disk_space_low"
    condition: "disk_free_gb < 50"
    duration: 0
    level: "warning"
    message: "Less than 50GB disk space remaining"
    actions:
      - "send_notification"
      - "cleanup_old_artifacts"

  - name: "gpu_overheating"
    condition: "gpu_temperature_celsius > 85"
    duration: 60
    level: "critical"
    message: "GPU temperature above 85°C"
    actions:
      - "send_notification"
      - "pause_training"

  # Application
  - name: "training_not_improving"
    condition: "epochs_without_improvement > 20"
    duration: 0
    level: "info"
    message: "Training hasn't improved for 20 epochs"
    actions:
      - "early_stopping"

  - name: "nan_in_loss"
    condition: "training_loss == NaN"
    duration: 0
    level: "critical"
    message: "NaN detected in training loss"
    actions:
      - "stop_training"
      - "send_notification"
      - "save_debug_state"

  - name: "api_rate_limit_hits"
    condition: "api_rate_limit_hits_per_hour > 10"
    duration: 0
    level: "warning"
    message: "API rate limit hit more than 10 times in an hour"
    actions:
      - "send_notification"
      - "increase_backoff_delay"

  # Business
  - name: "model_drift_detected"
    condition: "model_prediction_drift_psi > 0.2"
    duration: 0
    level: "warning"
    message: "Significant model drift detected (PSI > 0.2)"
    actions:
      - "send_notification"
      - "trigger_retraining"

  - name: "strategy_drawdown_high"
    condition: "strategy_max_drawdown < -0.20"
    duration: 0
    level: "critical"
    message: "Strategy drawdown exceeded -20%"
    actions:
      - "send_notification"
      - "pause_trading"

  - name: "backtest_negative_pnl"
    condition: "backtest_pnl < 0"
    duration: 0
    level: "info"
    message: "Backtest resulted in negative PnL"
    actions:
      - "log_alert"
```

### 3.3 Alert Manager

```python
class AlertManager:
    """Управление алертами"""

    def __init__(self, config_path: Path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.rules = config['alert_rules']

        self.active_alerts = {}  # alert_name -> first_triggered_time
        self.alert_history = []
        self.notification_channels = self._setup_channels()

    def check_alerts(self, metrics: dict):
        """Проверить метрики на соответствие правилам алертов"""
        for rule in self.rules:
            should_alert = self._evaluate_condition(rule['condition'], metrics)

            if should_alert:
                self._handle_alert(rule, metrics)
            else:
                # Снять алерт если был активен
                if rule['name'] in self.active_alerts:
                    self._resolve_alert(rule['name'])

    def _evaluate_condition(self, condition: str, metrics: dict) -> bool:
        """Оценить условие алерта"""
        try:
            # Безопасная оценка с ограниченным контекстом
            allowed_names = {
                **metrics,
                'NaN': float('nan'),
                'Inf': float('inf'),
            }
            return eval(condition, {"__builtins__": {}}, allowed_names)
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False

    def _handle_alert(self, rule: dict, metrics: dict):
        """Обработать активацию алерта"""
        alert_name = rule['name']
        current_time = time.time()

        # Проверить duration
        if alert_name in self.active_alerts:
            first_triggered = self.active_alerts[alert_name]
            if current_time - first_triggered < rule['duration']:
                # Ещё не прошло достаточно времени
                return
        else:
            # Первая активация
            self.active_alerts[alert_name] = current_time
            if rule['duration'] > 0:
                # Ждём duration перед отправкой
                return

        # Отправить алерт
        alert = {
            'name': alert_name,
            'level': rule['level'],
            'message': rule['message'],
            'timestamp': datetime.now(),
            'metrics': metrics,
        }

        self._send_alert(alert, rule['actions'])
        self.alert_history.append(alert)

    def _send_alert(self, alert: dict, actions: list):
        """Отправить алерт через каналы"""
        for action in actions:
            if action == 'send_notification':
                for channel in self.notification_channels:
                    channel.send(alert)
            elif action == 'log_alert':
                logger.warning(f"ALERT: {alert['message']}", extra=alert)
            else:
                # Кастомные действия
                self._execute_action(action, alert)

    def _resolve_alert(self, alert_name: str):
        """Снять алерт"""
        if alert_name in self.active_alerts:
            del self.active_alerts[alert_name]
            logger.info(f"Alert resolved: {alert_name}")

    def _setup_channels(self) -> list:
        """Настроить каналы уведомлений"""
        channels = []

        # Telegram
        if os.getenv('TELEGRAM_BOT_TOKEN'):
            channels.append(TelegramNotifier())

        # Email
        if os.getenv('SMTP_HOST'):
            channels.append(EmailNotifier())

        # Slack
        if os.getenv('SLACK_WEBHOOK_URL'):
            channels.append(SlackNotifier())

        # Console (fallback)
        channels.append(ConsoleNotifier())

        return channels

    def _execute_action(self, action: str, alert: dict):
        """Выполнить кастомное действие"""
        # Реализация специфичных действий
        pass
```

### 3.4 Notification Channels

```python
class NotificationChannel(ABC):
    """Базовый класс для канала уведомлений"""

    @abstractmethod
    def send(self, alert: dict):
        pass

class TelegramNotifier(NotificationChannel):
    """Уведомления в Telegram"""

    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')

    def send(self, alert: dict):
        """Отправить сообщение в Telegram"""
        import requests

        emoji = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'critical': '🔴',
            'emergency': '🚨'
        }

        message = f"{emoji[alert['level']]} *{alert['name']}*\n\n{alert['message']}"

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }

        try:
            requests.post(url, data=data, timeout=10)
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

class EmailNotifier(NotificationChannel):
    """Email уведомления"""

    def send(self, alert: dict):
        import smtplib
        from email.mime.text import MIMEText

        msg = MIMEText(alert['message'])
        msg['Subject'] = f"[{alert['level'].upper()}] {alert['name']}"
        msg['From'] = os.getenv('SMTP_FROM')
        msg['To'] = os.getenv('ALERT_EMAIL_TO')

        try:
            with smtplib.SMTP(os.getenv('SMTP_HOST'), int(os.getenv('SMTP_PORT', 587))) as server:
                server.starttls()
                server.login(os.getenv('SMTP_USER'), os.getenv('SMTP_PASSWORD'))
                server.send_message(msg)
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

class SlackNotifier(NotificationChannel):
    """Slack уведомления"""

    def send(self, alert: dict):
        import requests

        webhook_url = os.getenv('SLACK_WEBHOOK_URL')

        color_map = {
            'info': '#36a64f',
            'warning': '#ff9900',
            'critical': '#ff0000',
            'emergency': '#990000'
        }

        payload = {
            'attachments': [{
                'color': color_map[alert['level']],
                'title': alert['name'],
                'text': alert['message'],
                'footer': 'Trading Platform Monitor',
                'ts': int(alert['timestamp'].timestamp())
            }]
        }

        try:
            requests.post(webhook_url, json=payload, timeout=10)
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
```

## 4. Dashboards

### 4.1 Grafana Dashboard

```yaml
# configs/monitoring/grafana_dashboard.yaml
dashboard:
  title: "Trading Platform Monitoring"
  refresh: "10s"

  panels:
    - title: "System Resources"
      type: "graph"
      metrics:
        - cpu_usage_percent
        - memory_usage_percent
        - gpu_usage_percent
      thresholds:
        - value: 80
          color: "yellow"
        - value: 90
          color: "red"

    - title: "Training Progress"
      type: "graph"
      metrics:
        - training_loss
        - val_loss

    - title: "Backtest PnL"
      type: "stat"
      metrics:
        - backtest_pnl
      color_mode: "value"

    - title: "API Rate Limits"
      type: "gauge"
      metrics:
        - api_requests_per_minute
      max: 300
```

### 4.2 Streamlit Dashboard

```python
# src/interfaces/gui/monitoring_dashboard.py
import streamlit as st
import plotly.graph_objects as go

def monitoring_dashboard():
    """Дашборд мониторинга в Streamlit"""

    st.title("🔍 System Monitoring")

    # Обновление каждые 5 секунд
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()

    # System Resources
    st.header("System Resources")

    col1, col2, col3, col4 = st.columns(4)

    metrics = get_current_metrics()

    with col1:
        st.metric(
            "CPU",
            f"{metrics['cpu_usage_percent']:.1f}%",
            delta=f"{metrics['cpu_usage_percent'] - 50:.1f}%"
        )

    with col2:
        st.metric(
            "Memory",
            f"{metrics['memory_usage_percent']:.1f}%",
            delta=f"{metrics['memory_usage_percent'] - 50:.1f}%"
        )

    with col3:
        if 'gpu_usage_percent' in metrics:
            st.metric(
                "GPU",
                f"{metrics['gpu_usage_percent']:.1f}%"
            )

    with col4:
        st.metric(
            "Disk",
            f"{metrics['disk_free_gb']:.1f} GB free"
        )

    # Active Pipelines
    st.header("Active Pipelines")

    pipelines = get_active_pipelines()
    for pipeline in pipelines:
        with st.expander(f"📊 {pipeline['name']} ({pipeline['status']})"):
            st.progress(pipeline['progress'])
            st.json(pipeline['metrics'])

    # Recent Alerts
    st.header("Recent Alerts")

    alerts = get_recent_alerts(limit=10)
    for alert in alerts:
        alert_color = {
            'info': '🔵',
            'warning': '🟡',
            'critical': '🔴'
        }.get(alert['level'], '⚪')

        st.write(f"{alert_color} {alert['timestamp']}: {alert['message']}")

    # Auto-refresh
    time.sleep(5)
    st.rerun()
```

## 5. Metrics Storage

### 5.1 Time Series Database

```python
# Prometheus metrics export
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Счётчики
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'status']
)

training_epochs_total = Counter(
    'training_epochs_total',
    'Total training epochs completed',
    ['model_type']
)

# Gauges (текущие значения)
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('memory_usage_percent', 'Memory usage percentage')
active_pipelines = Gauge('active_pipelines', 'Number of active pipelines')

# Histograms (распределения)
api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method']
)

training_batch_duration = Histogram(
    'training_batch_duration_seconds',
    'Training batch duration'
)

# Запуск HTTP сервера для Prometheus
start_http_server(8000)
```

### 5.2 MLflow Integration

```python
class MLflowMonitor:
    """Интеграция с MLflow для трекинга экспериментов"""

    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)

    def log_system_metrics(self, metrics: dict):
        """Логировать системные метрики"""
        with mlflow.start_run(run_name="system_monitoring", nested=True):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=int(time.time()))

    def log_pipeline_metrics(self, pipeline_name: str, metrics: dict):
        """Логировать метрики пайплайна"""
        with mlflow.start_run(run_name=pipeline_name, nested=True):
            mlflow.log_params({'pipeline': pipeline_name})
            mlflow.log_metrics(metrics)
```

## 6. Health Checks

### 6.1 Service Health

```python
class HealthChecker:
    """Проверка здоровья сервисов"""

    def check_all(self) -> dict:
        """Проверить все компоненты"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now(),
            'components': {}
        }

        # Database
        health['components']['database'] = self._check_database()

        # API
        health['components']['api'] = self._check_api()

        # Storage
        health['components']['storage'] = self._check_storage()

        # GPU
        health['components']['gpu'] = self._check_gpu()

        # Определить общий статус
        if any(c['status'] == 'unhealthy' for c in health['components'].values()):
            health['status'] = 'unhealthy'
        elif any(c['status'] == 'degraded' for c in health['components'].values()):
            health['status'] = 'degraded'

        return health

    def _check_database(self) -> dict:
        """Проверить доступность БД"""
        try:
            # Попытка подключения
            return {'status': 'healthy', 'latency_ms': 5.2}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}

    def _check_api(self) -> dict:
        """Проверить Tinkoff API"""
        try:
            # Тестовый запрос
            return {'status': 'healthy', 'latency_ms': 120}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}

    def _check_storage(self) -> dict:
        """Проверить хранилище"""
        disk = psutil.disk_usage('/')
        if disk.percent > 95:
            return {'status': 'unhealthy', 'reason': 'disk full'}
        elif disk.percent > 90:
            return {'status': 'degraded', 'reason': 'disk almost full'}
        return {'status': 'healthy', 'free_gb': disk.free / (1024**3)}

    def _check_gpu(self) -> dict:
        """Проверить GPU"""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return {'status': 'unavailable'}

            gpu = gpus[0]
            if gpu.temperature > 85:
                return {'status': 'degraded', 'reason': 'overheating'}

            return {'status': 'healthy', 'temperature': gpu.temperature}
        except:
            return {'status': 'unavailable'}
```

## 7. Best Practices

### 7.1 Рекомендации
- ✅ Мониторить все критичные компоненты
- ✅ Устанавливать разумные пороги для алертов
- ✅ Использовать несколько каналов уведомлений
- ✅ Логировать все алерты для анализа
- ✅ Регулярно пересматривать правила алертов
- ✅ Автоматизировать реакцию на типовые проблемы

### 7.2 Антипаттерны
- ❌ Слишком много ложных алертов (alert fatigue)
- ❌ Отсутствие приоритизации алертов
- ❌ Игнорирование warnings до critical
- ❌ Мониторинг без действий
- ❌ Отсутствие документации алертов
