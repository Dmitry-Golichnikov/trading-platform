"""Модуль оркестрации."""

from src.orchestration.executor import Task, TaskExecutor, TaskStatus
from src.orchestration.experiment_manager import ExperimentManager
from src.orchestration.experiment_tracker import ExperimentTracker
from src.orchestration.mlflow_integration import MLflowManager
from src.orchestration.monitoring import (
    BaseMonitoringHook,
    CallbackHook,
    FileHook,
    LoggingHook,
    MetricsHook,
    MLflowHook,
    MonitoringEvent,
    MonitoringHookManager,
    emit_event,
    get_monitoring_manager,
)
from src.orchestration.scheduler import (
    CronSchedule,
    ScheduledTask,
    TaskScheduler,
)
from src.orchestration.task_queue import (
    LocalTaskQueue,
    RedisTaskQueue,
    TaskMessage,
    TaskQueueManager,
)

__all__ = [
    # Experiment management
    "ExperimentManager",
    "ExperimentTracker",
    "MLflowManager",
    # Task execution
    "Task",
    "TaskExecutor",
    "TaskStatus",
    # Scheduling
    "CronSchedule",
    "ScheduledTask",
    "TaskScheduler",
    # Task queue
    "TaskMessage",
    "LocalTaskQueue",
    "RedisTaskQueue",
    "TaskQueueManager",
    # Monitoring
    "MonitoringEvent",
    "BaseMonitoringHook",
    "LoggingHook",
    "FileHook",
    "MetricsHook",
    "CallbackHook",
    "MLflowHook",
    "MonitoringHookManager",
    "get_monitoring_manager",
    "emit_event",
]
