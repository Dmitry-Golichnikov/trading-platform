"""Monitoring hooks and event system."""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class MonitoringEvent:
    """Monitoring event container."""

    def __init__(
        self,
        event_type: str,
        source: str,
        data: Dict[str, Any],
        severity: str = "info",
        timestamp: Optional[datetime] = None,
    ):
        """
        Initialize monitoring event.

        Args:
            event_type: Type of event (e.g., 'experiment_start', 'training_complete').
            source: Source of event (e.g., component name).
            data: Event data.
            severity: Event severity ('debug', 'info', 'warning', 'error', 'critical').
            timestamp: Event timestamp.
        """
        self.event_type = event_type
        self.source = source
        self.data = data
        self.severity = severity
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type,
            "source": self.source,
            "data": self.data,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:
        return f"MonitoringEvent(type={self.event_type}, " f"source={self.source}, severity={self.severity})"


class BaseMonitoringHook(ABC):
    """Base class for monitoring hooks."""

    @abstractmethod
    def on_event(self, event: MonitoringEvent):
        """Handle monitoring event."""
        pass


class LoggingHook(BaseMonitoringHook):
    """Hook that logs events to logger."""

    def __init__(self, logger_name: Optional[str] = None):
        """
        Initialize logging hook.

        Args:
            logger_name: Logger name. If None, uses default logger.
        """
        self.logger = logging.getLogger(logger_name or __name__)

    def on_event(self, event: MonitoringEvent):
        """Log event."""
        log_level = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }.get(event.severity, logging.INFO)

        self.logger.log(
            log_level,
            f"[{event.source}] {event.event_type}: {event.data}",
        )


class FileHook(BaseMonitoringHook):
    """Hook that writes events to file."""

    def __init__(
        self,
        file_path: Union[str, Path],
        format: str = "json",
    ):
        """
        Initialize file hook.

        Args:
            file_path: Path to output file.
            format: Output format ('json' or 'text').
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.format = format

    def on_event(self, event: MonitoringEvent):
        """Write event to file."""
        try:
            with open(self.file_path, "a") as f:
                if self.format == "json":
                    f.write(json.dumps(event.to_dict()) + "\n")
                else:
                    f.write(
                        f"[{event.timestamp.isoformat()}] "
                        f"{event.severity.upper()} - "
                        f"{event.source} - {event.event_type}: "
                        f"{event.data}\n"
                    )

        except Exception as e:
            logger.error(f"Failed to write event to file: {e}")


class MetricsHook(BaseMonitoringHook):
    """Hook that collects metrics."""

    def __init__(self):
        """Initialize metrics hook."""
        self.metrics: Dict[str, List[float]] = {}
        self.counters: Dict[str, int] = {}
        self.latest_values: Dict[str, Any] = {}

    def on_event(self, event: MonitoringEvent):
        """Collect metrics from event."""
        # Count events by type
        counter_key = f"event_{event.event_type}"
        self.counters[counter_key] = self.counters.get(counter_key, 0) + 1

        # Extract numeric metrics
        for key, value in event.data.items():
            if isinstance(value, (int, float)):
                metric_key = f"{event.source}_{key}"

                if metric_key not in self.metrics:
                    self.metrics[metric_key] = []

                self.metrics[metric_key].append(value)
                self.latest_values[metric_key] = value

    def get_metric(self, name: str) -> List[float]:
        """Get metric values."""
        return self.metrics.get(name, [])

    def get_counter(self, name: str) -> int:
        """Get counter value."""
        return self.counters.get(name, 0)

    def get_latest(self, name: str) -> Any:
        """Get latest value."""
        return self.latest_values.get(name)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "metrics": self.metrics,
            "counters": self.counters,
            "latest": self.latest_values,
        }


class CallbackHook(BaseMonitoringHook):
    """Hook that calls custom callback function."""

    def __init__(self, callback: Callable[[MonitoringEvent], None]):
        """
        Initialize callback hook.

        Args:
            callback: Callback function that takes MonitoringEvent.
        """
        self.callback = callback

    def on_event(self, event: MonitoringEvent):
        """Call callback with event."""
        try:
            self.callback(event)
        except Exception as e:
            logger.error(f"Callback hook error: {e}")


class MLflowHook(BaseMonitoringHook):
    """Hook that logs events to MLflow."""

    def __init__(self, mlflow_manager=None):
        """
        Initialize MLflow hook.

        Args:
            mlflow_manager: MLflowManager instance.
        """
        self.mlflow_manager = mlflow_manager

    def on_event(self, event: MonitoringEvent):
        """Log event to MLflow."""
        if not self.mlflow_manager or not self.mlflow_manager.mlflow_available:
            return

        try:
            # Log metrics if present
            numeric_data = {k: v for k, v in event.data.items() if isinstance(v, (int, float))}

            if numeric_data:
                self.mlflow_manager.log_metrics(numeric_data)

            # Log event as tag or parameter
            self.mlflow_manager.set_tag(
                f"event_{event.event_type}",
                event.timestamp.isoformat(),
            )

        except Exception as e:
            logger.error(f"MLflow hook error: {e}")


class MonitoringHookManager:
    """
    Manager for monitoring hooks.

    Coordinates multiple hooks and event dispatching.
    """

    def __init__(self):
        """Initialize monitoring hook manager."""
        self.hooks: List[BaseMonitoringHook] = []
        self.event_history: List[MonitoringEvent] = []
        self.max_history_size = 1000

    def add_hook(self, hook: BaseMonitoringHook):
        """
        Add monitoring hook.

        Args:
            hook: Monitoring hook instance.
        """
        self.hooks.append(hook)
        logger.debug(f"Added hook: {type(hook).__name__}")

    def remove_hook(self, hook: BaseMonitoringHook):
        """
        Remove monitoring hook.

        Args:
            hook: Monitoring hook instance.
        """
        if hook in self.hooks:
            self.hooks.remove(hook)
            logger.debug(f"Removed hook: {type(hook).__name__}")

    def emit(
        self,
        event_type: str,
        source: str,
        data: Dict[str, Any],
        severity: str = "info",
    ):
        """
        Emit monitoring event.

        Args:
            event_type: Event type.
            source: Event source.
            data: Event data.
            severity: Event severity.
        """
        event = MonitoringEvent(
            event_type=event_type,
            source=source,
            data=data,
            severity=severity,
        )

        # Store in history
        self.event_history.append(event)

        # Limit history size
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size :]

        # Dispatch to hooks
        for hook in self.hooks:
            try:
                hook.on_event(event)
            except Exception as e:
                logger.error(f"Hook error ({type(hook).__name__}): {e}")

    # ========== Convenience Methods for Common Events ==========

    def on_experiment_start(self, experiment_id: str, config: Dict[str, Any]):
        """Emit experiment start event."""
        self.emit(
            event_type="experiment_start",
            source="experiment_manager",
            data={
                "experiment_id": experiment_id,
                "config": config,
            },
            severity="info",
        )

    def on_experiment_end(
        self,
        experiment_id: str,
        result: Dict[str, Any],
        success: bool = True,
    ):
        """Emit experiment end event."""
        self.emit(
            event_type="experiment_end",
            source="experiment_manager",
            data={
                "experiment_id": experiment_id,
                "result": result,
                "success": success,
            },
            severity="info" if success else "error",
        )

    def on_training_start(self, model_name: str, config: Dict[str, Any]):
        """Emit training start event."""
        self.emit(
            event_type="training_start",
            source="training",
            data={
                "model_name": model_name,
                "config": config,
            },
            severity="info",
        )

    def on_training_epoch(
        self,
        model_name: str,
        epoch: int,
        metrics: Dict[str, float],
    ):
        """Emit training epoch event."""
        self.emit(
            event_type="training_epoch",
            source="training",
            data={
                "model_name": model_name,
                "epoch": epoch,
                **metrics,
            },
            severity="debug",
        )

    def on_training_complete(
        self,
        model_name: str,
        metrics: Dict[str, float],
        duration: float,
    ):
        """Emit training complete event."""
        self.emit(
            event_type="training_complete",
            source="training",
            data={
                "model_name": model_name,
                "duration": duration,
                **metrics,
            },
            severity="info",
        )

    def on_backtest_start(self, strategy_name: str, config: Dict[str, Any]):
        """Emit backtest start event."""
        self.emit(
            event_type="backtest_start",
            source="backtest",
            data={
                "strategy_name": strategy_name,
                "config": config,
            },
            severity="info",
        )

    def on_backtest_complete(
        self,
        strategy_name: str,
        metrics: Dict[str, float],
        duration: float,
    ):
        """Emit backtest complete event."""
        self.emit(
            event_type="backtest_complete",
            source="backtest",
            data={
                "strategy_name": strategy_name,
                "duration": duration,
                **metrics,
            },
            severity="info",
        )

    def on_error(
        self,
        component: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Emit error event."""
        self.emit(
            event_type="error",
            source=component,
            data={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {},
            },
            severity="error",
        )

    def on_warning(self, component: str, message: str, data: Optional[Dict] = None):
        """Emit warning event."""
        self.emit(
            event_type="warning",
            source=component,
            data={
                "message": message,
                **(data or {}),
            },
            severity="warning",
        )

    def on_metric_update(self, source: str, metrics: Dict[str, float]):
        """Emit metric update event."""
        self.emit(
            event_type="metric_update",
            source=source,
            data=metrics,
            severity="debug",
        )

    # ========== Query Methods ==========

    def get_events(
        self,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        severity: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[MonitoringEvent]:
        """
        Get filtered events.

        Args:
            event_type: Filter by event type.
            source: Filter by source.
            severity: Filter by severity.
            limit: Maximum number of events to return.

        Returns:
            List of events.
        """
        events = self.event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if source:
            events = [e for e in events if e.source == source]

        if severity:
            events = [e for e in events if e.severity == severity]

        if limit:
            events = events[-limit:]

        return events

    def get_event_summary(self) -> Dict[str, Any]:
        """Get summary of events."""
        event_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {}

        for event in self.event_history:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1

        return {
            "total_events": len(self.event_history),
            "event_counts": event_counts,
            "severity_counts": severity_counts,
        }

    def clear_history(self):
        """Clear event history."""
        self.event_history.clear()
        logger.info("Cleared event history")


# Global monitoring hook manager instance
_global_monitoring_manager: Optional[MonitoringHookManager] = None


def get_monitoring_manager() -> MonitoringHookManager:
    """Get global monitoring manager instance."""
    global _global_monitoring_manager

    if _global_monitoring_manager is None:
        _global_monitoring_manager = MonitoringHookManager()

        # Add default logging hook
        _global_monitoring_manager.add_hook(LoggingHook())

    return _global_monitoring_manager


def emit_event(
    event_type: str,
    source: str,
    data: Dict[str, Any],
    severity: str = "info",
):
    """
    Convenience function to emit event to global manager.

    Args:
        event_type: Event type.
        source: Event source.
        data: Event data.
        severity: Event severity.
    """
    manager = get_monitoring_manager()
    manager.emit(event_type, source, data, severity)
