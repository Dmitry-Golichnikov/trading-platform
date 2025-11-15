"""Tests for monitoring system."""

import tempfile
from pathlib import Path

import pytest

from src.orchestration.monitoring import (
    CallbackHook,
    FileHook,
    LoggingHook,
    MetricsHook,
    MonitoringEvent,
    MonitoringHookManager,
)


@pytest.fixture
def temp_file():
    """Create temporary file for file hook."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def hook_manager():
    """Create monitoring hook manager."""
    manager = MonitoringHookManager()
    # Remove default logging hook for tests
    manager.hooks = []
    return manager


def test_monitoring_event_creation():
    """Test creating monitoring event."""
    event = MonitoringEvent(
        event_type="test_event",
        source="test_source",
        data={"key": "value"},
        severity="info",
    )

    assert event.event_type == "test_event"
    assert event.source == "test_source"
    assert event.data["key"] == "value"
    assert event.severity == "info"
    assert event.timestamp is not None


def test_monitoring_event_to_dict():
    """Test converting event to dictionary."""
    event = MonitoringEvent(
        event_type="test_event",
        source="test_source",
        data={"key": "value"},
    )

    event_dict = event.to_dict()

    assert event_dict["event_type"] == "test_event"
    assert event_dict["source"] == "test_source"
    assert event_dict["data"]["key"] == "value"
    assert "timestamp" in event_dict


def test_logging_hook():
    """Test logging hook."""
    hook = LoggingHook()

    event = MonitoringEvent(
        event_type="test_event",
        source="test_source",
        data={"key": "value"},
    )

    # Should not raise error
    hook.on_event(event)


def test_file_hook_json(temp_file):
    """Test file hook with JSON format."""
    hook = FileHook(temp_file, format="json")

    event = MonitoringEvent(
        event_type="test_event",
        source="test_source",
        data={"key": "value"},
    )

    hook.on_event(event)

    # Verify file was written
    assert temp_file.exists()

    # Check content
    content = temp_file.read_text()
    assert "test_event" in content
    assert "test_source" in content


def test_file_hook_text(temp_file):
    """Test file hook with text format."""
    hook = FileHook(temp_file, format="text")

    event = MonitoringEvent(
        event_type="test_event",
        source="test_source",
        data={"key": "value"},
    )

    hook.on_event(event)

    # Verify file was written
    assert temp_file.exists()

    # Check content
    content = temp_file.read_text()
    assert "test_event" in content
    assert "test_source" in content


def test_metrics_hook():
    """Test metrics hook."""
    hook = MetricsHook()

    # Emit events with metrics
    event1 = MonitoringEvent(
        event_type="training",
        source="model",
        data={"loss": 0.5, "accuracy": 0.85},
    )

    event2 = MonitoringEvent(
        event_type="training",
        source="model",
        data={"loss": 0.4, "accuracy": 0.87},
    )

    hook.on_event(event1)
    hook.on_event(event2)

    # Check metrics
    loss_values = hook.get_metric("model_loss")
    assert len(loss_values) == 2
    assert loss_values == [0.5, 0.4]

    accuracy_values = hook.get_metric("model_accuracy")
    assert len(accuracy_values) == 2
    assert accuracy_values == [0.85, 0.87]

    # Check counters
    event_count = hook.get_counter("event_training")
    assert event_count == 2

    # Check latest values
    latest_loss = hook.get_latest("model_loss")
    assert latest_loss == 0.4


def test_callback_hook():
    """Test callback hook."""
    events_received = []

    def callback(event):
        events_received.append(event)

    hook = CallbackHook(callback)

    event = MonitoringEvent(
        event_type="test_event",
        source="test_source",
        data={"key": "value"},
    )

    hook.on_event(event)

    assert len(events_received) == 1
    assert events_received[0] == event


def test_hook_manager_emit(hook_manager):
    """Test emitting events through hook manager."""
    events_received = []

    def callback(event):
        events_received.append(event)

    hook_manager.add_hook(CallbackHook(callback))

    hook_manager.emit(
        event_type="test_event",
        source="test_source",
        data={"key": "value"},
        severity="info",
    )

    assert len(events_received) == 1
    assert events_received[0].event_type == "test_event"


def test_hook_manager_multiple_hooks(hook_manager):
    """Test multiple hooks in manager."""
    events_1 = []
    events_2 = []

    hook_manager.add_hook(CallbackHook(lambda e: events_1.append(e)))
    hook_manager.add_hook(CallbackHook(lambda e: events_2.append(e)))

    hook_manager.emit(
        event_type="test_event",
        source="test_source",
        data={},
    )

    assert len(events_1) == 1
    assert len(events_2) == 1


def test_hook_manager_convenience_methods(hook_manager):
    """Test convenience methods for common events."""
    events = []
    hook_manager.add_hook(CallbackHook(lambda e: events.append(e)))

    # Test various convenience methods
    hook_manager.on_experiment_start("exp_1", {"config": "value"})
    hook_manager.on_experiment_end("exp_1", {"result": "success"})
    hook_manager.on_training_start("model_1", {"lr": 0.001})
    hook_manager.on_training_epoch("model_1", 1, {"loss": 0.5})
    hook_manager.on_training_complete("model_1", {"accuracy": 0.9}, 100.0)
    hook_manager.on_error("component", ValueError("test"), {"context": "test"})
    hook_manager.on_warning("component", "test warning")

    assert len(events) == 7


def test_hook_manager_get_events(hook_manager):
    """Test getting filtered events."""
    hook_manager.emit("event1", "source1", {}, "info")
    hook_manager.emit("event2", "source2", {}, "warning")
    hook_manager.emit("event1", "source1", {}, "error")

    # Filter by event type
    events = hook_manager.get_events(event_type="event1")
    assert len(events) == 2

    # Filter by source
    events = hook_manager.get_events(source="source2")
    assert len(events) == 1

    # Filter by severity
    events = hook_manager.get_events(severity="error")
    assert len(events) == 1


def test_hook_manager_get_summary(hook_manager):
    """Test getting event summary."""
    hook_manager.emit("event1", "source1", {}, "info")
    hook_manager.emit("event1", "source1", {}, "info")
    hook_manager.emit("event2", "source2", {}, "warning")

    summary = hook_manager.get_event_summary()

    assert summary["total_events"] == 3
    assert summary["event_counts"]["event1"] == 2
    assert summary["event_counts"]["event2"] == 1
    assert summary["severity_counts"]["info"] == 2
    assert summary["severity_counts"]["warning"] == 1


def test_hook_manager_clear_history(hook_manager):
    """Test clearing event history."""
    hook_manager.emit("event1", "source1", {}, "info")
    hook_manager.emit("event2", "source2", {}, "info")

    assert len(hook_manager.event_history) == 2

    hook_manager.clear_history()

    assert len(hook_manager.event_history) == 0


def test_hook_manager_remove_hook(hook_manager):
    """Test removing a hook."""
    events = []
    hook = CallbackHook(lambda e: events.append(e))

    hook_manager.add_hook(hook)
    hook_manager.emit("event", "source", {})

    assert len(events) == 1

    hook_manager.remove_hook(hook)
    hook_manager.emit("event", "source", {})

    # Should still be 1 since hook was removed
    assert len(events) == 1
