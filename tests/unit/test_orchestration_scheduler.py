"""Tests for TaskScheduler."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.orchestration import CronSchedule, TaskScheduler


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def scheduler(temp_storage_dir):
    """Create task scheduler."""
    storage_path = temp_storage_dir / "tasks.json"
    return TaskScheduler(storage_path=storage_path, check_interval=1)


def test_cron_schedule_every_minute():
    """Test cron schedule for every minute."""
    schedule = CronSchedule("* * * * *")

    # Should match any time
    assert schedule.should_run(datetime(2024, 1, 1, 12, 0))
    assert schedule.should_run(datetime(2024, 1, 1, 12, 30))


def test_cron_schedule_specific_time():
    """Test cron schedule for specific time."""
    # Every day at 9:00
    schedule = CronSchedule("0 9 * * *")

    assert schedule.should_run(datetime(2024, 1, 1, 9, 0))
    assert not schedule.should_run(datetime(2024, 1, 1, 9, 30))
    assert not schedule.should_run(datetime(2024, 1, 1, 10, 0))


def test_cron_schedule_specific_day():
    """Test cron schedule for specific day of week."""
    # Every Monday at 9:00
    schedule = CronSchedule("0 9 * * 1")

    # 2024-01-01 is Monday
    assert schedule.should_run(datetime(2024, 1, 1, 9, 0))

    # 2024-01-02 is Tuesday
    assert not schedule.should_run(datetime(2024, 1, 2, 9, 0))


def test_cron_schedule_range():
    """Test cron schedule with range."""
    # Every hour from 9 to 17
    schedule = CronSchedule("0 9-17 * * *")

    assert schedule.should_run(datetime(2024, 1, 1, 9, 0))
    assert schedule.should_run(datetime(2024, 1, 1, 12, 0))
    assert schedule.should_run(datetime(2024, 1, 1, 17, 0))
    assert not schedule.should_run(datetime(2024, 1, 1, 8, 0))
    assert not schedule.should_run(datetime(2024, 1, 1, 18, 0))


def test_cron_schedule_list():
    """Test cron schedule with list."""
    # At minutes 0, 15, 30, 45
    schedule = CronSchedule("0,15,30,45 * * * *")

    assert schedule.should_run(datetime(2024, 1, 1, 12, 0))
    assert schedule.should_run(datetime(2024, 1, 1, 12, 15))
    assert schedule.should_run(datetime(2024, 1, 1, 12, 30))
    assert schedule.should_run(datetime(2024, 1, 1, 12, 45))
    assert not schedule.should_run(datetime(2024, 1, 1, 12, 10))


def test_schedule_task(scheduler):
    """Test scheduling a task."""
    executed = []

    def test_func():
        executed.append(True)

    task = scheduler.schedule_task(
        task_id="test_task",
        name="Test Task",
        func_or_name=test_func,
        schedule="* * * * *",  # Every minute
    )

    assert task.task_id == "test_task"

    # Get task
    retrieved_task = scheduler.get_task(task.task_id)
    assert retrieved_task is not None
    assert retrieved_task.name == "Test Task"
    assert retrieved_task.enabled


def test_task_execution(scheduler):
    """Test task execution."""
    executed = []

    def test_func():
        executed.append(datetime.now())

    scheduler.schedule_task(
        task_id="test_task",
        name="Test Task",
        func_or_name=test_func,
        schedule="* * * * *",
    )

    task = scheduler.get_task("test_task")

    # Execute task
    task.execute()

    assert len(executed) == 1
    assert task.run_count == 1
    assert task.last_run is not None


def test_task_with_args(scheduler):
    """Test task with arguments."""
    results = []

    def test_func(a, b, c=None):
        results.append((a, b, c))

    scheduler.schedule_task(
        task_id="test_task",
        name="Test Task",
        func_or_name=test_func,
        schedule="* * * * *",
        args=(1, 2),
        kwargs={"c": 3},
    )

    task = scheduler.get_task("test_task")
    task.execute()

    assert results == [(1, 2, 3)]


def test_enable_disable_task(scheduler):
    """Test enabling and disabling tasks."""

    def test_func():
        pass

    task = scheduler.schedule_task(
        task_id="test_task",
        name="Test Task",
        func_or_name=test_func,
        schedule="* * * * *",
    )

    # Disable task
    scheduler.disable_task(task.task_id)
    task = scheduler.get_task(task.task_id)
    assert not task.enabled

    # Enable task
    scheduler.enable_task(task.task_id)
    task = scheduler.get_task(task.task_id)
    assert task.enabled


def test_remove_task(scheduler):
    """Test removing a task."""

    def test_func():
        pass

    task_id = scheduler.schedule_task(
        task_id="test_task",
        name="Test Task",
        func_or_name=test_func,
        schedule="* * * * *",
    )

    # Remove task
    scheduler.remove_task(task_id)

    # Task should not exist
    task = scheduler.get_task(task_id)
    assert task is None


def test_list_tasks(scheduler):
    """Test listing tasks."""

    def test_func():
        pass

    # Schedule multiple tasks
    for i in range(3):
        scheduler.schedule_task(
            task_id=f"task_{i}",
            name=f"Task {i}",
            func_or_name=test_func,
            schedule="* * * * *",
        )

    tasks = scheduler.list_tasks()
    assert len(tasks) == 3


def test_register_function(scheduler):
    """Test registering functions."""

    def my_function():
        return "result"

    # Register function
    scheduler.register_function("my_func", my_function)

    # Schedule using registered name
    scheduler.schedule_task(
        task_id="test_task",
        name="Test Task",
        func_or_name="my_func",
        schedule="* * * * *",
    )

    task = scheduler.get_task("test_task")
    result = task.execute()

    assert result == "result"


def test_get_next_run_time(scheduler):
    """Test getting next run time."""

    def test_func():
        pass

    # Schedule task for 9:00 every day
    task = scheduler.schedule_task(
        task_id="test_task",
        name="Test Task",
        func_or_name=test_func,
        schedule="0 9 * * *",
    )

    next_run = scheduler.get_next_run_time(task.task_id)

    assert next_run is not None
    assert next_run.hour == 9
    assert next_run.minute == 0
