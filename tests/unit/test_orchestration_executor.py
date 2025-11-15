"""Tests for TaskExecutor."""

import time

import pytest

from src.orchestration import Task, TaskExecutor, TaskStatus


@pytest.fixture
def executor():
    """Create task executor."""
    return TaskExecutor(max_workers=2, execution_mode="thread")


def test_task_creation():
    """Test task creation."""

    def test_func(x):
        return x * 2

    task = Task(
        task_id="test_task",
        func=test_func,
        args=(5,),
        name="Test Task",
        priority=1,
    )

    assert task.task_id == "test_task"
    assert task.name == "Test Task"
    assert task.priority == 1
    assert task.status == TaskStatus.PENDING


def test_task_execution():
    """Test task execution."""

    def test_func(x):
        return x * 2

    task = Task(
        task_id="test_task",
        func=test_func,
        args=(5,),
    )

    result = task.execute()

    assert result == 10
    assert task.status == TaskStatus.COMPLETED
    assert task.result == 10
    assert task.started_at is not None
    assert task.completed_at is not None


def test_task_error():
    """Test task error handling."""

    def failing_func():
        raise ValueError("Test error")

    task = Task(
        task_id="test_task",
        func=failing_func,
    )

    with pytest.raises(ValueError):
        task.execute()

    assert task.status == TaskStatus.FAILED
    assert task.error is not None


def test_submit_task(executor):
    """Test submitting a task."""

    def test_func(x):
        return x * 2

    task_id = executor.submit(
        test_func,
        5,
        name="Test Task",
    )

    assert task_id is not None

    # Wait for task to complete
    result = executor.get_result(task_id, timeout=5)

    assert result == 10


def test_submit_multiple_tasks(executor):
    """Test submitting multiple tasks."""

    def test_func(x):
        time.sleep(0.1)
        return x * 2

    task_ids = []
    for i in range(5):
        task_id = executor.submit(test_func, i, name=f"Task {i}")
        task_ids.append(task_id)

    # Wait for all tasks
    executor.wait_for_all(task_ids, timeout=10)

    # Check results
    for i, task_id in enumerate(task_ids):
        result = executor.get_result(task_id)
        assert result == i * 2


def test_get_task_status(executor):
    """Test getting task status."""

    def test_func():
        time.sleep(0.1)

    task_id = executor.submit(test_func, name="Test Task")

    # Initially should be pending or running
    status = executor.get_status(task_id)
    assert status in [TaskStatus.PENDING, TaskStatus.RUNNING]

    # Wait for completion
    executor.wait_for_task(task_id, timeout=5)

    status = executor.get_status(task_id)
    assert status == TaskStatus.COMPLETED


def test_list_tasks(executor):
    """Test listing tasks."""

    def test_func():
        pass

    # Submit multiple tasks
    for i in range(3):
        executor.submit(test_func, name=f"Task {i}")

    tasks = executor.list_tasks()
    assert len(tasks) == 3


def test_list_tasks_by_status(executor):
    """Test listing tasks by status."""

    def test_func():
        time.sleep(0.1)

    # Submit tasks
    task_ids = []
    for i in range(3):
        task_id = executor.submit(test_func, name=f"Task {i}")
        task_ids.append(task_id)

    # Wait for all to complete
    executor.wait_for_all(task_ids, timeout=5)

    # List completed tasks
    completed = executor.list_tasks(status=TaskStatus.COMPLETED)
    assert len(completed) == 3


def test_cancel_task(executor):
    """Test cancelling a task."""

    def slow_func():
        time.sleep(10)

    task_id = executor.submit(slow_func, name="Slow Task")

    # Cancel immediately
    cancelled = executor.cancel_task(task_id)

    # Note: cancellation might not always succeed if task already started
    # This is expected behavior
    if cancelled:
        status = executor.get_status(task_id)
        assert status == TaskStatus.CANCELLED


def test_executor_statistics(executor):
    """Test getting executor statistics."""

    def test_func(x):
        return x * 2

    # Submit tasks
    for i in range(5):
        executor.submit(test_func, i)

    # Wait for completion
    time.sleep(1)

    stats = executor.get_statistics()

    assert stats["total_tasks"] == 5
    assert stats["max_workers"] == 2
    assert stats["execution_mode"] == "thread"


def test_clear_completed(executor):
    """Test clearing completed tasks."""

    def test_func():
        pass

    # Submit and complete tasks
    task_ids = []
    for i in range(3):
        task_id = executor.submit(test_func)
        task_ids.append(task_id)

    executor.wait_for_all(task_ids, timeout=5)

    # Clear completed
    executor.clear_completed()

    tasks = executor.list_tasks()
    assert len(tasks) == 0


def test_submit_batch(executor):
    """Test batch submission."""

    def test_func(x):
        return x * 2

    tasks = [{"func": test_func, "args": (i,), "name": f"Task {i}"} for i in range(5)]

    task_ids = executor.submit_batch(tasks)

    assert len(task_ids) == 5

    executor.wait_for_all(task_ids, timeout=5)

    for i, task_id in enumerate(task_ids):
        result = executor.get_result(task_id)
        assert result == i * 2


def test_context_manager(executor):
    """Test executor as context manager."""

    def test_func():
        return "result"

    with executor:
        task_id = executor.submit(test_func)
        result = executor.get_result(task_id, timeout=5)

    assert result == "result"
