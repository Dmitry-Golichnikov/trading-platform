"""Task executor for running jobs asynchronously."""

import logging
import multiprocessing
import threading
import time
import uuid
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task:
    """Task container."""

    def __init__(
        self,
        task_id: str,
        func: Callable,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        name: Optional[str] = None,
        priority: int = 0,
    ):
        """
        Initialize task.

        Args:
            task_id: Unique task ID.
            func: Function to execute.
            args: Positional arguments.
            kwargs: Keyword arguments.
            name: Task name.
            priority: Task priority (higher = more important).
        """
        self.task_id = task_id
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.name = name or task_id
        self.priority = priority

        self.status = TaskStatus.PENDING
        self.result: Any = None
        self.error: Optional[Exception] = None
        self.future: Optional[Future] = None

        self.submitted_at: Optional[datetime] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    def execute(self) -> Any:
        """Execute task synchronously."""
        try:
            self.status = TaskStatus.RUNNING
            self.started_at = datetime.now()

            logger.info(f"Executing task: {self.name}")

            result = self.func(*self.args, **self.kwargs)

            self.result = result
            self.status = TaskStatus.COMPLETED
            self.completed_at = datetime.now()

            logger.info(f"Task completed: {self.name}")
            return result

        except Exception as e:
            self.error = e
            self.status = TaskStatus.FAILED
            self.completed_at = datetime.now()

            logger.error(f"Task failed: {self.name} - {e}")
            raise

    def cancel(self) -> bool:
        """
        Cancel task.

        Returns:
            True if cancelled successfully.
        """
        if self.status == TaskStatus.PENDING:
            self.status = TaskStatus.CANCELLED
            return True

        if self.future and not self.future.done():
            cancelled = self.future.cancel()
            if cancelled:
                self.status = TaskStatus.CANCELLED
            return cancelled

        return False

    def is_done(self) -> bool:
        """Check if task is done."""
        return self.status in [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ]

    def get_duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status.value,
            "priority": self.priority,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration": self.get_duration(),
            "error": str(self.error) if self.error else None,
        }


class TaskExecutor:
    """
    Task executor for running jobs asynchronously.

    Supports:
    - Thread-based execution (I/O bound tasks)
    - Process-based execution (CPU bound tasks)
    - Task prioritization
    - Progress tracking
    - Cancellation
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        execution_mode: str = "thread",
        storage_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize task executor.

        Args:
            max_workers: Maximum number of workers. If None, uses CPU count.
            execution_mode: 'thread' or 'process'.
            storage_dir: Directory to store task metadata.
        """
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.execution_mode = execution_mode

        if storage_dir:
            self.storage_dir: Optional[Path] = Path(storage_dir)
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.storage_dir = None

        # Task tracking
        self.tasks: Dict[str, Task] = {}
        self.task_lock = threading.Lock()

        # Executor
        self.executor: Executor
        if execution_mode == "thread":
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        elif execution_mode == "process":
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            raise ValueError(f"Invalid execution mode: {execution_mode}")

        logger.info(f"TaskExecutor initialized: mode={execution_mode}, " f"workers={self.max_workers}")

    def submit(
        self,
        func: Callable,
        *func_args,
        task_id: Optional[str] = None,
        name: Optional[str] = None,
        priority: int = 0,
        **func_kwargs,
    ) -> str:
        """
        Submit task for execution.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            task_id: Optional task ID.
            name: Optional task name.
            priority: Task priority.
            **kwargs: Keyword arguments.

        Returns:
            Task ID.
        """
        # Generate task ID if not provided
        if task_id is None:
            task_id = f"task_{uuid.uuid4().hex}"

        # Create task
        task = Task(
            task_id=task_id,
            func=func,
            args=func_args,
            kwargs=func_kwargs,
            name=name,
            priority=priority,
        )

        task.submitted_at = datetime.now()

        # Submit to executor
        future = self.executor.submit(self._execute_task, task)
        task.future = future

        # Store task
        with self.task_lock:
            self.tasks[task_id] = task

        logger.info(f"Submitted task: {task.name} ({task_id})")
        return task_id

    def _execute_task(self, task: Task) -> Any:
        """Execute task (internal)."""
        try:
            return task.execute()

        except Exception as e:
            logger.error(f"Task execution error: {e}")
            raise

    def submit_batch(
        self,
        tasks: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Submit multiple tasks.

        Args:
            tasks: List of task specifications.
                Each dict should have: 'func', 'args', 'kwargs', 'name', etc.

        Returns:
            List of task IDs.
        """
        task_ids = []

        for task_spec in tasks:
            args: Sequence[Any] = task_spec.get("args", ())
            kwargs: Dict[str, Any] = task_spec.get("kwargs", {})

            task_id = self.submit(
                task_spec["func"],
                *args,
                task_id=task_spec.get("task_id"),
                name=task_spec.get("name"),
                priority=task_spec.get("priority", 0),
                **kwargs,
            )
            task_ids.append(task_id)

        return task_ids

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID.

        Args:
            task_id: Task ID.

        Returns:
            Task object.
        """
        with self.task_lock:
            return self.tasks.get(task_id)

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get task result (blocks until complete).

        Args:
            task_id: Task ID.
            timeout: Timeout in seconds.

        Returns:
            Task result.
        """
        task = self.get_task(task_id)

        if task is None:
            raise ValueError(f"Task not found: {task_id}")

        if task.future is None:
            raise ValueError(f"Task not submitted: {task_id}")

        # Wait for result
        result = task.future.result(timeout=timeout)
        return result

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel task.

        Args:
            task_id: Task ID.

        Returns:
            True if cancelled successfully.
        """
        task = self.get_task(task_id)

        if task is None:
            logger.warning(f"Task not found: {task_id}")
            return False

        cancelled = task.cancel()

        if cancelled:
            logger.info(f"Cancelled task: {task_id}")

        return cancelled

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
    ) -> List[Dict[str, Any]]:
        """
        List tasks.

        Args:
            status: Filter by status.

        Returns:
            List of task info dictionaries.
        """
        with self.task_lock:
            tasks = list(self.tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]

        return [task.to_dict() for task in tasks]

    def get_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get task status.

        Args:
            task_id: Task ID.

        Returns:
            Task status.
        """
        task = self.get_task(task_id)

        if task is None:
            return None

        return task.status

    def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Wait for task to complete.

        Args:
            task_id: Task ID.
            timeout: Timeout in seconds.

        Returns:
            True if completed, False if timeout.
        """
        task = self.get_task(task_id)

        if task is None:
            raise ValueError(f"Task not found: {task_id}")

        if task.future is None:
            return task.is_done()

        start_time = time.time()

        while not task.is_done():
            if timeout and (time.time() - start_time) > timeout:
                return False

            time.sleep(0.1)

        return True

    def wait_for_all(
        self,
        task_ids: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Wait for all tasks to complete.

        Args:
            task_ids: List of task IDs. If None, waits for all tasks.
            timeout: Timeout in seconds.

        Returns:
            True if all completed, False if timeout.
        """
        if task_ids is None:
            with self.task_lock:
                task_ids = list(self.tasks.keys())

        start_time = time.time()

        for task_id in task_ids:
            remaining_timeout = None
            if timeout:
                remaining_timeout = timeout - (time.time() - start_time)
                if remaining_timeout <= 0:
                    return False

            if not self.wait_for_task(task_id, timeout=remaining_timeout):
                return False

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get executor statistics.

        Returns:
            Statistics dictionary.
        """
        with self.task_lock:
            tasks = list(self.tasks.values())

        status_counts: Dict[str, int] = {}
        for task in tasks:
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # Calculate durations
        completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        durations: List[float] = [duration for t in completed_tasks if (duration := t.get_duration()) is not None]

        avg_duration = sum(durations) / len(durations) if durations else 0.0
        min_duration = min(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0.0

        return {
            "total_tasks": len(tasks),
            "status_counts": status_counts,
            "max_workers": self.max_workers,
            "execution_mode": self.execution_mode,
            "avg_duration": avg_duration,
            "min_duration": min_duration,
            "max_duration": max_duration,
        }

    def clear_completed(self):
        """Clear completed tasks from memory."""
        with self.task_lock:
            task_ids_to_remove = [
                task_id
                for task_id, task in self.tasks.items()
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
            ]

            for task_id in task_ids_to_remove:
                del self.tasks[task_id]

        logger.info(f"Cleared {len(task_ids_to_remove)} completed tasks")

    def shutdown(self, wait: bool = True):
        """
        Shutdown executor.

        Args:
            wait: Whether to wait for pending tasks.
        """
        logger.info("Shutting down executor")
        self.executor.shutdown(wait=wait)
        logger.info("Executor shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown(wait=True)
