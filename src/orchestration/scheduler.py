"""Task scheduler with cron-like scheduling support."""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class CronSchedule:
    """
    Simple cron-like schedule parser and matcher.

    Supports format: "minute hour day month day_of_week"
    Use "*" for any value, numbers for specific values.
    """

    def __init__(self, schedule: str):
        """
        Initialize cron schedule.

        Args:
            schedule: Cron-like schedule string.
                Examples:
                - "* * * * *" - every minute
                - "0 * * * *" - every hour
                - "0 0 * * *" - every day at midnight
                - "0 9 * * 1" - every Monday at 9am
        """
        self.schedule = schedule
        self.parts = schedule.split()

        if len(self.parts) != 5:
            raise ValueError("Invalid cron schedule. Expected format: " "'minute hour day month day_of_week'")

        self.minute = self._parse_field(self.parts[0], 0, 59)
        self.hour = self._parse_field(self.parts[1], 0, 23)
        self.day = self._parse_field(self.parts[2], 1, 31)
        self.month = self._parse_field(self.parts[3], 1, 12)
        self.day_of_week = self._parse_field(self.parts[4], 0, 7)
        if self.day_of_week is not None:
            # Normalize: 7 -> 0 (both mean Sunday in cron)
            normalized = []
            for val in self.day_of_week:
                if val == 7:
                    normalized.append(0)
                else:
                    normalized.append(val)
            self.day_of_week = normalized

    def _parse_field(
        self,
        field: str,
        min_val: int,
        max_val: int,
    ) -> Optional[List[int]]:
        """Parse cron field."""
        if field == "*":
            return None  # Match any

        # Handle lists (e.g., "1,3,5")
        if "," in field:
            values = [int(x) for x in field.split(",")]
            for val in values:
                if not (min_val <= val <= max_val):
                    raise ValueError(f"Value {val} out of range [{min_val}, {max_val}]")
            return values

        # Handle ranges (e.g., "1-5")
        if "-" in field:
            start, end = field.split("-")
            return list(range(int(start), int(end) + 1))

        # Single value
        val = int(field)
        if not (min_val <= val <= max_val):
            raise ValueError(f"Value {val} out of range [{min_val}, {max_val}]")
        return [val]

    def should_run(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if task should run at given datetime.

        Args:
            dt: Datetime to check. If None, uses current time.

        Returns:
            True if task should run.
        """
        if dt is None:
            dt = datetime.now()

        # Check each field
        if self.minute is not None and dt.minute not in self.minute:
            return False

        if self.hour is not None and dt.hour not in self.hour:
            return False

        if self.day is not None and dt.day not in self.day:
            return False

        if self.month is not None and dt.month not in self.month:
            return False

        if self.day_of_week is not None:
            # Convert Python weekday to cron-style
            # Python: Monday=0, Tuesday=1, ..., Sunday=6
            # Cron: Sunday=0, Monday=1, Tuesday=2, ..., Saturday=6
            python_weekday = dt.weekday()  # Monday=0, Sunday=6
            # Convert: Python Sunday (6) -> Cron Sunday (0)
            #          Python Monday (0) -> Cron Monday (1)
            #          Python Tuesday (1) -> Cron Tuesday (2)
            #          etc.
            cron_weekday = (python_weekday + 1) % 7
            if cron_weekday not in self.day_of_week:
                return False

        return True


class ScheduledTask:
    """Scheduled task container."""

    def __init__(
        self,
        task_id: str,
        name: str,
        func: Callable,
        schedule: Union[str, CronSchedule],
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        enabled: bool = True,
    ):
        """
        Initialize scheduled task.

        Args:
            task_id: Unique task ID.
            name: Task name.
            func: Function to execute.
            schedule: Cron schedule string or CronSchedule object.
            args: Positional arguments for function.
            kwargs: Keyword arguments for function.
            enabled: Whether task is enabled.
        """
        self.task_id = task_id
        self.name = name
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.enabled = enabled

        # Parse schedule
        if isinstance(schedule, str):
            self.schedule = CronSchedule(schedule)
        else:
            self.schedule = schedule

        # Tracking
        self.last_run: Optional[datetime] = None
        self.next_run: Optional[datetime] = None
        self.run_count = 0
        self.error_count = 0
        self.last_error: Optional[str] = None

    def should_run(self, dt: Optional[datetime] = None) -> bool:
        """Check if task should run."""
        if not self.enabled:
            return False

        if dt is None:
            dt = datetime.now()

        # Check schedule
        if not self.schedule.should_run(dt):
            return False

        # Check if already ran this minute
        if self.last_run:
            # Prevent multiple runs in same minute
            if (
                self.last_run.year == dt.year
                and self.last_run.month == dt.month
                and self.last_run.day == dt.day
                and self.last_run.hour == dt.hour
                and self.last_run.minute == dt.minute
            ):
                return False

        return True

    def execute(self):
        """Execute task."""
        try:
            logger.info(f"Executing task: {self.name} ({self.task_id})")

            self.last_run = datetime.now()
            result = self.func(*self.args, **self.kwargs)
            self.run_count += 1

            logger.info(f"Task completed: {self.name}")
            return result

        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)

            logger.error(f"Task failed: {self.name} - {e}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "schedule": self.schedule.schedule,
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
        }


class TaskScheduler:
    """
    Task scheduler with cron-like scheduling.

    Supports:
    - Cron-like schedules
    - Task management (add, remove, enable, disable)
    - Persistent task storage
    - Background execution
    """

    def __init__(
        self,
        storage_path: Optional[Union[str, Path]] = None,
        check_interval: int = 60,
    ):
        """
        Initialize task scheduler.

        Args:
            storage_path: Path to store task configurations.
            check_interval: Interval in seconds to check for tasks.
        """
        self.storage_path = Path(storage_path or "artifacts/scheduler/tasks.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.check_interval = check_interval
        self.tasks: Dict[str, ScheduledTask] = {}
        self.task_registry: Dict[str, Callable] = {}

        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Load tasks
        self._load_tasks()

    def register_function(self, name: str, func: Callable):
        """
        Register function for scheduling.

        Args:
            name: Function name/identifier.
            func: Function to register.
        """
        self.task_registry[name] = func
        logger.debug(f"Registered function: {name}")

    def schedule_task(
        self,
        task_id: str,
        name: str,
        func_or_name: Union[Callable, str],
        schedule: str,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        enabled: bool = True,
    ) -> ScheduledTask:
        """
        Schedule a task.

        Args:
            task_id: Unique task ID.
            name: Task name.
            func_or_name: Function or registered function name.
            schedule: Cron schedule string.
            args: Positional arguments.
            kwargs: Keyword arguments.
            enabled: Whether task is enabled.

        Returns:
            ScheduledTask object.
        """
        # Resolve function
        if isinstance(func_or_name, str):
            if func_or_name not in self.task_registry:
                raise ValueError(f"Function not registered: {func_or_name}")
            func = self.task_registry[func_or_name]
        else:
            func = func_or_name

        # Create task
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            func=func,
            schedule=schedule,
            args=args,
            kwargs=kwargs,
            enabled=enabled,
        )

        self.tasks[task_id] = task
        self._save_tasks()

        logger.info(f"Scheduled task: {name} ({task_id}) - {schedule}")
        return task

    def schedule_training(
        self,
        config: Dict[str, Any],
        schedule: str,
        task_id: Optional[str] = None,
    ) -> str:
        """
        Schedule model training.

        Args:
            config: Training configuration.
            schedule: Cron schedule.
            task_id: Optional task ID.

        Returns:
            Task ID.
        """
        if task_id is None:
            task_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        def training_func():
            # Import here to avoid circular dependencies
            from src.pipelines.training_pipeline import TrainingPipeline

            pipeline = TrainingPipeline()
            return pipeline.execute(config)

        task = self.schedule_task(
            task_id=task_id,
            name=f"Training: {config.get('model_name', 'unnamed')}",
            func_or_name=training_func,
            schedule=schedule,
        )

        return task.task_id

    def schedule_backtest(
        self,
        model_id: str,
        schedule: str,
        task_id: Optional[str] = None,
    ) -> str:
        """
        Schedule backtest.

        Args:
            model_id: Model ID to backtest.
            schedule: Cron schedule.
            task_id: Optional task ID.

        Returns:
            Task ID.
        """
        if task_id is None:
            task_id = f"backtest_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        def backtest_func():
            # Import here to avoid circular dependencies
            from src.pipelines.backtest_pipeline import BacktestPipeline

            pipeline = BacktestPipeline()
            return pipeline.execute({"model_id": model_id})

        task = self.schedule_task(
            task_id=task_id,
            name=f"Backtest: {model_id}",
            func_or_name=backtest_func,
            schedule=schedule,
        )

        return task.task_id

    def schedule_pipeline(
        self,
        pipeline_name: str,
        config: Dict[str, Any],
        schedule: str,
        task_id: Optional[str] = None,
    ) -> str:
        """
        Schedule pipeline execution.

        Args:
            pipeline_name: Pipeline name.
            config: Pipeline configuration.
            schedule: Cron schedule.
            task_id: Optional task ID.

        Returns:
            Task ID.
        """
        if task_id is None:
            task_id = f"pipeline_{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        def pipeline_func():
            # Import here to avoid circular dependencies
            from src.pipelines.pipeline_factory import PipelineFactory

            pipeline = PipelineFactory.create(pipeline_name)
            return pipeline.execute(config)

        task = self.schedule_task(
            task_id=task_id,
            name=f"Pipeline: {pipeline_name}",
            func_or_name=pipeline_func,
            schedule=schedule,
        )

        return task.task_id

    def remove_task(self, task_id: str):
        """
        Remove scheduled task.

        Args:
            task_id: Task ID.
        """
        if task_id in self.tasks:
            del self.tasks[task_id]
            self._save_tasks()
            logger.info(f"Removed task: {task_id}")
        else:
            logger.warning(f"Task not found: {task_id}")

    def enable_task(self, task_id: str):
        """Enable task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
            self._save_tasks()
            logger.info(f"Enabled task: {task_id}")

    def disable_task(self, task_id: str):
        """Disable task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False
            self._save_tasks()
            logger.info(f"Disabled task: {task_id}")

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all tasks."""
        return [task.to_dict() for task in self.tasks.values()]

    def start(self):
        """Start scheduler."""
        if self.running:
            logger.warning("Scheduler already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        logger.info("Scheduler started")

    def stop(self):
        """Stop scheduler."""
        self.running = False

        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None

        logger.info("Scheduler stopped")

    def _run_loop(self):
        """Main scheduler loop."""
        logger.info("Scheduler loop started")

        while self.running:
            try:
                current_time = datetime.now()

                # Check each task
                for task in self.tasks.values():
                    if task.should_run(current_time):
                        try:
                            task.execute()
                        except Exception as e:
                            logger.error(f"Task execution failed: {task.name} - {e}")

                # Sleep until next check
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(self.check_interval)

        logger.info("Scheduler loop stopped")

    def _save_tasks(self):
        """Save tasks to storage."""
        tasks_data = []

        for task in self.tasks.values():
            # Only save metadata, not the function itself
            task_data = {
                "task_id": task.task_id,
                "name": task.name,
                "schedule": task.schedule.schedule,
                "enabled": task.enabled,
                "last_run": task.last_run.isoformat() if task.last_run else None,
                "run_count": task.run_count,
                "error_count": task.error_count,
            }

            tasks_data.append(task_data)

        with open(self.storage_path, "w") as f:
            json.dump(tasks_data, f, indent=2)

    def _load_tasks(self):
        """Load tasks from storage."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r") as f:
                tasks_data = json.load(f)

            logger.info(f"Loaded {len(tasks_data)} task configurations")

            # Note: Functions are not persisted, they need to be re-registered
            # This is intentional for security and flexibility

        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")

    def get_next_run_time(self, task_id: str) -> Optional[datetime]:
        """
        Get next scheduled run time for task.

        Args:
            task_id: Task ID.

        Returns:
            Next run datetime.
        """
        task = self.get_task(task_id)

        if not task or not task.enabled:
            return None

        # Find next matching time
        current_time = datetime.now()
        check_time = current_time.replace(second=0, microsecond=0)

        # Check up to 7 days ahead
        for _ in range(7 * 24 * 60):
            check_time += timedelta(minutes=1)

            if task.schedule.should_run(check_time):
                return check_time

        return None
