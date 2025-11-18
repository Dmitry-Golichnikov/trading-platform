"""Task queue implementations (local and Redis-based)."""

import json
import logging
import queue
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

logger = logging.getLogger(__name__)


class TaskMessage:
    """Task message container."""

    def __init__(
        self,
        task_id: str,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 0,
        retry_count: int = 0,
        max_retries: int = 3,
    ):
        """
        Initialize task message.

        Args:
            task_id: Unique task ID.
            task_type: Type of task.
            payload: Task payload/data.
            priority: Task priority.
            retry_count: Current retry count.
            max_retries: Maximum retries.
        """
        self.task_id = task_id
        self.task_type = task_type
        self.payload = payload
        self.priority = priority
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.created_at = datetime.now()
        self.processed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskMessage":
        """Create from dictionary."""
        msg = cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            payload=data["payload"],
            priority=data.get("priority", 0),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
        )

        if data.get("created_at"):
            msg.created_at = datetime.fromisoformat(data["created_at"])

        if data.get("processed_at"):
            msg.processed_at = datetime.fromisoformat(data["processed_at"])

        return msg

    def __lt__(self, other):
        """Compare for priority queue."""
        return self.priority > other.priority  # Higher priority first


class BaseTaskQueue(ABC):
    """Base class for task queues."""

    @abstractmethod
    def put(self, message: TaskMessage):
        """Put message in queue."""
        pass

    @abstractmethod
    def get(self, timeout: Optional[float] = None) -> Optional[TaskMessage]:
        """Get message from queue."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get queue size."""
        pass

    @abstractmethod
    def clear(self):
        """Clear queue."""
        pass


class LocalTaskQueue(BaseTaskQueue):
    """
    Local in-memory task queue.

    Uses Python's queue.PriorityQueue for thread-safe operations.
    """

    def __init__(
        self,
        maxsize: int = 0,
        persistence_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize local task queue.

        Args:
            maxsize: Maximum queue size (0 = unlimited).
            persistence_path: Path to persist queue state.
        """
        self.queue: queue.PriorityQueue[TaskMessage] = queue.PriorityQueue(maxsize=maxsize)
        self.persistence_path: Optional[Path] = Path(persistence_path) if persistence_path else None

        if self.persistence_path:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

        logger.info("LocalTaskQueue initialized")

    def put(self, message: TaskMessage):
        """Put message in queue."""
        self.queue.put(message)

        if self.persistence_path:
            self._save_to_disk()

        logger.debug(f"Queued task: {message.task_id} (type={message.task_type})")

    def get(self, timeout: Optional[float] = None) -> Optional[TaskMessage]:
        """Get message from queue."""
        try:
            message = self.queue.get(timeout=timeout)

            if self.persistence_path:
                self._save_to_disk()

            return message

        except queue.Empty:
            return None

    def size(self) -> int:
        """Get queue size."""
        return self.queue.qsize()

    def clear(self):
        """Clear queue."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

        if self.persistence_path:
            self._save_to_disk()

        logger.info("Cleared queue")

    def _save_to_disk(self):
        """Save queue to disk."""
        if not self.persistence_path:
            return

        try:
            # Get all items without removing them
            items = []
            temp_queue = queue.PriorityQueue()

            while not self.queue.empty():
                try:
                    item = self.queue.get_nowait()
                    items.append(item)
                    temp_queue.put(item)
                except queue.Empty:
                    break

            # Restore queue
            self.queue = temp_queue

            # Save to disk
            serialized = [item.to_dict() for item in items]

            with open(self.persistence_path, "w") as f:
                json.dump(serialized, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save queue to disk: {e}")

    def _load_from_disk(self):
        """Load queue from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            with open(self.persistence_path, "r") as f:
                data = json.load(f)

            for item_dict in data:
                message = TaskMessage.from_dict(item_dict)
                self.queue.put(message)

            logger.info(f"Loaded {len(data)} tasks from disk")

        except Exception as e:
            logger.error(f"Failed to load queue from disk: {e}")


class RedisTaskQueue(BaseTaskQueue):
    """
    Redis-based distributed task queue.

    Uses Redis lists for FIFO queue and sorted sets for priority.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        queue_name: str = "tasks",
        use_priority: bool = True,
    ):
        """
        Initialize Redis task queue.

        Args:
            redis_url: Redis connection URL.
            queue_name: Queue name.
            use_priority: Whether to use priority queue.
        """
        self.redis_url = redis_url
        self.queue_name = queue_name
        self.use_priority = use_priority

        self.redis_available = False
        self.redis: Optional[Any] = None
        self.fallback_queue: Optional[LocalTaskQueue] = None

        self._init_redis()

    def _init_redis(self):
        """Initialize Redis connection."""
        try:
            import redis

            self.redis = redis.from_url(self.redis_url)
            self.redis.ping()
            self.redis_available = True

            logger.info(f"Redis queue initialized: {self.queue_name}")

        except ImportError:
            logger.warning("Redis not installed. Using local queue fallback.")
            self._fallback_to_local()

        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using local queue fallback.")
            self._fallback_to_local()

    def _fallback_to_local(self):
        """Fallback to local queue if Redis unavailable."""
        self.fallback_queue = LocalTaskQueue()
        logger.info("Using local queue fallback")

    def put(self, message: TaskMessage):
        """Put message in queue."""
        if not self.redis_available or self.redis is None:
            if self.fallback_queue:
                self.fallback_queue.put(message)
            return

        try:
            # Serialize message
            serialized = json.dumps(message.to_dict())

            if self.use_priority:
                # Use sorted set for priority
                # Score = -priority (for descending order)
                self.redis.zadd(
                    self.queue_name,
                    {serialized: -message.priority},
                )
            else:
                # Use list for FIFO
                self.redis.rpush(self.queue_name, serialized)

            logger.debug(f"Queued task to Redis: {message.task_id}")

        except Exception as e:
            logger.error(f"Failed to queue task to Redis: {e}")
            raise

    def get(self, timeout: Optional[float] = None) -> Optional[TaskMessage]:
        """Get message from queue."""
        if not self.redis_available or self.redis is None:
            if self.fallback_queue:
                return self.fallback_queue.get(timeout=timeout)
            return None

        try:
            if self.use_priority:
                # Get from sorted set (highest priority)
                result = self.redis.zpopmin(self.queue_name, count=1)

                if not result:
                    if timeout:
                        # Simple polling for timeout
                        start_time = time.time()
                        while time.time() - start_time < timeout:
                            result = self.redis.zpopmin(self.queue_name, count=1)
                            if result:
                                break
                            time.sleep(0.1)

                    if not result:
                        return None

                serialized, _ = result[0]

            else:
                # Get from list (FIFO)
                if timeout:
                    result = self.redis.blpop(self.queue_name, timeout=int(timeout))
                    if not result:
                        return None
                    _, serialized = result
                else:
                    serialized = self.redis.lpop(self.queue_name)
                    if not serialized:
                        return None

            # Deserialize
            data = json.loads(serialized)
            message = TaskMessage.from_dict(data)

            return message

        except Exception as e:
            logger.error(f"Failed to get task from Redis: {e}")
            raise

    def size(self) -> int:
        """Get queue size."""
        if not self.redis_available or self.redis is None:
            if self.fallback_queue:
                return self.fallback_queue.size()
            return 0

        try:
            if self.use_priority:
                return self.redis.zcard(self.queue_name)
            else:
                return self.redis.llen(self.queue_name)

        except Exception as e:
            logger.error(f"Failed to get queue size from Redis: {e}")
            return 0

    def clear(self):
        """Clear queue."""
        if not self.redis_available or self.redis is None:
            if self.fallback_queue:
                self.fallback_queue.clear()
            return

        try:
            self.redis.delete(self.queue_name)
            logger.info(f"Cleared Redis queue: {self.queue_name}")

        except Exception as e:
            logger.error(f"Failed to clear Redis queue: {e}")


class TaskQueueManager:
    """
    Manager for task queues.

    Provides unified interface and automatic fallback.
    """

    def __init__(
        self,
        backend: str = "local",
        redis_url: str = "redis://localhost:6379",
        queue_name: str = "tasks",
        persistence_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize task queue manager.

        Args:
            backend: Backend type ('local' or 'redis').
            redis_url: Redis URL for redis backend.
            queue_name: Queue name.
            persistence_path: Path for local queue persistence.
        """
        self.backend = backend
        self.queue_name = queue_name

        # Create queue
        if backend == "redis":
            self.queue: BaseTaskQueue = RedisTaskQueue(
                redis_url=redis_url,
                queue_name=queue_name,
                use_priority=True,
            )
        elif backend == "local":
            self.queue = LocalTaskQueue(
                persistence_path=persistence_path,
            )
        else:
            raise ValueError(f"Invalid backend: {backend}")

        # Statistics
        self.stats = {
            "total_queued": 0,
            "total_processed": 0,
            "total_failed": 0,
        }

        logger.info(f"TaskQueueManager initialized: backend={backend}")

    def enqueue(
        self,
        task_id: str,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 0,
    ) -> str:
        """
        Enqueue task.

        Args:
            task_id: Task ID.
            task_type: Task type.
            payload: Task payload.
            priority: Task priority.

        Returns:
            Task ID.
        """
        message = TaskMessage(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
        )

        self.queue.put(message)
        self.stats["total_queued"] += 1

        return task_id

    def dequeue(self, timeout: Optional[float] = None) -> Optional[TaskMessage]:
        """
        Dequeue task.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Task message or None.
        """
        message = self.queue.get(timeout=timeout)

        if message:
            message.processed_at = datetime.now()
            self.stats["total_processed"] += 1

        return message

    def size(self) -> int:
        """Get queue size."""
        return self.queue.size()

    def clear(self):
        """Clear queue."""
        self.queue.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            **self.stats,
            "current_size": self.size(),
            "backend": self.backend,
            "queue_name": self.queue_name,
        }

    def process_tasks(
        self,
        processor: Callable[[TaskMessage], None],
        max_tasks: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """
        Process tasks from queue.

        Args:
            processor: Function to process task message.
            max_tasks: Maximum tasks to process.
            timeout: Timeout for getting tasks.
        """
        processed = 0

        while True:
            # Check max_tasks
            if max_tasks and processed >= max_tasks:
                break

            # Get task
            message = self.dequeue(timeout=timeout)

            if message is None:
                break

            # Process task
            try:
                processor(message)
                logger.debug(f"Processed task: {message.task_id}")

            except Exception as e:
                logger.error(f"Failed to process task {message.task_id}: {e}")
                self.stats["total_failed"] += 1

                # Retry logic
                if message.retry_count < message.max_retries:
                    message.retry_count += 1
                    self.queue.put(message)
                    logger.info(f"Re-queued task for retry: {message.task_id}")

            processed += 1

        logger.info(f"Processed {processed} tasks")
