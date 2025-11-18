"""
Labeling Task Manager Service

Handles long-running labeling tasks with pause/resume support.
"""

import asyncio
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.interfaces.gui.backend.api.models import (
    LabelingTaskCreateRequest,
    LabelingTaskInfo,
    TaskStatus,
    TaskUpdate,
)
from src.interfaces.gui.backend.api.routers.websocket import manager as ws_manager
from src.interfaces.gui.backend.services.labeling_service import LabelingService
from src.labeling.methods.horizon import HorizonLabeler
from src.labeling.methods.triple_barrier import TripleBarrierLabeler
from src.labeling.methods.regression_targets import RegressionTargetsLabeler
from src.labeling.pipeline import LabelingPipeline


@dataclass
class LabelingTaskState:
    """Internal representation of labeling task"""

    id: str
    name: str
    dataset_id: str
    labeling_set_id: str
    method: str
    config: Dict[str, Any]
    status: TaskStatus
    progress: float
    processed_rows: int
    total_rows: int
    message: Optional[str]
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    description: Optional[str] = None
    feature_set_id: Optional[str] = None
    class_distribution: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "dataset_id": self.dataset_id,
            "labeling_set_id": self.labeling_set_id,
            "method": self.method,
            "config": self.config,
            "status": self.status.value,
            "progress": self.progress,
            "processed_rows": self.processed_rows,
            "total_rows": self.total_rows,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "description": self.description,
            "feature_set_id": self.feature_set_id,
            "class_distribution": self.class_distribution,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "LabelingTaskState":
        def _parse_dt(value: Optional[str]) -> Optional[datetime]:
            return datetime.fromisoformat(value) if value else None

        return LabelingTaskState(
            id=payload["id"],
            name=payload["name"],
            dataset_id=payload["dataset_id"],
            labeling_set_id=payload["labeling_set_id"],
            method=payload.get("method", "horizon"),
            config=payload.get("config", {}),
            status=TaskStatus(payload.get("status", "queued")),
            progress=payload.get("progress", 0.0),
            processed_rows=payload.get("processed_rows", 0),
            total_rows=payload.get("total_rows", 0),
            message=payload.get("message"),
            created_at=_parse_dt(payload.get("created_at")) or datetime.utcnow(),
            updated_at=_parse_dt(payload.get("updated_at")) or datetime.utcnow(),
            started_at=_parse_dt(payload.get("started_at")),
            finished_at=_parse_dt(payload.get("finished_at")),
            description=payload.get("description"),
            feature_set_id=payload.get("feature_set_id"),
            class_distribution=payload.get("class_distribution"),
        )


class TaskRuntime:
    """Runtime controls for pause/resume/cancel"""

    def __init__(self) -> None:
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.cancel_event = threading.Event()

    def pause(self) -> None:
        self.pause_event.clear()

    def resume(self) -> None:
        self.pause_event.set()

    def cancel(self) -> None:
        self.cancel_event.set()

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def wait_if_paused(self) -> None:
        while not self.pause_event.is_set() and not self.cancel_event.is_set():
            time.sleep(0.25)


class LabelingTaskManager:
    """Manager for long-running labeling tasks"""

    def __init__(self, labeling_service: LabelingService, max_workers: int = 1):
        self.labeling_service = labeling_service
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, LabelingTaskState] = {}
        self.runtimes: Dict[str, TaskRuntime] = {}
        self.lock = threading.Lock()
        self.tasks_dir = self.labeling_service.labeling_dir / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self._load_tasks()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        dataset_id: Optional[str] = None,
    ) -> List[LabelingTaskInfo]:
        with self.lock:
            states = list(self.tasks.values())

        filtered = []
        for state in states:
            if status and state.status != status:
                continue
            if dataset_id and state.dataset_id != dataset_id:
                continue
            filtered.append(self._to_info(state))

        filtered.sort(key=lambda item: item.created_at, reverse=True)
        return filtered

    def get_task(self, task_id: str) -> LabelingTaskInfo:
        with self.lock:
            state = self.tasks.get(task_id)
            if not state:
                raise ValueError(f"Task {task_id} not found")
            return self._to_info(state)

    def create_tasks(self, request: LabelingTaskCreateRequest) -> List[LabelingTaskInfo]:
        dataset_ids = request.dataset_ids or []
        if request.apply_to_all or not dataset_ids:
            available = [ds.id for ds in self.labeling_service.dataset_service.list_datasets()]
            if dataset_ids:
                dataset_ids = [ds for ds in available if ds in dataset_ids]
            else:
                dataset_ids = available

        if not dataset_ids:
            raise ValueError("Нет доступных датасетов для разметки")

        created: List[LabelingTaskInfo] = []

        for dataset_id in dataset_ids:
            task_state = LabelingTaskState(
                id=str(uuid.uuid4()),
                name=request.name,
                dataset_id=dataset_id,
                labeling_set_id=self.labeling_service.build_labeling_set_id(dataset_id, request.name),
                method=request.method.value,
                config=request.config,
                status=TaskStatus.queued,
                progress=0.0,
                processed_rows=0,
                total_rows=0,
                message="Queued",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                description=request.description,
                feature_set_id=request.feature_set_id,
            )

            with self.lock:
                self.tasks[task_state.id] = task_state
                self._persist_task_locked(task_state)
                info = self._to_info(task_state)
            created.append(info)

            if request.auto_start:
                self._submit(task_state.id)

        return created

    def pause_task(self, task_id: str) -> LabelingTaskInfo:
        with self.lock:
            state = self._require_task(task_id)
            runtime = self.runtimes.get(task_id)
            if state.status != TaskStatus.running:
                return self._to_info(state)
            state.status = TaskStatus.paused
            state.message = "Paused by user"
            state.updated_at = datetime.utcnow()
            self._persist_task_locked(state)
            info = self._to_info(state)

        if runtime:
            runtime.pause()
        self._emit_update(info)
        return info

    def resume_task(self, task_id: str) -> LabelingTaskInfo:
        with self.lock:
            state = self._require_task(task_id)
            runtime = self.runtimes.get(task_id)

            if runtime and state.status == TaskStatus.paused:
                runtime.resume()
                state.status = TaskStatus.running
                state.message = "Resuming"
                state.updated_at = datetime.utcnow()
                self._persist_task_locked(state)
                info = self._to_info(state)
                self._emit_update(info)
                return info

            if state.status in {TaskStatus.completed, TaskStatus.running}:
                return self._to_info(state)

            state.status = TaskStatus.queued
            state.message = "Queued"
            state.updated_at = datetime.utcnow()
            self._persist_task_locked(state)
            info = self._to_info(state)

        self._submit(task_id)
        return info

    def cancel_task(self, task_id: str) -> LabelingTaskInfo:
        with self.lock:
            state = self._require_task(task_id)
            runtime = self.runtimes.get(task_id)
            state.status = TaskStatus.cancelled
            state.message = "Cancelled by user"
            state.finished_at = datetime.utcnow()
            state.updated_at = datetime.utcnow()
            self._persist_task_locked(state)
            info = self._to_info(state)

        if runtime:
            runtime.cancel()
        self._emit_update(info)
        return info

    def restart_task(self, task_id: str) -> LabelingTaskInfo:
        with self.lock:
            state = self._require_task(task_id)
            state.status = TaskStatus.queued
            state.progress = 0.0
            state.processed_rows = 0
            state.total_rows = 0
            state.started_at = None
            state.finished_at = None
            state.message = "Restarted"
            state.updated_at = datetime.utcnow()
            self._persist_task_locked(state)
            info = self._to_info(state)

        self._submit(task_id)
        return info

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _submit(self, task_id: str) -> None:
        self.executor.submit(self._run_task, task_id)

    def _run_task(self, task_id: str) -> None:
        runtime = TaskRuntime()
        with self.lock:
            state = self._require_task(task_id)
            self.runtimes[task_id] = runtime
            state.status = TaskStatus.running
            state.started_at = state.started_at or datetime.utcnow()
            state.message = "Running"
            state.updated_at = datetime.utcnow()
            self._persist_task_locked(state)
            info = self._to_info(state)
        self._emit_update(info)

        try:
            # Load dataset
            data = self._load_dataset(state.dataset_id)
            
            with self.lock:
                current = self.tasks.get(task_id)
                if current:
                    current.total_rows = len(data)
                    current.message = "Loaded dataset"
                    current.updated_at = datetime.utcnow()
                    self._persist_task_locked(current)
                    self._emit_update(self._to_info(current))

            # Check if cancelled
            if runtime.is_cancelled():
                raise RuntimeError("Task cancelled")

            # Create labeler
            labeler = self._create_labeler(state.method, state.config)

            # Label data
            with self.lock:
                current = self.tasks.get(task_id)
                if current:
                    current.message = "Labeling data..."
                    current.updated_at = datetime.utcnow()
                    self._persist_task_locked(current)
                    self._emit_update(self._to_info(current))

            labeled_data = labeler.label(data)

            # Apply filters if configured
            if state.config.get("filters"):
                labeled_data = self._apply_filters(labeled_data, state.config["filters"])

            # Save result
            labeling_info = self.labeling_service.save_labeling_result(
                labeling_set_id=state.labeling_set_id,
                name=state.name,
                dataset_id=state.dataset_id,
                method=state.method,
                config=state.config,
                labels_df=labeled_data,
                description=state.description,
                feature_set_id=state.feature_set_id,
            )

            with self.lock:
                current = self._require_task(task_id)
                current.status = TaskStatus.completed
                current.progress = 1.0
                current.processed_rows = len(labeled_data)
                current.total_rows = len(labeled_data)
                current.message = "Completed"
                current.finished_at = datetime.utcnow()
                current.updated_at = datetime.utcnow()
                current.class_distribution = labeling_info.class_distribution
                self._persist_task_locked(current)
                info_final = self._to_info(current)
            self._emit_update(info_final)

        except RuntimeError as exc:
            reason = str(exc)
            if "cancelled" in reason.lower():
                self._mark_cancelled(task_id, "Cancelled")
            else:
                self._mark_failed(task_id, reason)
        except Exception as exc:
            import traceback
            error_msg = f"{str(exc)}\n{traceback.format_exc()}"
            self._mark_failed(task_id, error_msg)
        finally:
            with self.lock:
                self.runtimes.pop(task_id, None)

    def _load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load dataset from storage"""
        try:
            df = self.labeling_service.dataset_service.get_dataset_data(dataset_id=dataset_id, limit=-1)
        except Exception as exc:
            raise ValueError(f"Failed to load dataset {dataset_id}: {exc}") from exc

        if df.empty:
            raise ValueError(f"Dataset {dataset_id} is empty")

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        elif isinstance(df.index, pd.DatetimeIndex):
            pass
        else:
            raise ValueError("Dataset must contain a 'timestamp' column")

        return df

    def _create_labeler(self, method: str, config: Dict[str, Any]):
        """Create labeler instance based on method and config"""
        if method == "horizon":
            return self._create_horizon_labeler(config)
        elif method == "triple_barrier":
            return self._create_triple_barrier_labeler(config)
        elif method == "regression":
            return self._create_regression_labeler(config)
        else:
            raise ValueError(f"Unknown labeling method: {method}")

    def _create_horizon_labeler(self, config: Dict[str, Any]) -> HorizonLabeler:
        """Create Horizon labeler from config"""
        params = {
            "horizon": config.get("horizon", 20),
            "direction": config.get("direction", "long+short"),
            "threshold_pct": config.get("threshold_pct", 0.01),
        }

        if config.get("adaptive", False):
            params["horizon"] = "adaptive"
            params["adaptive_method"] = "atr"

        return HorizonLabeler(**params)

    def _create_triple_barrier_labeler(self, config: Dict[str, Any]) -> TripleBarrierLabeler:
        """Create Triple Barrier labeler from config"""
        upper = config.get("upper_barrier", {})
        lower = config.get("lower_barrier", {})

        params = {
            "upper_barrier": upper.get("value", 0.02) if isinstance(upper, dict) else upper,
            "lower_barrier": lower.get("value", 0.02) if isinstance(lower, dict) else lower,
            "time_barrier": config.get("time_barrier", 20),
            "direction": config.get("direction", "long+short"),
            "min_return": config.get("min_return", 0.0),
        }

        # Handle ATR-based barriers
        if isinstance(upper, dict) and upper.get("type") == "atr":
            params["upper_barrier"] = "atr"
            params["atr_multiplier"] = upper.get("value", 2.0)

        if isinstance(lower, dict) and lower.get("type") == "atr":
            params["lower_barrier"] = "atr"

        # Commission handling
        if config.get("commission_rate"):
            params["include_commissions"] = True
            params["commission_pct"] = config["commission_rate"]

        return TripleBarrierLabeler(**params)

    def _create_regression_labeler(self, config: Dict[str, Any]) -> RegressionTargetsLabeler:
        """Create Regression labeler from config"""
        params = {
            "target_type": config.get("target", "future_return"),
            "horizon": config.get("horizon", 20),
        }

        return RegressionTargetsLabeler(**params)

    def _apply_filters(self, data: pd.DataFrame, filters_config: List[Dict[str, Any]]) -> pd.DataFrame:
        """Apply post-filters to labeled data"""
        # Import filters
        from src.labeling.filters.smoothing import SmoothingFilter
        from src.labeling.filters.sequence_filter import SequenceFilter
        from src.labeling.filters.majority_vote import MajorityVoteFilter
        from src.labeling.filters.danger_zones import DangerZonesFilter

        result = data.copy()
        labels = result["label"].copy()

        for filter_config in filters_config:
            filter_type = filter_config.get("type")
            filter_params = filter_config.get("params", {})

            if filter_type == "smoothing":
                smoother = SmoothingFilter(**filter_params)
                labels = smoother.apply(labels)
            elif filter_type == "sequence":
                seq_filter = SequenceFilter(**filter_params)
                labels = seq_filter.apply(labels)
            elif filter_type == "majority_vote":
                mv_filter = MajorityVoteFilter(**filter_params)
                labels = mv_filter.apply(labels)
            elif filter_type == "danger_zones":
                dz_filter = DangerZonesFilter(**filter_params)
                labels = dz_filter.apply(labels=labels, data=result)

        result["label"] = labels
        return result

    def _mark_failed(self, task_id: str, message: str) -> None:
        with self.lock:
            state = self.tasks.get(task_id)
            if not state:
                return
            state.status = TaskStatus.failed
            state.message = message[:500]  # Limit message length
            state.finished_at = datetime.utcnow()
            state.updated_at = datetime.utcnow()
            self._persist_task_locked(state)
            info = self._to_info(state)
        self._emit_update(info)

    def _mark_cancelled(self, task_id: str, message: str) -> None:
        with self.lock:
            state = self.tasks.get(task_id)
            if not state:
                return
            state.status = TaskStatus.cancelled
            state.message = message
            state.finished_at = datetime.utcnow()
            state.updated_at = datetime.utcnow()
            self._persist_task_locked(state)
            info = self._to_info(state)
        self._emit_update(info)

    def _to_info(self, state: LabelingTaskState) -> LabelingTaskInfo:
        return LabelingTaskInfo(
            id=state.id,
            name=state.name,
            dataset_id=state.dataset_id,
            labeling_set_id=state.labeling_set_id,
            status=state.status,
            progress=state.progress,
            processed_rows=state.processed_rows,
            total_rows=state.total_rows,
            config=state.config,
            message=state.message,
            created_at=state.created_at,
            updated_at=state.updated_at,
            started_at=state.started_at,
            finished_at=state.finished_at,
            method=state.method,
            class_distribution=state.class_distribution,
            feature_set_id=state.feature_set_id,
        )

    def _task_file_path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.json"

    def _persist_task_locked(self, state: LabelingTaskState) -> None:
        path = self._task_file_path(state.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)

    def _load_tasks(self) -> None:
        for task_file in sorted(self.tasks_dir.glob("*.json")):
            try:
                with open(task_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                state = LabelingTaskState.from_dict(payload)
            except Exception as exc:
                print(f"Failed to load labeling task state from {task_file}: {exc}")
                continue

            if state.status == TaskStatus.running:
                state.status = TaskStatus.paused
                state.message = "Paused after restart"
            self.tasks[state.id] = state

    def _emit_update(self, info: LabelingTaskInfo) -> None:
        update = TaskUpdate(
            task_id=info.id,
            status=info.status,
            progress=info.progress,
            message=info.message,
        )

        try:
            asyncio.run(ws_manager.send_task_update(info.id, update))
        except RuntimeError:
            pass

        try:
            asyncio.run(ws_manager.broadcast({"type": "labeling_task", "data": info.dict()}))
        except RuntimeError:
            pass

    def _require_task(self, task_id: str) -> LabelingTaskState:
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        return task

