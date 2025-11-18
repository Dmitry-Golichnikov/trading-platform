"""
Feature Task Manager Service

Handles long-running feature generation tasks with pause/resume support.
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

from src.interfaces.gui.backend.api.models import (
    FeatureTaskCreateRequest,
    FeatureTaskInfo,
    TaskStatus,
    TaskUpdate,
)
from src.interfaces.gui.backend.api.routers.websocket import manager as ws_manager
from src.interfaces.gui.backend.services.feature_service import FeatureService


@dataclass
class FeatureTaskState:
    """Internal representation of feature generation task"""

    id: str
    name: str
    dataset_id: str
    feature_set_id: str
    config: Dict[str, Any]
    chunk_size: int
    incremental: bool
    status: TaskStatus
    progress: float
    processed_rows: int
    total_rows: int
    message: Optional[str]
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    last_processed_timestamp: Optional[str] = None
    dataset_hash: Optional[str] = None
    description: Optional[str] = None
    apply_to_all: bool = False
    batch_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "dataset_id": self.dataset_id,
            "feature_set_id": self.feature_set_id,
            "config": self.config,
            "chunk_size": self.chunk_size,
            "incremental": self.incremental,
            "status": self.status.value,
            "progress": self.progress,
            "processed_rows": self.processed_rows,
            "total_rows": self.total_rows,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "last_processed_timestamp": self.last_processed_timestamp,
            "dataset_hash": self.dataset_hash,
            "description": self.description,
            "apply_to_all": self.apply_to_all,
            "batch_id": self.batch_id,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "FeatureTaskState":
        def _parse_dt(value: Optional[str]) -> Optional[datetime]:
            return datetime.fromisoformat(value) if value else None

        return FeatureTaskState(
            id=payload["id"],
            name=payload["name"],
            dataset_id=payload["dataset_id"],
            feature_set_id=payload["feature_set_id"],
            config=payload.get("config", {}),
            chunk_size=payload.get("chunk_size", 5000),
            incremental=payload.get("incremental", True),
            status=TaskStatus(payload.get("status", "queued")),
            progress=payload.get("progress", 0.0),
            processed_rows=payload.get("processed_rows", 0),
            total_rows=payload.get("total_rows", 0),
            message=payload.get("message"),
            created_at=_parse_dt(payload.get("created_at")) or datetime.utcnow(),
            updated_at=_parse_dt(payload.get("updated_at")) or datetime.utcnow(),
            started_at=_parse_dt(payload.get("started_at")),
            finished_at=_parse_dt(payload.get("finished_at")),
            last_processed_timestamp=payload.get("last_processed_timestamp"),
            dataset_hash=payload.get("dataset_hash"),
            description=payload.get("description"),
            apply_to_all=payload.get("apply_to_all", False),
            batch_id=payload.get("batch_id"),
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


class FeatureTaskManager:
    """Manager for long-running feature generation tasks"""

    def __init__(self, feature_service: FeatureService, max_workers: int = 1):
        self.feature_service = feature_service
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, FeatureTaskState] = {}
        self.runtimes: Dict[str, TaskRuntime] = {}
        self.lock = threading.Lock()
        self.tasks_file = self.feature_service.features_dir / "tasks.json"
        self.tasks_dir = self.feature_service.features_dir / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_tasks_file()
        self._load_tasks()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        dataset_id: Optional[str] = None,
    ) -> List[FeatureTaskInfo]:
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

    def get_task(self, task_id: str) -> FeatureTaskInfo:
        with self.lock:
            state = self.tasks.get(task_id)
            if not state:
                raise ValueError(f"Task {task_id} not found")
            return self._to_info(state)

    def create_tasks(self, request: FeatureTaskCreateRequest) -> List[FeatureTaskInfo]:
        dataset_ids = request.dataset_ids or []
        if request.apply_to_all or not dataset_ids:
            available = [ds.id for ds in self.feature_service.dataset_service.list_datasets()]
            if dataset_ids:
                dataset_ids = [ds for ds in available if ds in dataset_ids]
            else:
                dataset_ids = available

        if not dataset_ids:
            raise ValueError("Нет доступных датасетов для генерации признаков")

        batch_id = str(uuid.uuid4()) if len(dataset_ids) > 1 else None
        created: List[FeatureTaskInfo] = []

        for dataset_id in dataset_ids:
            task_state = FeatureTaskState(
                id=str(uuid.uuid4()),
                name=request.name,
                dataset_id=dataset_id,
                feature_set_id=self.feature_service.build_feature_set_id(dataset_id, request.name),
                config=request.config,
                chunk_size=request.chunk_size,
                incremental=request.incremental,
                status=TaskStatus.queued,
                progress=0.0,
                processed_rows=0,
                total_rows=0,
                message="Queued",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                description=request.description,
                apply_to_all=request.apply_to_all,
                batch_id=batch_id,
            )

            with self.lock:
                self.tasks[task_state.id] = task_state
                self._persist_task_locked(task_state)
                info = self._to_info(task_state)
            created.append(info)

            if request.auto_start:
                self._submit(task_state.id)

        return created

    def pause_task(self, task_id: str) -> FeatureTaskInfo:
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

    def resume_task(self, task_id: str) -> FeatureTaskInfo:
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

    def cancel_task(self, task_id: str) -> FeatureTaskInfo:
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

    def restart_task(self, task_id: str) -> FeatureTaskInfo:
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

        def progress_callback(processed: int, total: int, message: str, last_ts: Optional[datetime]) -> None:
            with self.lock:
                current = self.tasks.get(task_id)
                if not current:
                    return
                current.processed_rows = processed
                current.total_rows = total
                current.progress = min(1.0, processed / total) if total else 0.0
                current.message = message
                if last_ts:
                    current.last_processed_timestamp = last_ts.isoformat()
                current.updated_at = datetime.utcnow()
                self._persist_task_locked(current)
                info_local = self._to_info(current)
            self._emit_update(info_local)

        try:
            feature_info = self.feature_service.generate_features_incremental(
                dataset_id=state.dataset_id,
                feature_set_name=state.name,
                config=state.config,
                chunk_size=state.chunk_size,
                incremental=state.incremental,
                progress_callback=progress_callback,
                wait_if_paused=runtime.wait_if_paused,
                is_cancelled=runtime.is_cancelled,
            )
            with self.lock:
                current = self._require_task(task_id)
                current.status = TaskStatus.completed
                current.progress = 1.0
                current.message = "Completed"
                current.finished_at = datetime.utcnow()
                current.updated_at = datetime.utcnow()
                current.dataset_hash = feature_info.dataset_hash
                if feature_info.last_processed_timestamp:
                    current.last_processed_timestamp = feature_info.last_processed_timestamp.isoformat()
                current.total_rows = max(current.total_rows, current.processed_rows)
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
            self._mark_failed(task_id, str(exc))
        finally:
            with self.lock:
                self.runtimes.pop(task_id, None)

    def _mark_failed(self, task_id: str, message: str) -> None:
        with self.lock:
            state = self.tasks.get(task_id)
            if not state:
                return
            state.status = TaskStatus.failed
            state.message = message
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

    def _to_info(self, state: FeatureTaskState) -> FeatureTaskInfo:
        last_ts = datetime.fromisoformat(state.last_processed_timestamp) if state.last_processed_timestamp else None
        return FeatureTaskInfo(
            id=state.id,
            name=state.name,
            dataset_id=state.dataset_id,
            feature_set_id=state.feature_set_id,
            status=state.status,
            progress=state.progress,
            processed_rows=state.processed_rows,
            total_rows=state.total_rows,
            config=state.config,
            config_hash=self.feature_service._calculate_config_hash(state.config),
            message=state.message,
            created_at=state.created_at,
            updated_at=state.updated_at,
            started_at=state.started_at,
            finished_at=state.finished_at,
            last_processed_timestamp=last_ts,
            dataset_hash=state.dataset_hash,
            description=state.description,
            apply_to_all=state.apply_to_all,
            batch_id=state.batch_id,
        )

    def _task_file_path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.json"

    def _persist_task_locked(self, state: FeatureTaskState) -> None:
        path = self._task_file_path(state.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)

    def _migrate_legacy_tasks_file(self) -> None:
        if not self.tasks_file.exists():
            return

        try:
            with open(self.tasks_file, "r", encoding="utf-8") as f:
                raw_data = f.read()
        except Exception as exc:
            print(f"Failed to read legacy tasks.json: {exc}")
            return

        decoder = json.JSONDecoder()
        idx = 0
        migrated = 0
        while idx < len(raw_data):
            char = raw_data[idx]
            if char in " \t\r\n,":
                idx += 1
                continue
            if char == "[":
                idx += 1
                continue
            if char == "]":
                break
            try:
                entry, offset = decoder.raw_decode(raw_data, idx)
            except json.JSONDecodeError as exc:
                print(f"Stopped migrating tasks.json at entry {migrated}: {exc}")
                break

            try:
                state = FeatureTaskState.from_dict(entry)
            except Exception as exc:
                print(f"Failed to migrate task entry: {exc}")
                idx = offset
                continue

            path = self._task_file_path(state.id)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2)
            migrated += 1
            idx = offset

        backup_path = self.tasks_file.with_suffix(".legacy.json")
        try:
            self.tasks_file.rename(backup_path)
        except Exception:
            pass

    def _load_tasks(self) -> None:
        for task_file in sorted(self.tasks_dir.glob("*.json")):
            try:
                with open(task_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                state = FeatureTaskState.from_dict(payload)
            except Exception as exc:
                print(f"Failed to load task state from {task_file}: {exc}")
                continue

            if state.status == TaskStatus.running:
                state.status = TaskStatus.paused
                state.message = "Paused after restart"
            self.tasks[state.id] = state

    def _emit_update(self, info: FeatureTaskInfo) -> None:
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
            asyncio.run(ws_manager.broadcast({"type": "feature_task", "data": info.dict()}))
        except RuntimeError:
            pass

    def _require_task(self, task_id: str) -> FeatureTaskState:
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        return task
