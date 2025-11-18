"""
WebSocket Router

WebSocket endpoints for real-time updates.
"""

import json
from datetime import datetime
from typing import Dict, Optional, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder

from src.interfaces.gui.backend.api.models import TaskUpdate, TrainingUpdate

router = APIRouter()


# Connection manager for WebSocket connections
class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        # experiment_id -> set of websockets
        self.training_connections: Dict[str, Set[WebSocket]] = {}
        # task_id -> set of websockets
        self.task_connections: Dict[str, Set[WebSocket]] = {}
        # all connections
        self.all_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket, channel: str, channel_id: Optional[str] = None):
        """Connect a websocket"""
        await websocket.accept()
        self.all_connections.add(websocket)

        if channel == "training" and channel_id:
            if channel_id not in self.training_connections:
                self.training_connections[channel_id] = set()
            self.training_connections[channel_id].add(websocket)
        elif channel == "task" and channel_id:
            if channel_id not in self.task_connections:
                self.task_connections[channel_id] = set()
            self.task_connections[channel_id].add(websocket)

    def disconnect(self, websocket: WebSocket, channel: str, channel_id: Optional[str] = None):
        """Disconnect a websocket"""
        self.all_connections.discard(websocket)

        if channel == "training" and channel_id:
            if channel_id in self.training_connections:
                self.training_connections[channel_id].discard(websocket)
                if not self.training_connections[channel_id]:
                    del self.training_connections[channel_id]
        elif channel == "task" and channel_id:
            if channel_id in self.task_connections:
                self.task_connections[channel_id].discard(websocket)
                if not self.task_connections[channel_id]:
                    del self.task_connections[channel_id]

    async def send_training_update(self, experiment_id: str, update: TrainingUpdate):
        """Send update to all connections for an experiment"""
        if experiment_id in self.training_connections:
            payload = jsonable_encoder({"type": "training_update", "data": update.dict()})
            message = json.dumps(payload)

            disconnected = set()
            for websocket in self.training_connections[experiment_id]:
                try:
                    await websocket.send_text(message)
                except Exception:
                    disconnected.add(websocket)

            # Remove disconnected websockets
            for ws in disconnected:
                self.disconnect(ws, "training", experiment_id)

    async def send_task_update(self, task_id: str, update: TaskUpdate):
        """Send update to all connections for a task"""
        if task_id in self.task_connections:
            payload = jsonable_encoder({"type": "task_update", "data": update.dict()})
            message = json.dumps(payload)

            disconnected = set()
            for websocket in self.task_connections[task_id]:
                try:
                    await websocket.send_text(message)
                except Exception:
                    disconnected.add(websocket)

            # Remove disconnected websockets
            for ws in disconnected:
                self.disconnect(ws, "task", task_id)

    async def broadcast(self, message: dict):
        """Broadcast message to all connections"""
        payload = jsonable_encoder(message)
        message_str = json.dumps(payload)

        disconnected = set()
        for websocket in self.all_connections:
            try:
                await websocket.send_text(message_str)
            except Exception:
                disconnected.add(websocket)

        # Remove disconnected websockets
        for ws in disconnected:
            self.all_connections.discard(ws)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/training/{experiment_id}")
async def training_websocket(websocket: WebSocket, experiment_id: str):
    """WebSocket endpoint for training updates"""
    await manager.connect(websocket, "training", experiment_id)

    try:
        # Send initial connection message
        await websocket.send_json(
            {"type": "connected", "data": {"experiment_id": experiment_id, "timestamp": datetime.now().isoformat()}}
        )

        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            # Echo back for now (can be used for control messages)
            await websocket.send_json({"type": "echo", "data": json.loads(data)})
    except WebSocketDisconnect:
        manager.disconnect(websocket, "training", experiment_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, "training", experiment_id)


@router.websocket("/ws/task/{task_id}")
async def task_websocket(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for task updates"""
    await manager.connect(websocket, "task", task_id)

    try:
        # Send initial connection message
        await websocket.send_json(
            {"type": "connected", "data": {"task_id": task_id, "timestamp": datetime.now().isoformat()}}
        )

        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "echo", "data": json.loads(data)})
    except WebSocketDisconnect:
        manager.disconnect(websocket, "task", task_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, "task", task_id)


@router.websocket("/ws/global")
async def global_websocket(websocket: WebSocket):
    """WebSocket endpoint for global updates"""
    await manager.connect(websocket, "global")

    try:
        # Send initial connection message
        await websocket.send_json({"type": "connected", "data": {"timestamp": datetime.now().isoformat()}})

        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Broadcast to all connections
            await manager.broadcast({"type": "broadcast", "data": json.loads(data)})
    except WebSocketDisconnect:
        manager.disconnect(websocket, "global")
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, "global")


# Export manager for use in other modules
def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager"""
    return manager
