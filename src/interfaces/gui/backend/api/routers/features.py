"""
Features Router

API endpoints for feature generation operations.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from src.interfaces.gui.backend.api.models import (
    FeatureGenerationRequest,
    FeatureSetInfo,
    FeatureTaskActionResponse,
    FeatureTaskCreateRequest,
    FeatureTaskInfo,
    TaskStatus,
)
from src.interfaces.gui.backend.services.feature_service import FeatureService
from src.interfaces.gui.backend.services.feature_task_service import FeatureTaskManager

router = APIRouter(prefix="/api/features", tags=["features"])
feature_service = FeatureService()
task_manager = FeatureTaskManager(feature_service, max_workers=2)


@router.get("", response_model=List[FeatureSetInfo])
async def list_feature_sets(dataset_id: Optional[str] = Query(None, description="Filter by dataset")):
    """List all feature sets"""
    try:
        return feature_service.list_feature_sets(dataset_id=dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks", response_model=List[FeatureTaskInfo])
async def list_feature_tasks(
    status: Optional[TaskStatus] = Query(None, description="Filter by status"),
    dataset_id: Optional[str] = Query(None, description="Filter by dataset"),
):
    try:
        return task_manager.list_tasks(status=status, dataset_id=dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}", response_model=FeatureTaskInfo)
async def get_feature_task(task_id: str):
    try:
        return task_manager.get_task(task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks", response_model=List[FeatureTaskInfo])
async def create_feature_tasks(request: FeatureTaskCreateRequest):
    try:
        return task_manager.create_tasks(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/pause", response_model=FeatureTaskActionResponse)
async def pause_feature_task(task_id: str):
    try:
        task = task_manager.pause_task(task_id)
        return FeatureTaskActionResponse(task=task, message="Task paused")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/resume", response_model=FeatureTaskActionResponse)
async def resume_feature_task(task_id: str):
    try:
        task = task_manager.resume_task(task_id)
        return FeatureTaskActionResponse(task=task, message="Task resumed")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/cancel", response_model=FeatureTaskActionResponse)
async def cancel_feature_task(task_id: str):
    try:
        task = task_manager.cancel_task(task_id)
        return FeatureTaskActionResponse(task=task, message="Task cancelled")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/restart", response_model=FeatureTaskActionResponse)
async def restart_feature_task(task_id: str):
    try:
        task = task_manager.restart_task(task_id)
        return FeatureTaskActionResponse(task=task, message="Task restarted")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{feature_set_id}", response_model=FeatureSetInfo)
async def get_feature_set(feature_set_id: str):
    """Get feature set information"""
    feature_set = feature_service.get_feature_set(feature_set_id)
    if not feature_set:
        raise HTTPException(status_code=404, detail=f"Feature set not found: {feature_set_id}")
    return feature_set


@router.get("/{feature_set_id}/data")
async def get_feature_data(
    feature_set_id: str,
    columns: Optional[List[str]] = Query(None, description="Columns to return"),
    limit: int = Query(1000, description="Max rows to return"),
):
    """Get feature data"""
    try:
        data = feature_service.get_feature_data(feature_set_id, columns=columns, limit=limit)
        return JSONResponse(content=data)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def generate_features(request: FeatureGenerationRequest):
    """Generate features from dataset"""
    try:
        feature_set_id = feature_service.generate_features(request)
        return {"status": "completed", "feature_set_id": feature_set_id, "message": "Features generated successfully"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{feature_set_id}")
async def delete_feature_set(feature_set_id: str):
    """Delete a feature set"""
    try:
        success = feature_service.delete_feature_set(feature_set_id)
        if success:
            return {"status": "deleted", "feature_set_id": feature_set_id}
        else:
            raise HTTPException(status_code=404, detail=f"Feature set not found: {feature_set_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
