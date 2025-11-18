"""
Labeling Router

API endpoints for labeling operations.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from src.interfaces.gui.backend.api.models import (
    LabelingMethod,
    LabelingSetInfo,
    LabelingTaskActionResponse,
    LabelingTaskCreateRequest,
    LabelingTaskInfo,
    TaskStatus,
)
from src.interfaces.gui.backend.services.labeling_service import LabelingService
from src.interfaces.gui.backend.services.labeling_task_service import LabelingTaskManager

router = APIRouter(prefix="/api/labeling", tags=["labeling"])
labeling_service = LabelingService()
task_manager = LabelingTaskManager(labeling_service, max_workers=2)


@router.get("", response_model=List[LabelingSetInfo])
async def list_labeling_sets(
    dataset_id: Optional[str] = Query(None, description="Filter by dataset"),
    method: Optional[LabelingMethod] = Query(None, description="Filter by method"),
):
    """List all labeling sets"""
    try:
        return labeling_service.list_labeling_sets(dataset_id=dataset_id, method=method)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks", response_model=List[LabelingTaskInfo])
async def list_labeling_tasks(
    status: Optional[TaskStatus] = Query(None, description="Filter by status"),
    dataset_id: Optional[str] = Query(None, description="Filter by dataset"),
):
    """List all labeling tasks"""
    try:
        return task_manager.list_tasks(status=status, dataset_id=dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}", response_model=LabelingTaskInfo)
async def get_labeling_task(task_id: str):
    """Get labeling task by ID"""
    try:
        return task_manager.get_task(task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks", response_model=List[LabelingTaskInfo])
async def create_labeling_tasks(request: LabelingTaskCreateRequest):
    """Create labeling tasks"""
    try:
        return task_manager.create_tasks(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/pause", response_model=LabelingTaskActionResponse)
async def pause_labeling_task(task_id: str):
    """Pause labeling task"""
    try:
        task = task_manager.pause_task(task_id)
        return LabelingTaskActionResponse(task=task, message="Task paused")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/resume", response_model=LabelingTaskActionResponse)
async def resume_labeling_task(task_id: str):
    """Resume labeling task"""
    try:
        task = task_manager.resume_task(task_id)
        return LabelingTaskActionResponse(task=task, message="Task resumed")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/cancel", response_model=LabelingTaskActionResponse)
async def cancel_labeling_task(task_id: str):
    """Cancel labeling task"""
    try:
        task = task_manager.cancel_task(task_id)
        return LabelingTaskActionResponse(task=task, message="Task cancelled")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/restart", response_model=LabelingTaskActionResponse)
async def restart_labeling_task(task_id: str):
    """Restart labeling task"""
    try:
        task = task_manager.restart_task(task_id)
        return LabelingTaskActionResponse(task=task, message="Task restarted")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{labeling_set_id}", response_model=LabelingSetInfo)
async def get_labeling_set(labeling_set_id: str):
    """Get labeling set information"""
    labeling_set = labeling_service.get_labeling_set(labeling_set_id)
    if not labeling_set:
        raise HTTPException(status_code=404, detail=f"Labeling set not found: {labeling_set_id}")
    return labeling_set


@router.get("/{labeling_set_id}/data")
async def get_labeling_data(
    labeling_set_id: str,
    limit: int = Query(1000, description="Max rows to return"),
):
    """Get labeling data"""
    try:
        data = labeling_service.get_labeling_data(labeling_set_id, limit=limit)
        return JSONResponse(content=data)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{labeling_set_id}")
async def delete_labeling_set(labeling_set_id: str):
    """Delete a labeling set"""
    try:
        success = labeling_service.delete_labeling_set(labeling_set_id)
        if success:
            return {"status": "deleted", "labeling_set_id": labeling_set_id}
        else:
            raise HTTPException(status_code=404, detail=f"Labeling set not found: {labeling_set_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{labeling_set_id}/visualize")
async def visualize_labeling(
    labeling_set_id: str,
    chart_type: str = Query("distribution", description="Chart type"),
):
    """Get visualization data for labeling set"""
    try:
        labeling_set = labeling_service.get_labeling_set(labeling_set_id)
        if not labeling_set:
            raise HTTPException(status_code=404, detail=f"Labeling set not found: {labeling_set_id}")

        if chart_type == "distribution":
            return {
                "type": "distribution",
                "data": labeling_set.class_distribution or {},
                "num_samples": labeling_set.num_samples,
            }

        return {"type": chart_type, "data": {}, "message": "Chart type not implemented"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

