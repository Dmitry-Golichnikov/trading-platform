"""
Experiments Router

API endpoints for experiment operations.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.interfaces.gui.backend.api.models import (
    ExperimentComparison,
    ExperimentCreateRequest,
    ExperimentInfo,
    ExperimentStatusResponse,
)
from src.interfaces.gui.backend.services.experiment_service import ExperimentService

router = APIRouter(prefix="/api/experiments", tags=["experiments"])
experiment_service = ExperimentService()


@router.get("", response_model=List[ExperimentInfo])
async def list_experiments(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, description="Max experiments to return"),
):
    """List all experiments"""
    try:
        return experiment_service.list_experiments(status=status, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}", response_model=ExperimentInfo)
async def get_experiment(experiment_id: str):
    """Get experiment information"""
    experiment = experiment_service.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")
    return experiment


@router.post("", response_model=ExperimentInfo)
async def create_experiment(request: ExperimentCreateRequest):
    """Create a new experiment"""
    try:
        return experiment_service.create_experiment(
            name=request.name, config=request.config, description=request.description, tags=request.tags
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}/status", response_model=ExperimentStatusResponse)
async def get_experiment_status(experiment_id: str):
    """Get experiment status"""
    try:
        return experiment_service.get_experiment_status(experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{experiment_id}/cancel")
async def cancel_experiment(experiment_id: str):
    """Cancel a running experiment"""
    try:
        success = experiment_service.cancel_experiment(experiment_id)
        if success:
            return {"status": "cancelled", "experiment_id": experiment_id}
        else:
            raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Delete an experiment"""
    try:
        success = experiment_service.delete_experiment(experiment_id)
        if success:
            return {"status": "deleted", "experiment_id": experiment_id}
        else:
            raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=ExperimentComparison)
async def compare_experiments(experiment_ids: List[str]):
    """Compare multiple experiments"""
    try:
        return experiment_service.compare_experiments(experiment_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
