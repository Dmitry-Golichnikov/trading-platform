"""
Models Router

API endpoints for model operations.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.interfaces.gui.backend.api.models import ModelInfo, ModelMetrics
from src.interfaces.gui.backend.services.model_service import ModelService

router = APIRouter(prefix="/api/models", tags=["models"])
model_service = ModelService()


@router.get("", response_model=List[ModelInfo])
async def list_models(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    experiment_id: Optional[str] = Query(None, description="Filter by experiment"),
    limit: int = Query(100, description="Max models to return"),
):
    """List all models"""
    try:
        return model_service.list_models(model_type=model_type, experiment_id=experiment_id, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """Get model information"""
    model = model_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return model


@router.get("/{model_id}/metrics", response_model=ModelMetrics)
async def get_model_metrics(model_id: str):
    """Get detailed model metrics"""
    metrics = model_service.get_model_metrics(model_id)
    if not metrics:
        raise HTTPException(status_code=404, detail=f"Model metrics not found: {model_id}")
    return metrics


@router.post("/{model_id}/deploy")
async def deploy_model(model_id: str):
    """Deploy a model"""
    try:
        success = model_service.deploy_model(model_id)
        if success:
            return {"status": "deployed", "model_id": model_id}
        else:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Delete a model"""
    try:
        success = model_service.delete_model(model_id)
        if success:
            return {"status": "deleted", "model_id": model_id}
        else:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
