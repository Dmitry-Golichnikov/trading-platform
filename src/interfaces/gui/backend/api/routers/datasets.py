"""
Datasets Router

API endpoints for dataset operations.
"""

from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from src.interfaces.gui.backend.api.models import DatasetInfo, DatasetQualityReport
from src.interfaces.gui.backend.services.dataset_service import DatasetService

router = APIRouter(prefix="/api/datasets", tags=["datasets"])
dataset_service = DatasetService()


@router.get("", response_model=List[DatasetInfo])
async def list_datasets(
    ticker: Optional[str] = Query(None, description="Filter by ticker"),
    timeframe: Optional[str] = Query(None, description="Filter by timeframe"),
):
    """List all available datasets"""
    try:
        return dataset_service.list_datasets(ticker=ticker, timeframe=timeframe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}", response_model=DatasetInfo)
async def get_dataset(dataset_id: str):
    """Get dataset information"""
    dataset = dataset_service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    return dataset


@router.get("/{dataset_id}/data")
async def get_dataset_data(
    dataset_id: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(-1, description="Max rows to return (-1 for all)"),
):
    """Get dataset data"""
    try:
        df = dataset_service.get_dataset_data(dataset_id, start_date=start_date, end_date=end_date, limit=limit)

        # Convert to JSON-serializable format
        df_reset = df.reset_index()

        # Convert timestamps to ISO format strings
        if "timestamp" in df_reset.columns:
            df_reset["timestamp"] = df_reset["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

        data = df_reset.to_dict(orient="records")

        return JSONResponse(
            content={"dataset_id": dataset_id, "num_rows": len(data), "columns": list(df.columns), "data": data}
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/quality", response_model=DatasetQualityReport)
async def get_dataset_quality(dataset_id: str):
    """Get dataset quality report"""
    try:
        return dataset_service.generate_quality_report(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    ticker: str = Query(..., description="Ticker symbol"),
    timeframe: str = Query(..., description="Timeframe"),
    source: str = Query("local", description="Data source"),
):
    """Upload a new dataset"""
    # TODO: Implement dataset upload
    # For now, return a placeholder response
    return {
        "status": "uploaded",
        "dataset_id": f"{ticker}_{timeframe}",
        "message": "Dataset upload not yet implemented",
    }


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    try:
        success = dataset_service.delete_dataset(dataset_id)
        if success:
            return {"status": "deleted", "dataset_id": dataset_id}
        else:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
