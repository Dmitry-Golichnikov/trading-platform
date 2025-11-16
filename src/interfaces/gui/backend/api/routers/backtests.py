"""
Backtests Router

API endpoints for backtest operations.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.interfaces.gui.backend.api.models import BacktestResult, BacktestRunRequest
from src.interfaces.gui.backend.services.backtest_service import BacktestService

router = APIRouter(prefix="/api/backtests", tags=["backtests"])
backtest_service = BacktestService()


@router.get("", response_model=List[BacktestResult])
async def list_backtests(
    model_id: Optional[str] = Query(None, description="Filter by model"),
    limit: int = Query(100, description="Max backtests to return"),
):
    """List all backtest results"""
    try:
        return backtest_service.list_backtests(model_id=model_id, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{backtest_id}", response_model=BacktestResult)
async def get_backtest(backtest_id: str):
    """Get backtest result"""
    backtest = backtest_service.get_backtest(backtest_id)
    if not backtest:
        raise HTTPException(status_code=404, detail=f"Backtest not found: {backtest_id}")
    return backtest


@router.post("/run")
async def run_backtest(request: BacktestRunRequest):
    """Run a backtest"""
    try:
        backtest_id = backtest_service.run_backtest(
            model_id=request.model_id, config=request.config, dataset=request.dataset
        )
        return {"status": "started", "backtest_id": backtest_id, "message": "Backtest submitted for execution"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{backtest_id}")
async def delete_backtest(backtest_id: str):
    """Delete a backtest result"""
    try:
        success = backtest_service.delete_backtest(backtest_id)
        if success:
            return {"status": "deleted", "backtest_id": backtest_id}
        else:
            raise HTTPException(status_code=404, detail=f"Backtest not found: {backtest_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
