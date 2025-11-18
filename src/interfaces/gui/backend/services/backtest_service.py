"""
Backtest Service

Business logic for backtest operations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.interfaces.gui.backend.api.models import BacktestResult


class BacktestService:
    """Service for backtest operations"""

    def __init__(self, artifacts_dir: Path = Path("artifacts")):
        self.artifacts_dir = artifacts_dir
        self.backtests_dir = artifacts_dir / "backtests"
        self.backtests_dir.mkdir(parents=True, exist_ok=True)

    def list_backtests(self, model_id: Optional[str] = None, limit: int = 100) -> List[BacktestResult]:
        """List backtest results"""
        results = []

        # Scan backtests directory
        for backtest_file in self.backtests_dir.glob("*_result.json"):
            try:
                with open(backtest_file, "r") as f:
                    data = json.load(f)

                # Filter by model_id
                if model_id and data.get("model_id") != model_id:
                    continue

                results.append(
                    BacktestResult(
                        id=backtest_file.stem.replace("_result", ""),
                        model_id=data.get("model_id", ""),
                        config=data.get("config", {}),
                        metrics=data.get("metrics", {}),
                        trades=data.get("trades", []),
                        equity_curve=data.get("equity_curve", []),
                        created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
                        duration_seconds=data.get("duration_seconds", 0.0),
                    )
                )
            except Exception as e:
                print(f"Error loading backtest {backtest_file}: {e}")
                continue

        # Sort by creation date (newest first)
        results.sort(key=lambda r: r.created_at, reverse=True)

        return results[:limit]

    def get_backtest(self, backtest_id: str) -> Optional[BacktestResult]:
        """Get backtest result by ID"""
        result_file = self.backtests_dir / f"{backtest_id}_result.json"
        if not result_file.exists():
            return None

        with open(result_file, "r") as f:
            data = json.load(f)

        return BacktestResult(
            id=backtest_id,
            model_id=data.get("model_id", ""),
            config=data.get("config", {}),
            metrics=data.get("metrics", {}),
            trades=data.get("trades", []),
            equity_curve=data.get("equity_curve", []),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            duration_seconds=data.get("duration_seconds", 0.0),
        )

    def run_backtest(self, model_id: str, config: Dict[str, Any], dataset: Optional[str] = None) -> str:
        """Run backtest (async operation, returns backtest_id)"""
        # Generate backtest ID
        backtest_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_id}"

        # Save initial result
        result_file = self.backtests_dir / f"{backtest_id}_result.json"
        initial_data = {
            "id": backtest_id,
            "model_id": model_id,
            "config": config,
            "status": "running",
            "created_at": datetime.now().isoformat(),
        }

        with open(result_file, "w") as f:
            json.dump(initial_data, f, indent=2)

        # TODO: Submit to task queue for async execution
        # For now, just return the ID

        return backtest_id

    def delete_backtest(self, backtest_id: str) -> bool:
        """Delete backtest result"""
        result_file = self.backtests_dir / f"{backtest_id}_result.json"
        if result_file.exists():
            result_file.unlink()
            return True
        return False
