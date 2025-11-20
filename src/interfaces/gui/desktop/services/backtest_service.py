from typing import Any, Dict, List

import pandas as pd

from src.backtesting.engine import BacktestConfig, BacktestEngine, BacktestResult
from src.backtesting.strategy import SimpleMAStrategy


class BacktestService:
    def get_available_strategies(self) -> List[str]:
        return ["SimpleMA", "ModelBased"]

    def get_default_config(self) -> Dict[str, Any]:
        return {"initial_capital": 100000, "commission_rate": 0.001, "use_stop_loss": True, "stop_loss_value": 1.0}

    def run_backtest(
        self, strategy_name: str, data: pd.DataFrame, strategy_params: Dict[str, Any], backtest_config: Dict[str, Any]
    ) -> BacktestResult:

        # Configure Engine
        config = BacktestConfig(**backtest_config)
        engine = BacktestEngine(config)

        # Initialize Strategy
        if strategy_name == "SimpleMA":
            strategy = SimpleMAStrategy(strategy_params)
        elif strategy_name == "ModelBased":
            # Mock model for now as passing complex objects in GUI prototype is tricky
            # In real app, pass the loaded model
            strategy = SimpleMAStrategy(strategy_params)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Run
        result = engine.run(strategy, data, show_progress=False)
        return result
