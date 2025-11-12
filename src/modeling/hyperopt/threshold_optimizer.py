"""Threshold optimization based on expected PnL."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ThresholdResult:
    """Result of threshold optimization."""

    optimal_threshold: float
    expected_pnl: float
    n_trades: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]
    constraints_satisfied: bool
    threshold_curve: pd.DataFrame  # All thresholds and their metrics


class ThresholdOptimizer:
    """
    Optimize classification threshold based on expected PnL.

    Formula:
        E[PnL] = p_up * TP - (1 - p_up) * SL - commission

    Where:
        - p_up: probability of positive outcome (from model)
        - TP: take profit level
        - SL: stop loss level
        - commission: trading commission per trade
    """

    def __init__(
        self,
        tp: float,
        sl: float,
        commission: float = 0.001,
        constraints: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize threshold optimizer.

        Args:
            tp: Take profit level (e.g., 0.02 for 2%).
            sl: Stop loss level (e.g., 0.01 for 1%).
            commission: Trading commission per trade (e.g., 0.001 for 0.1%).
            constraints: Dictionary of constraints:
                - min_trades: Minimum number of trades
                - max_drawdown: Maximum allowed drawdown
                - min_sharpe: Minimum Sharpe ratio
                - min_win_rate: Minimum win rate
                - risk_penalty: Penalty factor for risk (default 0.0)
        """
        self.tp = tp
        self.sl = sl
        self.commission = commission
        self.constraints = constraints or {}

    def calculate_expected_pnl(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray,
        threshold: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate expected PnL for given threshold.

        Args:
            y_proba: Predicted probabilities.
            y_true: True labels.
            threshold: Classification threshold.

        Returns:
            Expected PnL and dict with additional metrics.
        """
        # Get predictions
        y_pred = (y_proba >= threshold).astype(int)

        # Filter trades above threshold
        trades = y_pred == 1
        n_trades = trades.sum()

        if n_trades == 0:
            return -np.inf, {"n_trades": 0}

        # Get outcomes for trades
        outcomes = y_true[trades]

        # Calculate win rate
        wins = outcomes == 1
        n_wins = wins.sum()
        win_rate = n_wins / n_trades if n_trades > 0 else 0

        # Calculate expected PnL per trade
        expected_pnl_per_trade = win_rate * self.tp - (1 - win_rate) * self.sl - self.commission

        # Total expected PnL
        total_expected_pnl = expected_pnl_per_trade * n_trades

        # Calculate additional metrics
        metrics = {
            "n_trades": int(n_trades),
            "win_rate": float(win_rate),
            "expected_pnl_per_trade": float(expected_pnl_per_trade),
            "total_expected_pnl": float(total_expected_pnl),
        }

        # Calculate P&L series for Sharpe and drawdown
        pnl_series = np.zeros(len(y_true))
        for i in range(len(y_true)):
            if trades[i]:
                if y_true[i] == 1:
                    pnl_series[i] = self.tp - self.commission
                else:
                    pnl_series[i] = -self.sl - self.commission

        cumulative_pnl = np.cumsum(pnl_series)

        # Calculate Sharpe ratio
        if n_trades > 1:
            trade_returns = pnl_series[trades]
            if trade_returns.std() > 0:
                sharpe_ratio = trade_returns.mean() / trade_returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        metrics["sharpe_ratio"] = float(sharpe_ratio)

        # Calculate maximum drawdown
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0
        metrics["max_drawdown"] = float(max_drawdown)

        # Calculate average profit and loss
        if n_wins > 0:
            avg_profit = self.tp - self.commission
        else:
            avg_profit = 0.0

        if n_trades > n_wins:
            avg_loss = self.sl + self.commission
        else:
            avg_loss = 0.0

        metrics["avg_profit"] = float(avg_profit)
        metrics["avg_loss"] = float(avg_loss)

        return total_expected_pnl, metrics

    def check_constraints(self, metrics: Dict[str, Any]) -> bool:
        """Check if metrics satisfy constraints."""
        # Minimum trades
        min_trades = self.constraints.get("min_trades", 0)
        if metrics["n_trades"] < min_trades:
            return False

        # Maximum drawdown
        max_dd = self.constraints.get("max_drawdown")
        if max_dd is not None and metrics["max_drawdown"] > max_dd:
            return False

        # Minimum Sharpe
        min_sharpe = self.constraints.get("min_sharpe")
        if min_sharpe is not None and metrics["sharpe_ratio"] < min_sharpe:
            return False

        # Minimum win rate
        min_win_rate = self.constraints.get("min_win_rate")
        if min_win_rate is not None and metrics["win_rate"] < min_win_rate:
            return False

        return True

    def optimize_threshold(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray,
        threshold_range: Optional[Tuple[float, float]] = None,
        n_thresholds: int = 100,
    ) -> ThresholdResult:
        """
        Find optimal threshold.

        Args:
            y_proba: Predicted probabilities.
            y_true: True labels.
            threshold_range: Range of thresholds to try (min, max).
            n_thresholds: Number of thresholds to evaluate.

        Returns:
            ThresholdResult with optimal threshold and metrics.
        """
        if threshold_range is None:
            # Use range based on probability distribution
            threshold_range = (
                max(0.0, y_proba.min()),
                min(1.0, y_proba.max()),
            )

        # Generate thresholds
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)

        # Evaluate each threshold
        results = []
        for threshold in thresholds:
            expected_pnl, metrics = self.calculate_expected_pnl(y_proba, y_true, threshold)

            # Check constraints
            constraints_satisfied = self.check_constraints(metrics)

            # Apply risk penalty if specified
            risk_penalty = self.constraints.get("risk_penalty", 0.0)
            if risk_penalty > 0:
                penalty = risk_penalty * metrics.get("max_drawdown", 0)
                expected_pnl -= penalty

            results.append(
                {
                    "threshold": float(threshold),
                    "expected_pnl": float(expected_pnl),
                    "n_trades": metrics["n_trades"],
                    "win_rate": metrics["win_rate"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                    "avg_profit": metrics["avg_profit"],
                    "avg_loss": metrics["avg_loss"],
                    "constraints_satisfied": constraints_satisfied,
                }
            )

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Filter by constraints
        valid_results = results_df[results_df["constraints_satisfied"]]

        if len(valid_results) == 0:
            print("Warning: No thresholds satisfy constraints. Using best unconstrained.")
            valid_results = results_df

        # Find optimal threshold
        best_idx = valid_results["expected_pnl"].idxmax()
        best_row = valid_results.loc[best_idx]

        result = ThresholdResult(
            optimal_threshold=float(best_row["threshold"]),
            expected_pnl=float(best_row["expected_pnl"]),
            n_trades=int(best_row["n_trades"]),
            win_rate=float(best_row["win_rate"]),
            avg_profit=float(best_row["avg_profit"]),
            avg_loss=float(best_row["avg_loss"]),
            sharpe_ratio=float(best_row["sharpe_ratio"]),
            max_drawdown=float(best_row["max_drawdown"]),
            constraints_satisfied=bool(best_row["constraints_satisfied"]),
            threshold_curve=results_df,
        )

        return result

    def plot_threshold_curve(
        self,
        result: ThresholdResult,
        save_path: Optional[str] = None,
    ):
        """
        Plot threshold curve.

        Args:
            result: ThresholdResult from optimization.
            save_path: Path to save plot (optional).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, skipping plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        df = result.threshold_curve

        # Expected PnL
        ax = axes[0, 0]
        ax.plot(df["threshold"], df["expected_pnl"], linewidth=2)
        ax.axvline(
            result.optimal_threshold,
            color="red",
            linestyle="--",
            label=f"Optimal: {result.optimal_threshold:.3f}",
        )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Expected PnL")
        ax.set_title("Expected PnL vs Threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Number of trades
        ax = axes[0, 1]
        ax.plot(df["threshold"], df["n_trades"], linewidth=2, color="green")
        ax.axvline(result.optimal_threshold, color="red", linestyle="--")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Number of Trades")
        ax.set_title("Number of Trades vs Threshold")
        ax.grid(True, alpha=0.3)

        # Win rate
        ax = axes[1, 0]
        ax.plot(df["threshold"], df["win_rate"], linewidth=2, color="orange")
        ax.axvline(result.optimal_threshold, color="red", linestyle="--")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Win Rate")
        ax.set_title("Win Rate vs Threshold")
        ax.grid(True, alpha=0.3)

        # Sharpe ratio
        ax = axes[1, 1]
        ax.plot(df["threshold"], df["sharpe_ratio"], linewidth=2, color="purple")
        ax.axvline(result.optimal_threshold, color="red", linestyle="--")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Sharpe Ratio vs Threshold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot to: {save_path}")

        plt.show()
