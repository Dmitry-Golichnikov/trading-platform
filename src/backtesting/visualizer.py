"""Визуализация результатов бэктестинга."""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .position import Position

logger = logging.getLogger(__name__)

# Опциональный импорт seaborn
try:
    import seaborn as sns

    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logger.warning("seaborn not available, using basic matplotlib styling")

plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10


class BacktestVisualizer:
    """Визуализация результатов бэктеста."""

    def __init__(
        self,
        equity_curve: pd.DataFrame,
        trades: List[Position],
        portfolio_metrics: dict,
    ):
        """Инициализация.

        Args:
            equity_curve: DataFrame с историей капитала
            trades: Список закрытых сделок
            portfolio_metrics: Метрики портфеля
        """
        self.equity_curve = equity_curve
        self.trades = trades
        self.metrics = portfolio_metrics

    def plot_equity_curve(self, save_path: Optional[Path] = None, show: bool = True) -> None:
        """Построить equity curve.

        Args:
            save_path: Путь для сохранения графика
            show: Показать график
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Equity curve
        ax1.plot(
            self.equity_curve["timestamp"],
            self.equity_curve["equity"],
            label="Equity",
            linewidth=2,
        )
        ax1.axhline(
            y=self.equity_curve["equity"].iloc[0],
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Initial Capital",
        )
        ax1.set_ylabel("Equity", fontsize=12)
        ax1.set_title("Equity Curve", fontsize=14, fontweight="bold")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Returns
        if "returns" in self.equity_curve.columns:
            returns = self.equity_curve["returns"].dropna()
            ax2.bar(
                self.equity_curve["timestamp"][1:],
                returns * 100,
                label="Returns",
                alpha=0.6,
                color=np.where(returns > 0, "green", "red"),
            )
            ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax2.set_ylabel("Returns (%)", fontsize=12)
            ax2.set_xlabel("Time", fontsize=12)
            ax2.set_title("Period Returns", fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Equity curve saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_drawdown(self, save_path: Optional[Path] = None, show: bool = True) -> None:
        """Построить график просадок.

        Args:
            save_path: Путь для сохранения
            show: Показать график
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Вычисляем просадку
        equity = self.equity_curve["equity"]
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100

        # График просадки
        ax.fill_between(
            self.equity_curve["timestamp"],
            drawdown,
            0,
            alpha=0.3,
            color="red",
            label="Drawdown",
        )
        ax.plot(
            self.equity_curve["timestamp"],
            drawdown,
            color="red",
            linewidth=1.5,
        )

        # Максимальная просадка
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        if max_dd_idx in self.equity_curve.index:
            max_dd_time = self.equity_curve.loc[max_dd_idx, "timestamp"]
            ax.scatter(
                [max_dd_time],
                [max_dd],
                color="darkred",
                s=100,
                zorder=5,
                label=f"Max DD: {max_dd:.2f}%",
            )

        ax.set_ylabel("Drawdown (%)", fontsize=12)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_title("Drawdown Over Time", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Drawdown chart saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_trades(self, save_path: Optional[Path] = None, show: bool = True) -> None:
        """Построить график сделок.

        Args:
            save_path: Путь для сохранения
            show: Показать график
        """
        if len(self.trades) == 0:
            logger.warning("No trades to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. PnL по сделкам
        ax1 = axes[0, 0]
        pnls = [t.realized_pnl for t in self.trades]
        colors = ["green" if p > 0 else "red" for p in pnls]
        ax1.bar(range(len(pnls)), pnls, color=colors, alpha=0.6)
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax1.set_xlabel("Trade #", fontsize=11)
        ax1.set_ylabel("PnL", fontsize=11)
        ax1.set_title("PnL per Trade", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # 2. Кумулятивный PnL
        ax2 = axes[0, 1]
        cumulative_pnl = np.cumsum(pnls)
        ax2.plot(cumulative_pnl, linewidth=2, color="blue")
        ax2.fill_between(
            range(len(cumulative_pnl)),
            cumulative_pnl,
            0,
            alpha=0.3,
            color="blue",
        )
        ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax2.set_xlabel("Trade #", fontsize=11)
        ax2.set_ylabel("Cumulative PnL", fontsize=11)
        ax2.set_title("Cumulative PnL", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # 3. Распределение PnL
        ax3 = axes[1, 0]
        ax3.hist(pnls, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
        ax3.axvline(x=0, color="red", linestyle="--", linewidth=2)
        ax3.axvline(
            x=np.mean(pnls),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(pnls):.2f}",
        )
        ax3.set_xlabel("PnL", fontsize=11)
        ax3.set_ylabel("Frequency", fontsize=11)
        ax3.set_title("PnL Distribution", fontsize=12, fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Holding periods
        ax4 = axes[1, 1]
        holding_periods = [t.holding_period for t in self.trades if t.holding_period is not None]
        if holding_periods:
            ax4.hist(
                holding_periods,
                bins=30,
                alpha=0.7,
                color="orange",
                edgecolor="black",
            )
            ax4.axvline(
                x=np.mean(holding_periods),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {np.mean(holding_periods):.1f}h",
            )
            ax4.set_xlabel("Holding Period (hours)", fontsize=11)
            ax4.set_ylabel("Frequency", fontsize=11)
            ax4.set_title("Holding Period Distribution", fontsize=12, fontweight="bold")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Trades chart saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_returns_distribution(self, save_path: Optional[Path] = None, show: bool = True) -> None:
        """Построить распределение returns.

        Args:
            save_path: Путь для сохранения
            show: Показать график
        """
        if "returns" not in self.equity_curve.columns:
            logger.warning("No returns data available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        returns = self.equity_curve["returns"].dropna() * 100

        # Гистограмма
        ax1.hist(returns, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        ax1.axvline(x=0, color="red", linestyle="--", linewidth=2)
        ax1.axvline(
            x=returns.mean(),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {returns.mean():.3f}%",
        )
        ax1.set_xlabel("Returns (%)", fontsize=11)
        ax1.set_ylabel("Frequency", fontsize=11)
        ax1.set_title("Returns Distribution", fontsize=12, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats

        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Returns distribution saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_summary(self, save_path: Optional[Path] = None, show: bool = True) -> None:
        """Построить сводный график со всеми метриками.

        Args:
            save_path: Путь для сохранения
            show: Показать график
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(
            self.equity_curve["timestamp"],
            self.equity_curve["equity"],
            linewidth=2,
            color="blue",
        )
        ax1.set_title("Equity Curve", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Equity", fontsize=11)
        ax1.grid(True, alpha=0.3)

        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        equity = self.equity_curve["equity"]
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        ax2.fill_between(
            self.equity_curve["timestamp"],
            drawdown,
            0,
            alpha=0.3,
            color="red",
        )
        ax2.plot(self.equity_curve["timestamp"], drawdown, color="red", linewidth=1.5)
        ax2.set_title("Drawdown", fontsize=13, fontweight="bold")
        ax2.set_ylabel("DD (%)", fontsize=11)
        ax2.grid(True, alpha=0.3)

        # 3. PnL per trade
        if len(self.trades) > 0:
            ax3 = fig.add_subplot(gs[2, 0])
            pnls = [t.realized_pnl for t in self.trades]
            colors = ["green" if p > 0 else "red" for p in pnls]
            ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.6)
            ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax3.set_title("PnL per Trade", fontsize=12, fontweight="bold")
            ax3.set_xlabel("Trade #", fontsize=10)
            ax3.set_ylabel("PnL", fontsize=10)
            ax3.grid(True, alpha=0.3)

            # 4. Returns distribution
            ax4 = fig.add_subplot(gs[2, 1])
            if "returns" in self.equity_curve.columns:
                returns = self.equity_curve["returns"].dropna() * 100
                ax4.hist(returns, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
                ax4.axvline(x=0, color="red", linestyle="--", linewidth=1)
                ax4.set_title("Returns Distribution", fontsize=12, fontweight="bold")
                ax4.set_xlabel("Returns (%)", fontsize=10)
                ax4.grid(True, alpha=0.3)

        # 5. Metrics table
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis("off")

        metrics_text = [
            f"Total Return: {self.metrics.get('total_return', 0):.2f}%",
            f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}",
            f"Max DD: {self.metrics.get('max_drawdown', 0):.2f}%",
            f"Win Rate: {self.metrics.get('win_rate', 0):.2f}%",
            f"Profit Factor: {self.metrics.get('profit_factor', 0):.2f}",
            f"Total Trades: {self.metrics.get('total_trades', 0)}",
        ]

        y_pos = 0.9
        for text in metrics_text:
            ax5.text(
                0.1,
                y_pos,
                text,
                fontsize=11,
                verticalalignment="top",
                fontfamily="monospace",
            )
            y_pos -= 0.15

        ax5.set_title("Key Metrics", fontsize=12, fontweight="bold")

        plt.suptitle("Backtest Summary", fontsize=16, fontweight="bold", y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Summary chart saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def create_all_plots(self, output_dir: Path, show: bool = False) -> None:
        """Создать все графики и сохранить.

        Args:
            output_dir: Директория для сохранения
            show: Показывать графики
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating all plots in {output_dir}")

        self.plot_equity_curve(output_dir / "equity_curve.png", show=show)
        self.plot_drawdown(output_dir / "drawdown.png", show=show)
        self.plot_trades(output_dir / "trades.png", show=show)
        self.plot_returns_distribution(output_dir / "returns_distribution.png", show=show)
        self.plot_summary(output_dir / "summary.png", show=show)

        logger.info("All plots created successfully")
