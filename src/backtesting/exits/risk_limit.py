"""Risk Limit exit rule."""

import logging
from datetime import date, datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..position import Position
from .base import BaseExitRule

logger = logging.getLogger(__name__)


class RiskLimitExit(BaseExitRule):
    """Выход по риск-параметрам (дневной/портфельный лимит убытка)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация.

        Args:
            config: Конфигурация с параметрами:
                - risk_metric: "daily_loss", "portfolio_drawdown", "var", "exposure"
                - threshold: значение порога (проценты или абсолютные)
                - window: период расчета (для rolling метрик)
                - action: "close_all", "close_ticker", "reduce_position"
                - reset: условия сброса лимита
        """
        super().__init__(config)
        self.risk_metric: str = self.config.get("risk_metric", "daily_loss")
        self.threshold: float = self.config.get("threshold", 5.0)
        self.window: str = self.config.get("window", "day")
        self.action: str = self.config.get("action", "close_all")
        self.reset: str = self.config.get("reset", "new_day")

        # Состояние для отслеживания лимитов
        self._daily_pnl: Dict[str, float] = {}
        self._last_reset_date: Optional[pd.Timestamp] = None
        self._risk_limit_triggered: bool = False

    def should_exit(
        self, position: Position, bar: pd.Series, portfolio_state: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Проверить достижение риск-лимита.

        Args:
            position: Текущая позиция
            bar: Текущий бар
            portfolio_state: Состояние портфеля с информацией о текущем PnL

        Returns:
            (should_exit, exit_reason)
        """
        if not self.enabled:
            return False, None

        # Проверка reset условия
        self._check_reset(bar, portfolio_state)

        # Если лимит уже сработал и не сброшен
        if self._risk_limit_triggered:
            return True, f"risk_limit_{self.risk_metric}_triggered"

        # Вычисляем текущее значение риск-метрики
        risk_value = self._calculate_risk_metric(position, bar, portfolio_state)

        if risk_value is None:
            return False, None

        # Проверяем превышение порога
        if self._check_threshold_breach(risk_value):
            self._risk_limit_triggered = True
            reason = f"risk_limit_{self.risk_metric}_{self.threshold}"

            logger.warning(f"Risk limit breached: {self.risk_metric}={risk_value:.2f}, " f"threshold={self.threshold}")

            return True, reason

        return False, None

    def get_exit_price(self, position: Position, bar: pd.Series) -> Optional[float]:
        """Получить цену выхода.

        Args:
            position: Позиция
            bar: Текущий бар

        Returns:
            Цена выхода (рыночная - срочный выход)
        """
        return bar.get("close", bar.get("open"))

    def _calculate_risk_metric(
        self, position: Position, bar: pd.Series, portfolio_state: Dict[str, Any]
    ) -> Optional[float]:
        """Рассчитать значение риск-метрики.

        Args:
            position: Позиция
            bar: Текущий бар
            portfolio_state: Состояние портфеля

        Returns:
            Значение риск-метрики
        """
        if self.risk_metric == "daily_loss":
            # Дневной убыток
            daily_pnl = float(portfolio_state.get("daily_pnl", 0.0))
            initial_capital = float(portfolio_state.get("initial_capital", 100000.0))
            if initial_capital == 0:
                return 0.0
            return (daily_pnl / initial_capital) * 100  # В процентах

        if self.risk_metric == "portfolio_drawdown":
            # Просадка портфеля
            equity_history = portfolio_state.get("equity_history", [])
            if not equity_history:
                return 0.0

            try:
                peak = max(float(eq) for _, eq in equity_history)
            except (TypeError, ValueError):
                return 0.0

            current_equity = float(portfolio_state.get("current_equity", peak))

            if peak == 0:
                return 0.0

            drawdown = ((peak - current_equity) / peak) * 100
            return drawdown

        if self.risk_metric == "var":
            # Value at Risk
            returns = np.asarray(portfolio_state.get("returns", []), dtype=float)
            if returns.size < 30:
                return 0.0

            # VaR 95% = 1.65 * std
            var_95 = 1.65 * returns.std(ddof=0) * np.sqrt(252)
            return float(var_95 * 100)

        if self.risk_metric == "exposure":
            # Экспозиция по инструменту
            current_price = bar.get("close", bar.get("open"))
            if current_price is None:
                return 0.0

            try:
                current_price_value = float(current_price)
            except (TypeError, ValueError):
                return 0.0

            position_value = position.size * current_price_value
            equity = float(portfolio_state.get("current_equity", 100000.0))

            if equity == 0:
                return 0.0

            exposure = (position_value / equity) * 100
            return exposure

        return None

    def _check_threshold_breach(self, risk_value: float) -> bool:
        """Проверить превышение порога.

        Args:
            risk_value: Текущее значение риск-метрики

        Returns:
            True если порог превышен
        """
        if self.risk_metric == "daily_loss":
            # Для убытка: срабатывает если PnL < -threshold
            return risk_value < -self.threshold

        elif self.risk_metric == "portfolio_drawdown":
            # Для просадки: срабатывает если drawdown > threshold
            return risk_value > self.threshold

        elif self.risk_metric == "var":
            # Для VaR: срабатывает если VaR > threshold
            return risk_value > self.threshold

        elif self.risk_metric == "exposure":
            # Для экспозиции: срабатывает если exposure > threshold
            return risk_value > self.threshold

        return False

    def _check_reset(self, bar: pd.Series, portfolio_state: Dict[str, Any]) -> None:
        """Проверить условия сброса лимита.

        Args:
            bar: Текущий бар
            portfolio_state: Состояние портфеля
        """
        if self.reset == "new_day":
            # Сброс на новый торговый день
            current_timestamp = self._extract_timestamp(bar)
            if current_timestamp is None:
                return

            normalized = current_timestamp.normalize()

            if normalized != self._last_reset_date:
                self._risk_limit_triggered = False
                self._last_reset_date = normalized
                self._daily_pnl = {}

        elif self.reset == "manual":
            # Ручной сброс (через внешний вызов)
            pass

        elif self.reset == "never":
            # Никогда не сбрасывать
            pass

    def reset_limits(self) -> None:
        """Ручной сброс риск-лимитов."""
        self._risk_limit_triggered = False
        self._daily_pnl = {}

    @staticmethod
    def _extract_timestamp(bar: pd.Series) -> Optional[pd.Timestamp]:
        """Извлечь временную метку из бара."""
        raw_date = bar.get("date")
        if raw_date is not None:
            parsed = RiskLimitExit._to_timestamp(raw_date)
            if parsed is not None:
                return parsed

        raw_timestamp = bar.get("timestamp", bar.get("datetime"))
        return RiskLimitExit._to_timestamp(raw_timestamp)

    @staticmethod
    def _to_timestamp(value: Any) -> Optional[pd.Timestamp]:
        """Преобразовать значение к pd.Timestamp."""
        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            return value
        if isinstance(value, datetime):
            return pd.Timestamp(value)
        if isinstance(value, date):
            return pd.Timestamp(value)
        if isinstance(value, str):
            try:
                return pd.to_datetime(value)
            except ValueError:
                return None
        return None
