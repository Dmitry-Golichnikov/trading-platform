"""Базовый класс для exit rules."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd

from ..position import Position


class BaseExitRule(ABC):
    """Базовый класс для правил выхода."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация.

        Args:
            config: Конфигурация правила
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)

    @abstractmethod
    def should_exit(
        self, position: Position, bar: pd.Series, portfolio_state: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Проверить, нужно ли выходить из позиции.

        Args:
            position: Текущая позиция
            bar: Текущий бар данных
            portfolio_state: Состояние портфеля

        Returns:
            (should_exit, exit_reason) - True если нужно выходить и причина
        """
        pass

    @abstractmethod
    def get_exit_price(self, position: Position, bar: pd.Series) -> Optional[float]:
        """Получить цену выхода.

        Args:
            position: Текущая позиция
            bar: Текущий бар данных

        Returns:
            Цена выхода или None для рыночной цены
        """
        pass

    def update(self, position: Position, bar: pd.Series, portfolio_state: Dict[str, Any]) -> None:
        """Обновить состояние правила (например, для trailing stop).

        Args:
            position: Текущая позиция
            bar: Текущий бар данных
            portfolio_state: Состояние портфеля
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Получить конфигурацию правила.

        Returns:
            Конфигурация
        """
        return self.config.copy()
