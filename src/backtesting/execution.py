"""Модель исполнения ордеров."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from .order import Execution, Order, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Конфигурация исполнения."""

    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    min_commission: float = 0.0  # Минимальная комиссия
    allow_partial_fills: bool = False  # Частичное исполнение
    market_impact: bool = False  # Учет влияния на рынок


class ExecutionModel:
    """Модель исполнения ордеров."""

    def __init__(self, config: Optional[ExecutionConfig] = None):
        """Инициализация.

        Args:
            config: Конфигурация исполнения
        """
        self.config = config or ExecutionConfig()

    def execute_order(self, order: Order, bar: Dict[str, float], execution_time: datetime) -> Optional[Execution]:
        """Исполнить ордер.

        Args:
            order: Ордер для исполнения
            bar: Текущий бар с ценами (open, high, low, close, volume)
            execution_time: Время исполнения

        Returns:
            Execution если ордер исполнен, None если не исполнен
        """
        # Определяем цену исполнения
        execution_price = self._get_execution_price(order, bar)
        if execution_price is None:
            logger.warning(f"Cannot execute order {order.ticker}: no valid price")
            order.status = OrderStatus.REJECTED
            return None

        # Применяем slippage
        slippage_amount = 0.0
        if self.config.slippage_rate > 0:
            slippage_amount = execution_price * self.config.slippage_rate
            if order.is_buy():
                execution_price += slippage_amount  # Покупаем дороже
            else:
                execution_price -= slippage_amount  # Продаем дешевле

        # Размер исполнения
        executed_size = order.size
        if self.config.allow_partial_fills:
            # Можем добавить логику частичного исполнения на основе объема
            executed_size = self._calculate_partial_fill(order, bar)

        # Рассчитываем комиссию
        commission = self._calculate_commission(execution_price, executed_size)

        # Создаем исполнение
        execution = Execution(
            order=order,
            executed_price=execution_price,
            executed_size=executed_size,
            executed_at=execution_time,
            commission=commission,
            slippage=slippage_amount * executed_size,
        )

        # Обновляем статус ордера
        if executed_size == order.size:
            order.status = OrderStatus.FILLED
        elif executed_size < order.size:
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.status = OrderStatus.REJECTED

        logger.debug(
            f"Executed order {order.ticker}: "
            f"price={execution_price:.4f}, size={executed_size:.2f}, "
            f"commission={commission:.2f}, slippage={execution.slippage:.2f}"
        )

        return execution

    def _get_execution_price(self, order: Order, bar: Dict[str, float]) -> Optional[float]:
        """Определить цену исполнения.

        Args:
            order: Ордер
            bar: Текущий бар

        Returns:
            Цена исполнения или None
        """
        if order.is_market():
            # Рыночный ордер исполняется по цене открытия следующего бара
            # Или можно использовать close текущего бара + spread
            return bar.get("close") or bar.get("open")

        elif order.is_limit() and order.price:
            # Лимитный ордер исполняется, если цена достигла лимита
            high = bar.get("high")
            low = bar.get("low")

            if high is None or low is None:
                return None

            if order.is_buy():
                # Buy limit: исполняется если цена упала до лимита или ниже
                if low <= order.price <= high:
                    return order.price
            else:
                # Sell limit: исполняется если цена поднялась до лимита или выше
                if low <= order.price <= high:
                    return order.price

        return None

    def _calculate_commission(self, price: float, size: float) -> float:
        """Рассчитать комиссию.

        Args:
            price: Цена исполнения
            size: Размер

        Returns:
            Комиссия
        """
        value = price * size
        commission = value * self.config.commission_rate
        return max(commission, self.config.min_commission)

    def _calculate_partial_fill(self, order: Order, bar: Dict[str, float]) -> float:
        """Рассчитать размер частичного исполнения.

        Args:
            order: Ордер
            bar: Текущий бар

        Returns:
            Размер исполнения
        """
        # Простая реализация: исполняем весь ордер
        # В более сложной версии можно учитывать объем в баре
        volume = bar.get("volume", 0)
        if volume == 0:
            return order.size

        # Например, можем исполнить максимум 10% объема бара
        max_fill = volume * 0.1
        return min(order.size, max_fill)

    def calculate_slippage(self, side: OrderSide, price: float, size: float) -> float:
        """Рассчитать проскальзывание.

        Args:
            side: Направление ордера
            price: Цена
            size: Размер

        Returns:
            Проскальзывание
        """
        base_slippage = price * self.config.slippage_rate

        # Если учитываем влияние на рынок
        if self.config.market_impact:
            # Простая модель: slippage растет с размером ордера
            impact_factor = 1.0 + (size / 10000.0)  # Упрощенная формула
            base_slippage *= impact_factor

        return base_slippage * size
