"""Базовые классы стратегий."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .portfolio import Portfolio
from .position import PositionSide

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Базовая стратегия."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация.

        Args:
            config: Конфигурация стратегии
        """
        self.config = config or {}
        self.name = self.config.get("name", self.__class__.__name__)

    @abstractmethod
    def generate_signal(self, bar: pd.Series, portfolio: Portfolio) -> int:
        """Сгенерировать торговый сигнал.

        Args:
            bar: Текущий бар данных
            portfolio: Текущий портфель

        Returns:
            1 для покупки, -1 для продажи, 0 для ничего
        """
        pass

    @abstractmethod
    def size_position(self, signal: int, bar: pd.Series, portfolio: Portfolio) -> float:
        """Определить размер позиции.

        Args:
            signal: Торговый сигнал (1, -1, 0)
            bar: Текущий бар данных
            portfolio: Текущий портфель

        Returns:
            Размер позиции (количество акций/контрактов)
        """
        pass

    def calculate_stop_loss(self, side: PositionSide, entry_price: float, bar: pd.Series) -> Optional[float]:
        """Рассчитать уровень стоп-лосса.

        Args:
            side: Направление позиции
            entry_price: Цена входа
            bar: Текущий бар данных

        Returns:
            Уровень стоп-лосса или None
        """
        sl_type = self.config.get("stop_loss_type")
        if not sl_type:
            return None

        if sl_type == "percent":
            sl_percent = self.config.get("stop_loss_percent", 1.0)
            if side == PositionSide.LONG:
                return entry_price * (1 - sl_percent / 100)
            else:
                return entry_price * (1 + sl_percent / 100)

        elif sl_type == "atr":
            atr = bar.get("atr")
            if atr is None or np.isnan(atr):
                return None
            atr_multiplier = self.config.get("stop_loss_atr_multiplier", 2.0)
            if side == PositionSide.LONG:
                return entry_price - atr * atr_multiplier
            else:
                return entry_price + atr * atr_multiplier

        return None

    def calculate_take_profit(self, side: PositionSide, entry_price: float, bar: pd.Series) -> Optional[float]:
        """Рассчитать уровень тейк-профита.

        Args:
            side: Направление позиции
            entry_price: Цена входа
            bar: Текущий бар данных

        Returns:
            Уровень тейк-профита или None
        """
        tp_type = self.config.get("take_profit_type")
        if not tp_type:
            return None

        if tp_type == "percent":
            tp_percent = self.config.get("take_profit_percent", 2.0)
            if side == PositionSide.LONG:
                return entry_price * (1 + tp_percent / 100)
            else:
                return entry_price * (1 - tp_percent / 100)

        elif tp_type == "atr":
            atr = bar.get("atr")
            if atr is None or np.isnan(atr):
                return None
            atr_multiplier = self.config.get("take_profit_atr_multiplier", 3.0)
            if side == PositionSide.LONG:
                return entry_price + atr * atr_multiplier
            else:
                return entry_price - atr * atr_multiplier

        return None

    def on_bar(self, bar: pd.Series, portfolio: Portfolio) -> None:
        """Callback при получении нового бара.

        Args:
            bar: Новый бар данных
            portfolio: Текущий портфель
        """
        pass

    def on_trade(self, trade_info: Dict[str, Any]) -> None:
        """Callback при исполнении сделки.

        Args:
            trade_info: Информация о сделке
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """Получить параметры стратегии.

        Returns:
            Словарь параметров
        """
        return self.config.copy()


class ModelBasedStrategy(BaseStrategy):
    """Стратегия на основе модели."""

    def __init__(
        self,
        model: Any,
        config: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
    ):
        """Инициализация.

        Args:
            model: Обученная модель с методом predict или predict_proba
            config: Конфигурация стратегии
            threshold: Порог для генерации сигнала
        """
        super().__init__(config)
        self.model = model
        self.threshold = threshold
        self.feature_columns: List[str] = self.config.get("feature_columns", [])

    def generate_signal(self, bar: pd.Series, portfolio: Portfolio) -> int:
        """Сгенерировать торговый сигнал на основе модели.

        Args:
            bar: Текущий бар данных
            portfolio: Текущий портфель

        Returns:
            1 для покупки, -1 для продажи, 0 для ничего
        """
        try:
            # Подготовка признаков
            if self.feature_columns:
                feature_series = bar[self.feature_columns].astype(float)
            else:
                # Используем все числовые признаки, кроме таргета и временных меток
                exclude_cols = {"target", "timestamp", "datetime", "ticker"}
                numeric_values = []
                for col in bar.index:
                    if col in exclude_cols:
                        continue
                    value = bar[col]
                    try:
                        numeric_values.append(float(value))
                    except (TypeError, ValueError):
                        continue

                if not numeric_values:
                    return 0

                feature_series = pd.Series(numeric_values)

            features = np.asarray(feature_series, dtype=float).reshape(1, -1)

            # Проверка на NaN
            if np.any(np.isnan(features)):
                logger.warning("NaN values in features, returning neutral signal")
                return 0

            # Получение предсказания
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(features)[0]
                # Предполагаем бинарную классификацию
                if len(proba) == 2:
                    prob_up = proba[1]
                    if prob_up > self.threshold:
                        return 1
                    elif prob_up < (1 - self.threshold):
                        return -1
                # Мультиклассовая классификация
                elif len(proba) == 3:
                    # [down, neutral, up]
                    pred_class = np.argmax(proba)
                    if pred_class == 2:
                        return 1
                    elif pred_class == 0:
                        return -1
            else:
                prediction = self.model.predict(features)[0]
                if prediction > 0:
                    return 1
                elif prediction < 0:
                    return -1

            return 0

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return 0

    def size_position(self, signal: int, bar: pd.Series, portfolio: Portfolio) -> float:
        """Определить размер позиции.

        Args:
            signal: Торговый сигнал (1, -1, 0)
            bar: Текущий бар данных
            portfolio: Текущий портфель

        Returns:
            Размер позиции
        """
        if signal == 0:
            return 0.0

        # Получаем параметры размера позиции
        sizing_method = self.config.get("position_sizing", "fixed_amount")
        current_price_value = bar.get("close", bar.get("open"))
        if current_price_value is None:
            return 0.0
        try:
            current_price = float(current_price_value)
        except (TypeError, ValueError):
            return 0.0
        if current_price == 0 or np.isnan(current_price):
            return 0.0

        if sizing_method == "fixed_amount":
            # Фиксированная сумма в рублях/долларах
            amount = self.config.get("position_amount", 10000)
            size = amount / current_price

        elif sizing_method == "fixed_units":
            # Фиксированное количество акций
            size = float(self.config.get("position_units", 100))

        elif sizing_method == "percent_equity":
            # Процент от капитала
            current_prices = {bar.get("ticker", "UNKNOWN"): current_price}
            equity = portfolio.equity(current_prices)
            percent = self.config.get("position_percent", 10.0)
            amount = equity * (percent / 100)
            size = amount / current_price

        elif sizing_method == "risk_based":
            # На основе риска (Kelly criterion, fixed fractional)
            current_prices = {bar.get("ticker", "UNKNOWN"): current_price}
            equity = portfolio.equity(current_prices)
            risk_percent = self.config.get("risk_percent", 1.0)
            risk_amount = equity * (risk_percent / 100)

            # Рассчитываем риск на акцию (разница между входом и стоп-лоссом)
            side = PositionSide.LONG if signal > 0 else PositionSide.SHORT
            stop_loss = self.calculate_stop_loss(side, current_price, bar)

            if stop_loss is not None:
                risk_per_share = abs(current_price - stop_loss)
                if risk_per_share > 0:
                    size = risk_amount / risk_per_share
                else:
                    size = 0
            else:
                # Если нет стоп-лосса, используем фиксированный процент
                amount = equity * 0.1
                size = amount / current_price

        else:
            size = 100  # Default

        return max(0, int(size))  # Возвращаем целое число акций


class SimpleMAStrategy(BaseStrategy):
    """Простая стратегия на основе скользящих средних."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация.

        Args:
            config: Конфигурация с параметрами fast_period, slow_period
        """
        super().__init__(config)
        self.fast_period = int(self.config.get("fast_period", 10))
        self.slow_period = int(self.config.get("slow_period", 30))

    def generate_signal(self, bar: pd.Series, portfolio: Portfolio) -> int:
        """Сгенерировать сигнал на основе пересечения MA.

        Args:
            bar: Текущий бар данных
            portfolio: Текущий портфель

        Returns:
            1 для покупки, -1 для продажи, 0 для ничего
        """
        fast_ma = bar.get(f"sma_{self.fast_period}")
        slow_ma = bar.get(f"sma_{self.slow_period}")

        if fast_ma is None or slow_ma is None:
            return 0

        if np.isnan(fast_ma) or np.isnan(slow_ma):
            return 0

        # Простой crossover
        if fast_ma > slow_ma:
            return 1
        elif fast_ma < slow_ma:
            return -1

        return 0

    def size_position(self, signal: int, bar: pd.Series, portfolio: Portfolio) -> float:
        """Определить размер позиции.

        Args:
            signal: Торговый сигнал
            bar: Текущий бар данных
            portfolio: Текущий портфель

        Returns:
            Размер позиции
        """
        if signal == 0:
            return 0.0

        current_price_value = bar.get("close", bar.get("open"))
        if current_price_value is None:
            return 0.0
        try:
            current_price = float(current_price_value)
        except (TypeError, ValueError):
            return 0.0
        if current_price == 0 or np.isnan(current_price):
            return 0.0

        # Используем 10% капитала
        current_prices = {bar.get("ticker", "UNKNOWN"): current_price}
        equity = portfolio.equity(current_prices)
        amount = equity * 0.1
        size = amount / current_price

        return max(0, int(size))
