"""Triple Barrier метод (Lopez de Prado)."""

import logging
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd

from src.labeling.base import BaseLabeler

logger = logging.getLogger(__name__)


class TripleBarrierLabeler(BaseLabeler):
    """
    Triple Barrier метод разметки.

    Основан на методологии из книги "Advances in Financial Machine Learning"
    (Marcos Lopez de Prado).

    Для каждой точки входа устанавливаются три барьера:
    - Верхний барьер (upper barrier) - take profit
    - Нижний барьер (lower barrier) - stop loss
    - Временной барьер (time barrier) - максимальное время удержания

    Метка определяется по тому, какой барьер пробит первым.
    """

    def __init__(
        self,
        upper_barrier: Union[float, Literal["atr"]] = 0.02,
        lower_barrier: Union[float, Literal["atr"]] = 0.02,
        time_barrier: int = 20,
        direction: Literal["long", "short", "long+short"] = "long+short",
        min_return: float = 0.0,
        atr_window: int = 20,
        atr_multiplier: float = 2.0,
        price_type: Literal["close", "typical"] = "close",
        include_commissions: bool = False,
        commission_pct: float = 0.001,
    ):
        """
        Инициализация Triple Barrier labeler.

        Args:
            upper_barrier: Верхний барьер (% или 'atr')
            lower_barrier: Нижний барьер (% или 'atr')
            time_barrier: Временной барьер (кол-во баров)
            direction: Направление торговли
            min_return: Минимальный return для учёта сигнала
            atr_window: Окно для ATR
            atr_multiplier: Множитель ATR для барьеров
            price_type: Тип цены для расчётов
            include_commissions: Учитывать комиссии
            commission_pct: Размер комиссии (%)
        """
        super().__init__(
            upper_barrier=upper_barrier,
            lower_barrier=lower_barrier,
            time_barrier=time_barrier,
            direction=direction,
            min_return=min_return,
            atr_window=atr_window,
            atr_multiplier=atr_multiplier,
            price_type=price_type,
            include_commissions=include_commissions,
            commission_pct=commission_pct,
        )

    def _validate_params(self) -> None:
        """Валидация параметров."""
        upper = self.params["upper_barrier"]
        lower = self.params["lower_barrier"]

        if upper != "atr" and (not isinstance(upper, (int, float)) or upper <= 0):
            raise ValueError("upper_barrier должен быть положительным числом или 'atr'")

        if lower != "atr" and (not isinstance(lower, (int, float)) or lower <= 0):
            raise ValueError("lower_barrier должен быть положительным числом или 'atr'")

        if self.params["time_barrier"] <= 0:
            raise ValueError("time_barrier должен быть положительным")

        if self.params["direction"] not in ["long", "short", "long+short"]:
            raise ValueError("direction должен быть 'long', 'short' или 'long+short'")

    def label(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Разметка данных Triple Barrier методом.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с добавленными колонками:
            - label: метка (-1, 0, 1)
            - barrier_hit: какой барьер пробит ('upper', 'lower', 'time')
            - holding_period: период удержания
            - realized_return: реализованный return
        """
        self.validate_data(data)

        result = data.copy()

        # Вычисление цены для расчётов
        if self.params["price_type"] == "close":
            price = result["close"]
        else:  # typical price
            price = (result["high"] + result["low"] + result["close"]) / 3

        # Вычисление барьеров
        upper_barriers, lower_barriers = self._compute_barriers(result)

        # Инициализация выходных колонок
        labels = np.zeros(len(result))
        barriers_hit = np.full(len(result), "", dtype=object)
        holding_periods = np.zeros(len(result))
        realized_returns = np.full(len(result), np.nan)

        # Векторизованный расчёт для каждой точки входа
        for i in range(len(result) - 1):
            entry_price = price.iloc[i]
            upper_barrier = upper_barriers.iloc[i]
            lower_barrier = lower_barriers.iloc[i]
            time_limit = min(i + self.params["time_barrier"], len(result) - 1)

            # Поиск пробития барьера
            label, barrier, holding, ret = self._find_barrier_hit(
                price=price,
                start_idx=i,
                end_idx=time_limit,
                entry_price=entry_price,
                upper_barrier=upper_barrier,
                lower_barrier=lower_barrier,
            )

            labels[i] = label
            barriers_hit[i] = barrier
            holding_periods[i] = holding
            realized_returns[i] = ret

        # Последний бар не размечаем (нет будущих данных)
        barriers_hit[-1] = "no_data"

        result["label"] = labels.astype(int)
        result["barrier_hit"] = barriers_hit
        result["holding_period"] = holding_periods.astype(int)
        result["realized_return"] = realized_returns

        # Фильтрация по min_return
        if self.params["min_return"] > 0:
            mask = abs(result["realized_return"]) < self.params["min_return"]
            result.loc[mask, "label"] = 0

        logger.info(
            f"Triple Barrier labeling завершена: "
            f"{sum(labels == 1)} long, "
            f"{sum(labels == -1)} short, "
            f"{sum(labels == 0)} neutral | "
            f"Upper: {sum(barriers_hit == 'upper')}, "
            f"Lower: {sum(barriers_hit == 'lower')}, "
            f"Time: {sum(barriers_hit == 'time')}"
        )

        return result

    def _compute_barriers(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Вычисление барьеров для каждого бара.

        Args:
            data: DataFrame с данными

        Returns:
            Кортеж (upper_barriers, lower_barriers)
        """
        upper = self.params["upper_barrier"]
        lower = self.params["lower_barrier"]

        if self.params["price_type"] == "close":
            price = data["close"]
        else:
            price = (data["high"] + data["low"] + data["close"]) / 3

        if upper == "atr":
            atr = self._compute_atr(data)
            upper_barriers = price + (atr * self.params["atr_multiplier"])
        else:
            upper_barriers = price * (1 + upper)

        if lower == "atr":
            atr = self._compute_atr(data)
            lower_barriers = price - (atr * self.params["atr_multiplier"])
        else:
            lower_barriers = price * (1 - lower)

        return upper_barriers, lower_barriers

    def _compute_atr(self, data: pd.DataFrame) -> pd.Series:
        """
        Вычисление Average True Range.

        Args:
            data: DataFrame с данными

        Returns:
            Series с ATR значениями
        """
        tr = pd.DataFrame(
            {
                "hl": data["high"] - data["low"],
                "hc": abs(data["high"] - data["close"].shift(1)),
                "lc": abs(data["low"] - data["close"].shift(1)),
            }
        ).max(axis=1)

        atr = tr.rolling(window=self.params["atr_window"], min_periods=1).mean()
        return atr

    def _find_barrier_hit(
        self,
        price: pd.Series,
        start_idx: int,
        end_idx: int,
        entry_price: float,
        upper_barrier: float,
        lower_barrier: float,
    ) -> Tuple[int, str, int, float]:
        """
        Поиск первого пробития барьера.

        Args:
            price: Series цен
            start_idx: Индекс входа
            end_idx: Индекс временного барьера
            entry_price: Цена входа
            upper_barrier: Уровень верхнего барьера
            lower_barrier: Уровень нижнего барьера

        Returns:
            Кортеж (label, barrier_type, holding_period, realized_return)
        """
        direction = self.params["direction"]
        commission = self.params["commission_pct"] if self.params["include_commissions"] else 0.0

        # Проверяем каждый бар в интервале
        for i in range(start_idx + 1, end_idx + 1):
            current_price = price.iloc[i]

            # Проверка пробития барьеров
            # При совпадении SL и TP на одном баре, приоритет у SL
            if current_price <= lower_barrier:
                # Нижний барьер пробит
                ret = (current_price - entry_price) / entry_price - 2 * commission
                holding = i - start_idx

                if direction == "long+short":
                    return -1, "lower", holding, ret
                elif direction == "short":
                    # Для short стратегии нижний барьер = profit
                    return 1, "lower", holding, ret
                else:  # long
                    return -1, "lower", holding, ret

            elif current_price >= upper_barrier:
                # Верхний барьер пробит
                ret = (current_price - entry_price) / entry_price - 2 * commission
                holding = i - start_idx

                if direction == "long+short":
                    return 1, "upper", holding, ret
                elif direction == "long":
                    return 1, "upper", holding, ret
                else:  # short
                    # Для short стратегии верхний барьер = loss
                    return -1, "upper", holding, ret

        # Временной барьер достигнут
        exit_price = price.iloc[end_idx]
        ret = (exit_price - entry_price) / entry_price - 2 * commission
        holding = end_idx - start_idx

        # Классификация по направлению движения
        if abs(ret) < self.params["min_return"]:
            return 0, "time", holding, ret

        if direction == "long+short":
            if ret > 0:
                return 1, "time", holding, ret
            else:
                return -1, "time", holding, ret
        elif direction == "long":
            return 1 if ret > 0 else 0, "time", holding, ret
        else:  # short
            return 1 if ret < 0 else 0, "time", holding, ret
