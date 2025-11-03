"""
Volume Profile - Профиль объёма.
"""

from typing import List

import numpy as np
import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("volume_profile")
class VolumeProfile(Indicator):
    """
    Volume Profile - Профиль объёма.

    Создаёт распределение объёма по ценовым уровням за период.

    Параметры:
        window (int): Период окна (по умолчанию 100)
        bins (int): Количество ценовых уровней (по умолчанию 20)

    Выходные колонки:
        VP_poc: Point of Control (цена с максимальным объёмом)
        VP_vah: Value Area High (верхняя граница области стоимости)
        VP_val: Value Area Low (нижняя граница области стоимости)

    Применение:
        - POC - уровень максимального объёма (сильная поддержка/сопротивление)
        - Value Area - область где торговалось 70% объёма
        - Уровни для входов и стоп-лоссов

    Example:
        >>> vp = VolumeProfile(window=100, bins=20)
        >>> result = vp.calculate(data)
    """

    def __init__(self, window: int = 100, bins: int = 20):
        """
        Инициализация Volume Profile.

        Args:
            window: Период окна
            bins: Количество ценовых уровней
        """
        super().__init__(window=window, bins=bins)

    def validate_params(self) -> None:
        """Валидация параметров."""
        window = self.params.get("window", 100)
        bins = self.params.get("bins", 20)

        if not isinstance(window, int) or window < 1:
            raise ValueError(
                f"window должен быть положительным целым числом, получено: {window}"
            )
        if not isinstance(bins, int) or bins < 2:
            raise ValueError(f"bins должен быть >= 2, получено: {bins}")

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return ["high", "low", "close", "volume"]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        return self.params.get("window", 100)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать Volume Profile.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонками VP_poc, VP_vah, VP_val
        """
        self._validate_data(data)

        window = self.params["window"]
        bins = self.params["bins"]

        n = len(data)
        poc = np.full(n, np.nan)
        vah = np.full(n, np.nan)
        val = np.full(n, np.nan)

        # Рассчитываем для каждого бара (rolling window)
        for i in range(window - 1, n):
            # Берём данные за окно
            window_data = data.iloc[i - window + 1 : i + 1]

            # Диапазон цен
            price_min = window_data["low"].min()
            price_max = window_data["high"].max()

            # Создаём ценовые уровни
            price_levels = np.linspace(price_min, price_max, bins + 1)

            # Распределяем объём по уровням
            volume_per_level = np.zeros(bins)

            for _, row in window_data.iterrows():
                # Для каждого бара распределяем объём пропорционально
                # по ценовым уровням между low и high
                bar_low = row["low"]
                bar_high = row["high"]
                bar_volume = row["volume"]

                # Находим индексы уровней в диапазоне [low, high]
                for level_idx in range(bins):
                    level_low = price_levels[level_idx]
                    level_high = price_levels[level_idx + 1]

                    # Проверяем пересечение с баром
                    if level_high >= bar_low and level_low <= bar_high:
                        # Пропорция пересечения
                        overlap_low = max(level_low, bar_low)
                        overlap_high = min(level_high, bar_high)
                        overlap = overlap_high - overlap_low

                        if bar_high > bar_low:
                            proportion = overlap / (bar_high - bar_low)
                        else:
                            proportion = 1.0 / bins

                        volume_per_level[level_idx] += bar_volume * proportion

            # Point of Control (уровень с максимальным объёмом)
            poc_idx = np.argmax(volume_per_level)
            poc[i] = (price_levels[poc_idx] + price_levels[poc_idx + 1]) / 2

            # Value Area (70% объёма)
            total_volume = volume_per_level.sum()
            target_volume = total_volume * 0.70

            # Начинаем с POC и расширяем в обе стороны
            value_area_indices = {poc_idx}
            current_volume = volume_per_level[poc_idx]

            while current_volume < target_volume:
                # Находим соседние уровни
                candidates = []
                for idx in value_area_indices.copy():
                    if idx > 0 and (idx - 1) not in value_area_indices:
                        candidates.append((idx - 1, volume_per_level[idx - 1]))
                    if idx < bins - 1 and (idx + 1) not in value_area_indices:
                        candidates.append((idx + 1, volume_per_level[idx + 1]))

                if not candidates:
                    break

                # Добавляем уровень с максимальным объёмом
                best_idx = max(candidates, key=lambda x: x[1])[0]
                value_area_indices.add(best_idx)
                current_volume += volume_per_level[best_idx]

            # Value Area High и Low
            va_indices = sorted(value_area_indices)
            if va_indices:
                val[i] = price_levels[va_indices[0]]
                vah[i] = price_levels[va_indices[-1] + 1]

        result = pd.DataFrame(
            {"VP_poc": poc, "VP_vah": vah, "VP_val": val}, index=data.index
        )

        return result
