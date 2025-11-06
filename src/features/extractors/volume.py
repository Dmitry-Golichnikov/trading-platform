"""Извлечение объёмных признаков."""

from typing import List

import numpy as np
import pandas as pd


class VolumeExtractor:
    """Извлечение объёмных признаков."""

    def __init__(self, features: List[str]):
        """
        Инициализация extractor'а.

        Args:
            features: Список признаков для извлечения
        """
        self.features = features
        self._validate_features()

    def _validate_features(self):
        """Валидация списка признаков."""
        allowed = {
            "volume_change",
            "volume_ma_ratio",
            "money_volume",
            "relative_volume",
            "volume_volatility",
        }
        invalid = set(self.features) - allowed
        if invalid:
            raise ValueError(f"Неизвестные объёмные признаки: {invalid}")

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечь объёмные признаки.

        Args:
            data: DataFrame с OHLC и volume данными

        Returns:
            DataFrame с признаками
        """
        features_df = pd.DataFrame(index=data.index)

        for feature_name in self.features:
            method = getattr(self, f"_extract_{feature_name}")
            features_df[feature_name] = method(data)

        return features_df

    def _extract_volume_change(self, data: pd.DataFrame) -> pd.Series:
        """
        Процентное изменение объёма (каузальное).

        Returns:
            Изменение объёма относительно предыдущей свечи
        """
        return data["volume"].pct_change()

    def _extract_volume_ma_ratio(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Отношение текущего объёма к его скользящей средней (каузальное).

        Args:
            window: Окно для вычисления MA

        Returns:
            volume / MA(volume)
        """
        volume_ma = data["volume"].rolling(window=window, min_periods=1).mean()
        return data["volume"] / volume_ma

    def _extract_money_volume(self, data: pd.DataFrame) -> pd.Series:
        """
        Денежный объём (каузальное).

        Returns:
            volume * типичная цена (high + low + close) / 3
        """
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        return data["volume"] * typical_price

    def _extract_relative_volume(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Относительный объём (каузальное).

        Args:
            window: Окно для вычисления средних

        Returns:
            (volume - mean) / std
        """
        rolling_mean = data["volume"].rolling(window=window, min_periods=1).mean()
        rolling_std = data["volume"].rolling(window=window, min_periods=1).std()

        # Избегаем деления на ноль
        rolling_std = rolling_std.replace(0, np.nan)

        return (data["volume"] - rolling_mean) / rolling_std

    def _extract_volume_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Волатильность объёма (каузальное).

        Args:
            window: Окно для вычисления std

        Returns:
            Стандартное отклонение объёма
        """
        return data["volume"].rolling(window=window, min_periods=1).std()
