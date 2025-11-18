"""Извлечение календарных признаков."""

from typing import List, cast

import pandas as pd


class CalendarExtractor:
    """Извлечение календарных (временных) признаков."""

    def __init__(self, features: List[str]) -> None:
        self.features = features
        self._validate_features()

    def _validate_features(self) -> None:
        allowed = {
            "hour",
            "day_of_week",
            "month",
            "is_month_start",
            "is_month_end",
            "trading_day_of_month",
            "time_since_open",
            "time_to_close",
        }
        invalid = set(self.features) - allowed
        if invalid:
            raise ValueError(f"Неизвестные календарные признаки: {invalid}")

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Индекс должен быть DatetimeIndex")

        index = cast(pd.DatetimeIndex, data.index)
        features_df = pd.DataFrame(index=index)

        for feature_name in self.features:
            method = getattr(self, f"_extract_{feature_name}")
            features_df[feature_name] = method(index)

        return features_df

    def _extract_hour(self, index: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(index=index, data=index.hour, name="hour")

    def _extract_day_of_week(self, index: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(index=index, data=index.dayofweek, name="day_of_week")

    def _extract_month(self, index: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(index=index, data=index.month, name="month")

    def _extract_is_month_start(self, index: pd.DatetimeIndex) -> pd.Series:
        values = index.is_month_start.astype(int)
        return pd.Series(index=index, data=values, name="is_month_start")

    def _extract_is_month_end(self, index: pd.DatetimeIndex) -> pd.Series:
        values = index.is_month_end.astype(int)
        return pd.Series(index=index, data=values, name="is_month_end")

    def _extract_trading_day_of_month(self, index: pd.DatetimeIndex) -> pd.Series:
        series = index.to_series()
        counts = series.groupby([index.year, index.month]).cumcount() + 1
        return counts.rename("trading_day_of_month")

    def _extract_time_since_open(self, index: pd.DatetimeIndex, market_open: int = 10) -> pd.Series:
        hours = index.hour + index.minute / 60.0
        values = pd.Series(index=index, data=hours - market_open)
        return values.clip(lower=0).rename("time_since_open")

    def _extract_time_to_close(self, index: pd.DatetimeIndex, market_close: int = 18) -> pd.Series:
        hours = index.hour + index.minute / 60.0
        values = pd.Series(index=index, data=market_close - hours)
        return values.clip(lower=0).rename("time_to_close")
