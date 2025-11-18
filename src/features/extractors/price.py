"""Извлечение ценовых признаков."""

from typing import List

import numpy as np
import pandas as pd


class PriceExtractor:
    """Извлечение ценовых признаков из OHLC данных."""

    def __init__(self, features: List[str]) -> None:
        self.features = features
        self._validate_features()

    def _validate_features(self) -> None:
        allowed = {
            "returns",
            "log_returns",
            "high_low_ratio",
            "close_open_ratio",
            "body_size",
            "upper_wick",
            "lower_wick",
            "price_position",
        }
        invalid = set(self.features) - allowed
        if invalid:
            raise ValueError(f"Неизвестные ценовые признаки: {invalid}")

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        features_df = pd.DataFrame(index=data.index)
        for feature_name in self.features:
            method = getattr(self, f"_extract_{feature_name}")
            features_df[feature_name] = method(data)
        return features_df

    def _extract_returns(self, data: pd.DataFrame) -> pd.Series:
        return data["close"].pct_change()

    def _extract_log_returns(self, data: pd.DataFrame) -> pd.Series:
        ratio = data["close"].div(data["close"].shift(1))
        return ratio.transform(np.log)

    def _extract_high_low_ratio(self, data: pd.DataFrame) -> pd.Series:
        return data["high"].div(data["low"])

    def _extract_close_open_ratio(self, data: pd.DataFrame) -> pd.Series:
        return data["close"].div(data["open"])

    def _extract_body_size(self, data: pd.DataFrame) -> pd.Series:
        body = (data["close"] - data["open"]).abs()
        full_range = data["high"] - data["low"]
        return body.div(full_range.replace(0, np.nan))

    def _extract_upper_wick(self, data: pd.DataFrame) -> pd.Series:
        upper_wick = data["high"] - data[["open", "close"]].max(axis=1)
        full_range = data["high"] - data["low"]
        return upper_wick.div(full_range.replace(0, np.nan))

    def _extract_lower_wick(self, data: pd.DataFrame) -> pd.Series:
        lower_wick = data[["open", "close"]].min(axis=1) - data["low"]
        full_range = data["high"] - data["low"]
        return lower_wick.div(full_range.replace(0, np.nan))

    def _extract_price_position(self, data: pd.DataFrame) -> pd.Series:
        position = (data["close"] - data["low"]).div(data["high"] - data["low"])
        return position.replace([np.inf, -np.inf], np.nan)
