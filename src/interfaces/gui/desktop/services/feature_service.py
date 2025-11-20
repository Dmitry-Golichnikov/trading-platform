from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml

from src.features.generator import FeatureGenerator

logger = logging.getLogger(__name__)

INDICATOR_CATALOG: dict[str, list[str]] = {
    "Trend": ["SMA", "EMA", "WMA", "Ichimoku", "ADX", "Parabolic SAR", "Donchian Channels", "Keltner Channels"],
    "Momentum": [
        "RSI",
        "Stochastic Oscillator",
        "Stochastic RSI",
        "Williams %R",
        "MACD",
        "CCI",
        "TRIX",
        "Detrended Price Oscillator",
    ],
    "Volatility": ["ATR", "Bollinger Bands", "Keltner Channels", "Nadaraya-Watson Envelope"],
    "Volume": [
        "OBV",
        "VWAP",
        "Chaikin Money Flow",
        "Accumulation/Distribution",
        "Volume Profile",
        "Money Flow Index",
    ],
    "Price/Time": ["Heikin-Ashi", "Pivot Points", "Calendar", "Higher Timeframe"],
}


@dataclass(slots=True)
class FeatureGenerationResult:
    features: pd.DataFrame
    correlations: Optional[pd.DataFrame] = None


class FeatureService:
    """Работа с конфигурациями признаков и генерацией."""

    def __init__(self, *, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or Path("artifacts/features")

    def list_indicator_catalog(self) -> dict[str, list[str]]:
        return INDICATOR_CATALOG

    def validate_config(self, config: dict) -> None:
        if "features" not in config or not isinstance(config["features"], list):
            raise ValueError("Конфиг признаков должен содержать список features")

    def load_config(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def save_config(self, config: dict, path: Path) -> None:
        self.validate_config(config)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)

    def generate_features(
        self,
        data: pd.DataFrame,
        config: dict,
        *,
        dataset_id: Optional[str] = None,
        target: Optional[pd.Series] = None,
    ) -> FeatureGenerationResult:
        self.validate_config(config)
        generator = FeatureGenerator(config, cache_enabled=True, cache_dir=self.cache_dir)
        features = generator.generate(data, dataset_id=dataset_id, target=target, use_cache=True)
        corr = self._compute_correlations(features)
        return FeatureGenerationResult(features=features, correlations=corr)

    def _compute_correlations(self, features: pd.DataFrame) -> Optional[pd.DataFrame]:
        if features.empty:
            return None
        correlated = features.corr().round(3)
        return correlated

    def build_minimal_config(self, indicators: Iterable[str]) -> dict:
        feature_items = []
        for indicator in indicators:
            feature_items.append(
                {
                    "type": "indicator",
                    "name": indicator.lower().replace(" ", "_"),
                    "params": {"window": 14},
                    "columns": ["close"],
                    "prefix": indicator.lower(),
                }
            )

        return {
            "cache_enabled": True,
            "features": feature_items,
            "selection": {
                "enabled": False,
                "method": "variance_threshold",
                "top_k": 50,
                "params": {"threshold": 0.0},
            },
        }
