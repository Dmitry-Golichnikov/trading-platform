"""
Ресэмплинг OHLCV данных между таймфреймами.
"""

import logging
from typing import Sequence

import pandas as pd

from src.common.exceptions import PreprocessingError

logger = logging.getLogger(__name__)


class TimeframeResampler:
    """
    Ресэмплер для агрегации OHLCV данных.

    Поддерживаемые таймфреймы: 1m, 5m, 15m, 1h, 4h, 1d
    """

    # Маппинг таймфреймов на pandas freq strings
    TIMEFRAME_MAP = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
    }

    def resample(self, data: pd.DataFrame, from_tf: str, to_tf: str) -> pd.DataFrame:
        """
        Ресэмплировать данные из одного таймфрейма в другой.

        Args:
            data: DataFrame с OHLCV данными
            from_tf: Исходный таймфрейм ('1m', '5m', ...)
            to_tf: Целевой таймфрейм ('5m', '15m', ...)

        Returns:
            DataFrame с агрегированными данными

        Raises:
            PreprocessingError: При ошибке ресэмплинга
        """
        if from_tf not in self.TIMEFRAME_MAP:
            raise PreprocessingError(f"Unsupported timeframe: {from_tf}")
        if to_tf not in self.TIMEFRAME_MAP:
            raise PreprocessingError(f"Unsupported timeframe: {to_tf}")

        # Проверить что целевой таймфрейм больше исходного
        timeframes = list(self.TIMEFRAME_MAP.keys())
        if timeframes.index(to_tf) <= timeframes.index(from_tf):
            raise PreprocessingError(
                f"Cannot resample from {from_tf} to {to_tf}: "
                "target timeframe must be larger"
            )

        logger.info(f"Resampling from {from_tf} to {to_tf}")

        try:
            data = data.copy()

            # Убедиться что timestamp - это DatetimeIndex
            if "timestamp" in data.columns:
                data["timestamp"] = pd.to_datetime(data["timestamp"])
                data = data.set_index("timestamp")
            elif not isinstance(data.index, pd.DatetimeIndex):
                raise PreprocessingError(
                    "Data must have timestamp column or DatetimeIndex"
                )

            # Получить pandas freq string
            target_freq = self.TIMEFRAME_MAP[to_tf]

            # Сохранить ticker если есть
            ticker = data["ticker"].iloc[0] if "ticker" in data.columns else None

            # Ресэмплировать OHLCV данные
            resampled = data.resample(target_freq, label="left", closed="left").agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )

            # Удалить неполные бары (где нет данных)
            resampled = resampled.dropna()

            # Восстановить ticker
            if ticker:
                resampled["ticker"] = ticker

            # Сбросить индекс чтобы timestamp стал колонкой
            resampled = resampled.reset_index()

            logger.info(
                f"Resampled {len(data)} bars ({from_tf}) to "
                f"{len(resampled)} bars ({to_tf})"
            )

            return resampled

        except Exception as e:
            raise PreprocessingError(f"Resampling failed: {e}") from e

    def resample_multiple_timeframes(
        self,
        data: pd.DataFrame,
        source_tf: str = "1m",
        target_tfs: Sequence[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Ресэмплировать данные в несколько таймфреймов одновременно.

        Args:
            data: Исходные данные (обычно 1m)
            source_tf: Исходный таймфрейм
            target_tfs: Список целевых таймфреймов (по умолчанию все старшие)

        Returns:
            Словарь {таймфрейм: DataFrame}
        """
        if target_tfs is None:
            # Все таймфреймы старше source_tf
            all_tfs = list(self.TIMEFRAME_MAP.keys())
            source_idx = all_tfs.index(source_tf)
            start_index = source_idx + 1
            target_tfs = all_tfs[start_index:]

        results = {source_tf: data.copy()}

        for target_tf in target_tfs:
            try:
                resampled = self.resample(data, source_tf, target_tf)
                results[target_tf] = resampled
            except Exception as e:
                logger.error(f"Failed to resample to {target_tf}: {e}")

        return results
