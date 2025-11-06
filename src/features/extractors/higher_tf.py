"""Извлечение признаков из старших таймфреймов."""

from typing import List, Literal, Optional

import pandas as pd

from src.features.indicators.registry import IndicatorRegistry


class HigherTimeframeExtractor:
    """Извлечение признаков из старших таймфреймов с выравниванием."""

    def __init__(
        self,
        source_tf: str,
        indicators: List[str],
        alignment: Literal["forward_fill", "backward_fill", "interpolate"] = "forward_fill",
    ):
        """
        Инициализация extractor'а.

        Args:
            source_tf: Таймфрейм источника (например, '1h', '4h', '1d')
            indicators: Список названий индикаторов из старшего TF
            alignment: Метод выравнивания временных рядов
        """
        self.source_tf = source_tf
        self.indicators = indicators
        self.alignment = alignment
        self.registry = IndicatorRegistry()

    def extract(
        self,
        target_data: pd.DataFrame,
        source_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Извлечь признаки из старшего таймфрейма и выровнять с целевым.

        Args:
            target_data: DataFrame с данными целевого таймфрейма
            source_data: DataFrame с данными старшего таймфрейма (если None,
                        выполняется ресэмплинг target_data)

        Returns:
            DataFrame с признаками, выровненными по индексу target_data
        """
        try:
            target_index = pd.DatetimeIndex(target_data.index)
        except (TypeError, ValueError) as exc:
            raise ValueError("Индекс target_data должен быть DatetimeIndex") from exc

        # Если source_data не предоставлен, выполняем ресэмплинг
        if source_data is None:
            source_data = self._resample_data(target_data)

        features_df = pd.DataFrame(index=target_index)

        # Вычисляем индикаторы на старшем таймфрейме
        for indicator_name in self.indicators:
            indicator_values = self._compute_indicator(source_data, indicator_name)

            # Выравниваем с целевым таймфреймом
            aligned_values = self._align_with_target(indicator_values, target_index)

            # Добавляем префикс с таймфреймом
            feature_name = f"{indicator_name}_{self.source_tf}"
            features_df[feature_name] = aligned_values

        return features_df

    def _resample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ресэмплинг данных в старший таймфрейм.

        Args:
            data: DataFrame с исходными данными

        Returns:
            Ресэмплированный DataFrame
        """
        # Маппинг строковых обозначений в pandas offset aliases
        tf_mapping = {
            "1m": "1T",
            "5m": "5T",
            "15m": "15T",
            "30m": "30T",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
            "1w": "1W",
        }

        pandas_tf = tf_mapping.get(self.source_tf, self.source_tf)

        resampled = data.resample(pandas_tf).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

        # Удаляем строки с NaN (неполные свечи)
        resampled = resampled.dropna()

        return resampled

    def _compute_indicator(self, data: pd.DataFrame, indicator_name: str) -> pd.Series:
        """
        Вычислить индикатор на данных.

        Args:
            data: DataFrame с OHLC данными
            indicator_name: Название индикатора (может включать параметры)

        Returns:
            Series с значениями индикатора
        """
        # Парсим название индикатора и параметры
        # Формат: "SMA_20" или "RSI_14"
        parts = indicator_name.split("_")
        name = parts[0]

        # Пытаемся извлечь параметры из названия
        params = {}
        if len(parts) > 1:
            # Предполагаем, что последняя часть - это параметр window/period
            try:
                param_value = int(parts[-1])
                # Определяем название параметра в зависимости от индикатора
                if name in ["SMA", "EMA", "WMA"]:
                    params["window"] = param_value
                elif name in ["RSI", "ATR", "ADX"]:
                    params["period"] = param_value
            except ValueError:
                pass

        # Получаем класс индикатора из реестра
        indicator = self.registry.get(name, **params)
        result = indicator.calculate(data)

        # Индикатор может вернуть DataFrame с несколькими колонками
        # Берём первую колонку или ту, что совпадает с названием
        if isinstance(result, pd.DataFrame):
            if name in result.columns:
                return result[name]
            else:
                return result.iloc[:, 0]

        return result

    def _align_with_target(self, source_series: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
        """
        Выровнять данные старшего таймфрейма с целевым индексом.

        Args:
            source_series: Series с данными старшего таймфрейма
            target_index: Индекс целевого таймфрейма

        Returns:
            Series, выровненная по target_index
        """
        # Создаём Series с нужным индексом
        aligned = pd.Series(index=target_index, dtype=float)

        if self.alignment == "forward_fill":
            # Каждое значение из старшего TF распространяется вперёд
            # до следующего значения (каузальное)
            aligned = source_series.reindex(target_index, method="ffill")

        elif self.alignment == "backward_fill":
            # НЕ каузальное! Использовать только для анализа, не для обучения
            aligned = source_series.reindex(target_index, method="bfill")

        elif self.alignment == "interpolate":
            # Линейная интерполяция между значениями
            # НЕ каузальное! Использовать осторожно
            aligned = source_series.reindex(target_index).interpolate(method="linear")

        return aligned
