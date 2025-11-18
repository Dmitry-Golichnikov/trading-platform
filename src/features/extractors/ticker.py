"""Извлечение тикер-специфичных признаков."""

from typing import Literal, Optional

import pandas as pd


class TickerExtractor:
    """Извлечение тикер-специфичных признаков."""

    def __init__(
        self,
        encoding: Literal["label", "onehot", "target"] = "label",
        sector: bool = False,
        industry: bool = False,
    ):
        """
        Инициализация extractor'а.

        Args:
            encoding: Тип кодирования тикера
            sector: Добавить признак сектора
            industry: Добавить признак индустрии
        """
        self.encoding = encoding
        self.sector = sector
        self.industry = industry

    def extract(
        self,
        data: pd.DataFrame,
        ticker_column: str = "ticker",
        sector_mapping: Optional[dict] = None,
        industry_mapping: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Извлечь тикер-специфичные признаки.

        Args:
            data: DataFrame с колонкой ticker
            ticker_column: Название колонки с тикером
            sector_mapping: Маппинг тикер -> сектор
            industry_mapping: Маппинг тикер -> индустрия

        Returns:
            DataFrame с признаками
        """
        if ticker_column not in data.columns:
            raise ValueError(f"Колонка {ticker_column} не найдена в данных")

        features_df = pd.DataFrame(index=data.index)

        # Кодирование тикера
        if self.encoding == "label":
            features_df["ticker_encoded"] = pd.factorize(data[ticker_column])[0]
        elif self.encoding == "onehot":
            onehot = pd.get_dummies(data[ticker_column], prefix="ticker")
            features_df = pd.concat([features_df, onehot], axis=1)
        elif self.encoding == "target":
            # Target encoding требует таргет - пока оставляем как label
            features_df["ticker_encoded"] = pd.factorize(data[ticker_column])[0]

        # Добавление сектора
        if self.sector and sector_mapping:
            features_df["sector"] = data[ticker_column].map(sector_mapping)
            features_df["sector_encoded"] = pd.factorize(features_df["sector"])[0]

        # Добавление индустрии
        if self.industry and industry_mapping:
            features_df["industry"] = data[ticker_column].map(industry_mapping)
            features_df["industry_encoded"] = pd.factorize(features_df["industry"])[0]

        return features_df
