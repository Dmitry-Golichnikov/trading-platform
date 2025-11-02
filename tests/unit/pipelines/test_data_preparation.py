"""Тесты для вспомогательных методов DataPreparationPipeline."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from src.data.schemas import DatasetConfig
from src.pipelines.data_preparation import DataPreparationPipeline


class StubLoader:
    def __init__(self, listing_year: int | None = None) -> None:
        self.listing_year = listing_year

    def get_listing_year(
        self, ticker: str
    ) -> int | None:  # pragma: no cover - simple stub
        return self.listing_year

    def get_available_tickers(self) -> list[str]:  # pragma: no cover - simple stub
        return ["TEST"]

    # DataLoader protocol requires load method but не используется в тестах
    def load(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime,
        timeframe: str = "1m",
        **kwargs: Any,
    ) -> pd.DataFrame:  # pragma: no cover - not used
        return pd.DataFrame()


def make_pipeline(loader: StubLoader) -> DataPreparationPipeline:
    cfg = DatasetConfig(
        ticker="TEST",
        timeframe="1m",
        from_date=date(2000, 1, 1),
        to_date=date(2005, 12, 31),
        source_type="api",
        api_token="TEST",
    )
    pipeline = DataPreparationPipeline(cfg, loader=loader)
    return pipeline


def test_expected_years_uses_listing_year() -> None:
    pipeline = make_pipeline(StubLoader(listing_year=2002))
    years = pipeline._determine_expected_years("TEST")
    assert years == {2002, 2003, 2004, 2005}


def test_expected_years_fallback_when_listing_unknown() -> None:
    pipeline = make_pipeline(StubLoader(listing_year=None))
    years = pipeline._determine_expected_years("TEST")
    assert years == {2000, 2001, 2002, 2003, 2004, 2005}
