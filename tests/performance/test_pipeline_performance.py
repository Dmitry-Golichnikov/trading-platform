"""Performance тесты для пайплайна данных."""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data.schemas import DatasetConfig
from src.data.storage.catalog import DatasetCatalog
from src.data.storage.parquet_storage import ParquetStorage
from src.data.storage.versioning import DataVersioning
from src.pipelines import DataPreparationPipeline

UTC = timezone.utc


class LargeMockLoader:
    """Генерирует большой объём минутных данных."""

    def load(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime,
        timeframe: str = "1m",
        **_: Any,
    ) -> pd.DataFrame:
        timestamps = pd.date_range(from_date, to_date, freq="1min", tz=UTC)
        if timestamps.empty:
            return pd.DataFrame()

        base = pd.Series(range(len(timestamps)), dtype=float)
        data = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": 100 + base,
                "high": 100.5 + base,
                "low": 99.5 + base,
                "close": 100.2 + base,
                "volume": (base * 5 + 100).astype(int),
            }
        )
        return data

    def get_available_tickers(self) -> list[str]:
        return ["PERF"]


@pytest.mark.performance
def test_pipeline_handles_large_dataset(tmp_path: Path) -> None:
    """Пайплайн должен обрабатывать крупные датасеты в разумное время."""
    loader = LargeMockLoader()
    artifacts_dir = tmp_path / "artifacts"
    storage = ParquetStorage(base_path=artifacts_dir)
    catalog = DatasetCatalog(db_path=artifacts_dir / "db" / "catalog.db")
    versioning = DataVersioning(versions_dir=artifacts_dir / "versions")

    # Около 90 дней минутных данных (~130k строк)
    config = DatasetConfig(
        ticker="PERF",
        timeframe="1m",
        from_date=date(2020, 1, 1),
        to_date=date(2020, 3, 31),
        source_type="api",
        api_token="TEST",
    )

    pipeline = DataPreparationPipeline(
        config=config,
        loader=loader,
        storage=storage,
        catalog=catalog,
        versioning=versioning,
        target_timeframes=("1m", "5m", "15m", "1h"),
        concurrency=4,
    )

    start = time.perf_counter()

    async def _run() -> None:
        await pipeline.run()

    asyncio.run(_run())
    elapsed = time.perf_counter() - start

    # Проверка на разумный предел (5 секунд для тестовых данных)
    assert elapsed < 5.0, f"Pipeline too slow: {elapsed:.2f}s"
