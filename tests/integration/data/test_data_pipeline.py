"""Интеграционные тесты пайплайна подготовки данных."""

from __future__ import annotations

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


class MockLoader:
    """Простая реализация DataLoader для тестов."""

    def __init__(self, base_timeframe: str = "1m") -> None:
        self.base_timeframe = base_timeframe
        self.calls: list[tuple[str, datetime, datetime]] = []

    def load(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime,
        timeframe: str = "1m",
        **_: Any,
    ) -> pd.DataFrame:
        if timeframe != self.base_timeframe:
            raise ValueError(f"Unsupported timeframe request: {timeframe}")

        self.calls.append((ticker, from_date, to_date))

        # Создать минутные данные на заданный период
        # Добавляем 1 минуту, чтобы включить конечную дату
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
                "volume": (base * 10 + 100).astype(int),
            }
        )
        return data

    def get_available_tickers(self) -> list[str]:
        return ["TEST"]


@pytest.mark.asyncio
async def test_pipeline_full_cycle(tmp_path: Path) -> None:
    """Проверить полный цикл пайплайна для одного тикера."""
    loader = MockLoader()
    artifacts_dir = tmp_path / "artifacts"
    storage = ParquetStorage(base_path=artifacts_dir)
    catalog = DatasetCatalog(db_path=artifacts_dir / "db" / "catalog.db")
    versioning = DataVersioning(versions_dir=artifacts_dir / "versions")

    config = DatasetConfig(
        ticker="TEST",
        timeframe="1m",
        from_date=date(2020, 1, 1),
        to_date=date(2020, 1, 2),
        source_type="api",
        api_token="TEST",
    )

    pipeline = DataPreparationPipeline(
        config=config,
        loader=loader,
        storage=storage,
        catalog=catalog,
        versioning=versioning,
        target_timeframes=("1m", "5m", "15m"),
        concurrency=2,
    )

    results = await pipeline.run()

    assert results, "Pipeline must return results"
    assert results[0].ticker == "TEST"
    assert not results[0].missing_years

    # Проверить что данные сохранены во всех указанных таймфреймах
    metadata_1m = storage.get_metadata("TEST", "1m")
    assert metadata_1m.total_bars > 0

    data_5m = storage.load_dataset("TEST", "5m")
    assert not data_5m.empty

    # Каталог должен содержать записи
    datasets = catalog.search(ticker="TEST")
    assert datasets, "Datasets should be registered in catalog"

    # Версионирование должно сохранить хотя бы одну версию
    history = versioning.get_history("TEST", "1m")
    assert history, "Version history must not be empty"

    # Проверить что loader вызывался для каждого года
    years = {call[1].year for call in loader.calls}
    assert years == {2020}


@pytest.mark.asyncio
async def test_pipeline_updates_latest_year(tmp_path: Path) -> None:
    """Проверить что update_latest_year обновляет текущий год."""
    loader = MockLoader()
    artifacts_dir = tmp_path / "artifacts"
    storage = ParquetStorage(base_path=artifacts_dir)
    catalog = DatasetCatalog(db_path=artifacts_dir / "db" / "catalog.db")
    versioning = DataVersioning(versions_dir=artifacts_dir / "versions")

    current_year = datetime.now().year
    config = DatasetConfig(
        ticker="TEST",
        timeframe="1m",
        from_date=date(current_year, 1, 1),
        to_date=date(current_year, 1, 2),
        source_type="api",
        api_token="TEST",
        update_latest_year=True,
        backfill_missing=False,
    )

    pipeline = DataPreparationPipeline(
        config=config,
        loader=loader,
        storage=storage,
        catalog=catalog,
        versioning=versioning,
        concurrency=1,
    )

    await pipeline.run()
    first_call_count = len(loader.calls)
    assert first_call_count == 1

    # Повторный запуск должен снова обновить текущий год
    await pipeline.run()
    assert len(loader.calls) == 2
