from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src.data.schemas import DatasetMetadata
from src.data.storage.catalog import DatasetCatalog
from src.data.storage.parquet_storage import ParquetStorage, SourceLiteral, TimeframeLiteral

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DatasetFilters:
    """Фильтры списка датасетов."""

    ticker: Optional[str] = None
    timeframe: Optional[str] = None
    source: Optional[str] = None


class DatasetService:
    """Высокоуровневые операции с датасетами для GUI."""

    def __init__(self, *, catalog: Optional[DatasetCatalog] = None, storage: Optional[ParquetStorage] = None) -> None:
        self.catalog = catalog or DatasetCatalog()
        self.storage = storage or ParquetStorage()

    def list_datasets(self, filters: Optional[DatasetFilters] = None) -> list[DatasetMetadata]:
        filters = filters or DatasetFilters()
        datasets = self.catalog.search(ticker=filters.ticker, timeframe=filters.timeframe, source=filters.source)
        if not datasets:
            datasets = self.storage.list_datasets(filters.ticker, filters.timeframe)
        return sorted(datasets, key=lambda meta: (meta.ticker, meta.timeframe))

    def load_preview(self, metadata: DatasetMetadata, *, limit: int = 10_000) -> pd.DataFrame:
        df = self.storage.load_dataset(metadata.ticker, metadata.timeframe)
        if limit > 0:
            df = df.tail(limit)
        return df

    def import_from_file(
        self,
        file_path: Path,
        *,
        ticker: str,
        timeframe: TimeframeLiteral,
        source: SourceLiteral = "manual",
    ) -> DatasetMetadata:
        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path, parse_dates=["timestamp"])
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_path.suffix}")

        metadata = self.storage.save_dataset(
            df,
            ticker=ticker,
            timeframe=timeframe,
            source=source,
        )
        self.catalog.add_dataset(metadata)
        return metadata

    def delete_dataset(self, metadata: DatasetMetadata) -> None:
        logger.info("Удаляем датасет %s/%s", metadata.ticker, metadata.timeframe)
        self.catalog.delete(metadata.dataset_id)
        self.storage.delete_dataset(metadata.ticker, metadata.timeframe)

    def get_summary(self, datasets: Iterable[DatasetMetadata]) -> dict[str, int]:
        dataset_list = list(datasets)
        total_bars = sum(item.total_bars for item in dataset_list)
        uniq_tickers = {item.ticker for item in dataset_list}
        uniq_timeframes = {item.timeframe for item in dataset_list}
        return {
            "datasets": len(dataset_list),
            "tickers": len(uniq_tickers),
            "timeframes": len(uniq_timeframes),
            "total_bars": total_bars,
        }
