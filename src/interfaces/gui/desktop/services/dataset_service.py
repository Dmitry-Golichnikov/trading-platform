from typing import List
from uuid import UUID

import pandas as pd

from src.data.schemas import DatasetMetadata
from src.data.storage.catalog import DatasetCatalog
from src.data.storage.parquet_storage import ParquetStorage


class DatasetService:
    def __init__(self):
        self.catalog = DatasetCatalog()
        self.storage = ParquetStorage()

    def get_all_datasets(self) -> List[DatasetMetadata]:
        return self.catalog.search()

    def load_dataset_data(self, ticker: str, timeframe: str) -> pd.DataFrame:
        return self.storage.load_dataset(ticker, timeframe)

    def delete_dataset(self, dataset_id: UUID):
        # We need ticker/timeframe to delete from storage
        metadata = self.catalog.get_by_id(dataset_id)
        if metadata:
            self.storage.delete_dataset(metadata.ticker, metadata.timeframe)
            self.catalog.delete(dataset_id)
