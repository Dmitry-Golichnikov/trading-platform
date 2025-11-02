"""
Хранилище данных в формате Parquet с поддержкой артефактов.
"""

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from src.common.exceptions import StorageError
from src.data.schemas import DatasetMetadata

logger = logging.getLogger(__name__)

TimeframeLiteral = Literal["1m", "5m", "15m", "1h", "4h", "1d"]
SourceLiteral = Literal["local", "tinkoff", "manual"]


class ParquetStorage:
    """
    Хранилище для работы с датасетами в формате Parquet.

    Структура хранения:
    artifacts/
    ├── raw/
    │   ├── downloads/{ticker}/{year}.zip    # Скачанные архивы
    │   └── extracted/{ticker}/{year}/       # Распакованные CSV
    └── data/
        └── {ticker}/
            └── {timeframe}/
                ├── {year}.parquet           # Данные по годам
                └── metadata.json             # Метаданные датасета
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Инициализировать хранилище.

        Args:
            base_path: Базовый путь к artifacts/ (по умолчанию ./artifacts)
        """
        self.base_path = Path(base_path) if base_path else Path("artifacts")

        # Создать структуру директорий
        self.data_dir = self.base_path / "data"
        self.downloads_dir = self.base_path / "raw" / "downloads"
        self.extracted_dir = self.base_path / "raw" / "extracted"

        for dir_path in [self.data_dir, self.downloads_dir, self.extracted_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"ParquetStorage initialized: base_path={self.base_path}")

    def save_dataset(
        self,
        data: pd.DataFrame,
        ticker: str,
        timeframe: TimeframeLiteral,
        source: SourceLiteral = "local",
        metadata: Optional[DatasetMetadata] = None,
    ) -> DatasetMetadata:
        """
        Сохранить датасет в хранилище.

        Данные автоматически разбиваются по годам для эффективного хранения.

        Args:
            data: DataFrame с OHLCV данными
            ticker: Тикер инструмента
            timeframe: Временной интервал
            source: Источник данных
            metadata: Метаданные (если None, будут созданы автоматически)

        Returns:
            Метаданные сохранённого датасета

        Raises:
            StorageError: При ошибке сохранения
        """
        try:
            if data.empty:
                raise StorageError("Cannot save empty dataset")

            # Убедиться что timestamp есть
            if "timestamp" not in data.columns:
                raise StorageError("DataFrame must have 'timestamp' column")

            # Создать директорию для тикера/таймфрейма
            dataset_dir = self.data_dir / ticker / timeframe
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Разбить по годам и сохранить
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data["year"] = data["timestamp"].dt.year

            for year, year_data in data.groupby("year"):
                year_file = dataset_dir / f"{year}.parquet"
                year_data_clean = year_data.drop(columns=["year"])
                year_data_clean.to_parquet(year_file, compression="snappy", index=False)
                logger.debug(f"Saved {len(year_data_clean)} bars to {year_file}")

            # Создать или обновить метаданные
            if metadata is None:
                metadata = self._create_metadata(
                    data=data, ticker=ticker, timeframe=timeframe, source=source
                )

            # Сохранить метаданные
            metadata_file = dataset_dir / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata.model_dump(mode="json"), f, indent=2, default=str)

            logger.info(
                f"Saved dataset: {ticker}/{timeframe}, "
                f"{len(data)} bars, {len(data['year'].unique())} years"
            )

            return metadata

        except Exception as e:
            raise StorageError(f"Failed to save dataset: {e}") from e

    def load_dataset(
        self,
        ticker: str,
        timeframe: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Загрузить датасет из хранилища.

        Args:
            ticker: Тикер инструмента
            timeframe: Временной интервал
            from_date: Начальная дата (если None, с начала)
            to_date: Конечная дата (если None, до конца)

        Returns:
            DataFrame с OHLCV данными

        Raises:
            StorageError: При ошибке загрузки
        """
        try:
            dataset_dir = self.data_dir / ticker / timeframe

            if not dataset_dir.exists():
                raise StorageError(f"Dataset not found: {ticker}/{timeframe}")

            # Найти все файлы парquet
            parquet_files = sorted(dataset_dir.glob("*.parquet"))

            if not parquet_files:
                raise StorageError(f"No data files found for {ticker}/{timeframe}")

            # Загрузить все файлы
            dfs = []
            for file_path in parquet_files:
                df = pd.read_parquet(file_path)
                dfs.append(df)

            # Объединить
            data = pd.concat(dfs, ignore_index=True)
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data = data.sort_values("timestamp").reset_index(drop=True)

            # Применить фильтр по датам
            if from_date or to_date:
                mask = pd.Series([True] * len(data))
                if from_date:
                    if from_date.tzinfo is None:
                        from_date = from_date.replace(
                            tzinfo=pd.Timestamp("now", tz="UTC").tzinfo
                        )
                    mask &= data["timestamp"] >= from_date
                if to_date:
                    if to_date.tzinfo is None:
                        to_date = to_date.replace(
                            tzinfo=pd.Timestamp("now", tz="UTC").tzinfo
                        )
                    mask &= data["timestamp"] <= to_date
                data = data[mask].copy()

            logger.info(f"Loaded dataset: {ticker}/{timeframe}, {len(data)} bars")
            return data

        except Exception as e:
            raise StorageError(f"Failed to load dataset: {e}") from e

    def append_data(
        self,
        data: pd.DataFrame,
        ticker: str,
        timeframe: TimeframeLiteral,
        *,
        source: Optional[SourceLiteral] = None,
    ) -> DatasetMetadata:
        """
        Добавить новые данные к существующему датасету.

        Args:
            data: Новые данные для добавления
            ticker: Тикер
            timeframe: Таймфрейм

        Returns:
            Обновлённые метаданные
        """
        try:
            # Загрузить существующие данные
            existing_metadata: Optional[DatasetMetadata] = None
            try:
                existing_data = self.load_dataset(ticker, timeframe)
                existing_metadata = self.get_metadata(ticker, timeframe)

                combined = pd.concat([existing_data, data], ignore_index=True)
                combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
                combined = combined.sort_values("timestamp").reset_index(drop=True)

            except StorageError:
                combined = data

            metadata = self.save_dataset(
                combined,
                ticker,
                timeframe,
                source=source
                or (existing_metadata.source if existing_metadata else "local"),
            )

            logger.info(f"Appended data to {ticker}/{timeframe}")
            return metadata

        except Exception as e:
            raise StorageError(f"Failed to append data: {e}") from e

    def list_datasets(
        self, ticker: Optional[str] = None, timeframe: Optional[str] = None
    ) -> list[DatasetMetadata]:
        """
        Получить список всех датасетов.

        Args:
            ticker: Фильтр по тикеру (опционально)
            timeframe: Фильтр по таймфрейму (опционально)

        Returns:
            Список метаданных датасетов
        """
        datasets = []

        # Определить директории для поиска
        if ticker:
            ticker_dirs = (
                [self.data_dir / ticker] if (self.data_dir / ticker).exists() else []
            )
        else:
            ticker_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        for ticker_dir in ticker_dirs:
            if timeframe:
                timeframe_dirs = (
                    [ticker_dir / timeframe]
                    if (ticker_dir / timeframe).exists()
                    else []
                )
            else:
                timeframe_dirs = [d for d in ticker_dir.iterdir() if d.is_dir()]

            for timeframe_dir in timeframe_dirs:
                metadata_file = timeframe_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata_dict = json.load(f)
                        metadata = DatasetMetadata(**metadata_dict)
                        datasets.append(metadata)
                    except Exception as e:
                        logger.warning(
                            f"Failed to load metadata from {metadata_file}: {e}"
                        )

        return datasets

    def get_metadata(self, ticker: str, timeframe: str) -> DatasetMetadata:
        """
        Получить метаданные датасета.

        Args:
            ticker: Тикер
            timeframe: Таймфрейм

        Returns:
            Метаданные датасета

        Raises:
            StorageError: Если метаданные не найдены
        """
        metadata_file = self.data_dir / ticker / timeframe / "metadata.json"

        if not metadata_file.exists():
            raise StorageError(f"Metadata not found for {ticker}/{timeframe}")

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)
            return DatasetMetadata(**metadata_dict)
        except Exception as e:
            raise StorageError(f"Failed to load metadata: {e}") from e

    def delete_dataset(self, ticker: str, timeframe: str) -> None:
        """
        Удалить датасет из хранилища.

        Args:
            ticker: Тикер
            timeframe: Таймфрейм
        """
        dataset_dir = self.data_dir / ticker / timeframe

        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            logger.info(f"Deleted dataset: {ticker}/{timeframe}")
        else:
            logger.warning(f"Dataset not found: {ticker}/{timeframe}")

    def clean_temp_artifacts(self, older_than_days: int = 7) -> dict[str, int]:
        """
        Очистить временные артефакты (downloads, extracted).

        Args:
            older_than_days: Удалить файлы старше указанного количества дней

        Returns:
            Статистика удалённых файлов
        """
        from datetime import timedelta

        now = datetime.now()
        threshold = now - timedelta(days=older_than_days)

        stats = {"downloads": 0, "extracted": 0}

        # Очистить downloads
        for item in self.downloads_dir.rglob("*"):
            if item.is_file():
                mtime = datetime.fromtimestamp(item.stat().st_mtime)
                if mtime < threshold:
                    item.unlink()
                    stats["downloads"] += 1

        # Очистить extracted
        for item in self.extracted_dir.rglob("*"):
            if item.is_file():
                mtime = datetime.fromtimestamp(item.stat().st_mtime)
                if mtime < threshold:
                    item.unlink()
                    stats["extracted"] += 1

        logger.info(f"Cleaned temp artifacts: {stats}")
        return stats

    def _create_metadata(
        self,
        data: pd.DataFrame,
        ticker: str,
        timeframe: TimeframeLiteral,
        source: SourceLiteral,
    ) -> DatasetMetadata:
        """Создать метаданные для датасета."""
        data_hash = self._compute_hash(data)

        timestamps = pd.to_datetime(data["timestamp"])

        return DatasetMetadata(
            ticker=ticker,
            timeframe=timeframe,
            start_date=timestamps.min(),
            end_date=timestamps.max(),
            source=source,
            total_bars=len(data),
            missing_bars=0,  # Будет заполнено валидатором
            hash=data_hash,
            created_at=datetime.utcnow(),
            schema_version="1.0",
        )

    def _compute_hash(self, data: pd.DataFrame) -> str:
        """Вычислить SHA256 хэш данных."""
        # Использовать только критичные колонки для хэша
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        hash_data = data[cols].to_json(orient="records", date_format="iso")
        return hashlib.sha256(hash_data.encode()).hexdigest()
