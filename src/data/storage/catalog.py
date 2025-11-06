"""
Каталог датасетов с SQLite базой данных.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from src.common.exceptions import CatalogError
from src.data.schemas import DatasetMetadata

logger = logging.getLogger(__name__)


class DatasetCatalog:
    """
    Централизованный каталог всех датасетов.

    Использует SQLite для быстрого поиска и фильтрации.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Инициализировать каталог.

        Args:
            db_path: Путь к файлу БД (по умолчанию artifacts/db/catalog.db)
        """
        if db_path is None:
            db_path = Path("artifacts/db/catalog.db")

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()
        logger.info(f"DatasetCatalog initialized: db={self.db_path}")

    def _init_database(self) -> None:
        """Создать таблицы если их нет."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timezone TEXT NOT NULL,
                    total_bars INTEGER NOT NULL,
                    missing_bars INTEGER NOT NULL,
                    hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    schema_version TEXT NOT NULL
                )
            """
            )

            # Создать индексы
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON datasets(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timeframe ON datasets(timeframe)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_timeframe " "ON datasets(ticker, timeframe)")

            conn.commit()

    def add_dataset(self, metadata: DatasetMetadata) -> None:
        """Добавить или обновить датасет в каталоге."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    ("INSERT OR REPLACE INTO datasets VALUES " "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"),
                    (
                        str(metadata.dataset_id),
                        metadata.ticker,
                        metadata.timeframe,
                        metadata.start_date.isoformat(),
                        metadata.end_date.isoformat(),
                        metadata.source,
                        metadata.timezone,
                        metadata.total_bars,
                        metadata.missing_bars,
                        metadata.hash,
                        metadata.created_at.isoformat(),
                        metadata.schema_version,
                    ),
                )
                conn.commit()

            logger.debug(f"Added dataset to catalog: {metadata.ticker}/{metadata.timeframe}")

        except Exception as e:
            raise CatalogError(f"Failed to add dataset: {e}") from e

    def search(
        self,
        ticker: Optional[str] = None,
        timeframe: Optional[str] = None,
        source: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> list[DatasetMetadata]:
        """Найти датасеты по критериям."""
        query = "SELECT * FROM datasets WHERE 1=1"
        params = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)
        if source:
            query += " AND source = ?"
            params.append(source)
        if from_date:
            query += " AND end_date >= ?"
            params.append(from_date.isoformat())
        if to_date:
            query += " AND start_date <= ?"
            params.append(to_date.isoformat())

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

            results: list[DatasetMetadata] = []
            for row in rows:
                metadata = DatasetMetadata(
                    dataset_id=UUID(row["dataset_id"]),
                    ticker=row["ticker"],
                    timeframe=row["timeframe"],
                    start_date=datetime.fromisoformat(row["start_date"]),
                    end_date=datetime.fromisoformat(row["end_date"]),
                    source=row["source"],
                    timezone=row["timezone"],
                    total_bars=row["total_bars"],
                    missing_bars=row["missing_bars"],
                    hash=row["hash"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    schema_version=row["schema_version"],
                )
                results.append(metadata)

            return results

        except Exception as e:
            raise CatalogError(f"Search failed: {e}") from e

    def get_by_id(self, dataset_id: UUID) -> Optional[DatasetMetadata]:
        """Получить датасет по ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (str(dataset_id),))
            row = cursor.fetchone()

        if row:
            return DatasetMetadata(
                dataset_id=UUID(row["dataset_id"]),
                ticker=row["ticker"],
                timeframe=row["timeframe"],
                start_date=datetime.fromisoformat(row["start_date"]),
                end_date=datetime.fromisoformat(row["end_date"]),
                source=row["source"],
                timezone=row["timezone"],
                total_bars=row["total_bars"],
                missing_bars=row["missing_bars"],
                hash=row["hash"],
                created_at=datetime.fromisoformat(row["created_at"]),
                schema_version=row["schema_version"],
            )
        return None

    def delete(self, dataset_id: UUID) -> None:
        """Удалить датасет из каталога."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (str(dataset_id),))
            conn.commit()
        logger.debug(f"Deleted dataset from catalog: {dataset_id}")
