"""
Версионирование датасетов с хэшированием и историей изменений.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.common.exceptions import StorageError

logger = logging.getLogger(__name__)


class DataVersioning:
    """
    Управление версиями датасетов.

    Отслеживает изменения датасетов через хэширование,
    хранит историю версий и позволяет откатываться к предыдущим версиям.
    """

    def __init__(self, versions_dir: Optional[Path] = None):
        """
        Инициализировать систему версионирования.

        Args:
            versions_dir: Директория для хранения версий
                (по умолчанию artifacts/versions/)
        """
        self.versions_dir = versions_dir or Path("artifacts/versions")
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DataVersioning initialized: versions_dir=%s", self.versions_dir)

    def compute_hash(self, data: pd.DataFrame) -> str:
        """
        Вычислить SHA256 хэш данных.

        Args:
            data: DataFrame для хэширования

        Returns:
            SHA256 хэш в hex формате
        """
        # Использовать только данные, а не metadata
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        available_cols = [c for c in cols if c in data.columns]

        hash_data = data[available_cols].to_json(
            orient="records", date_format="iso", double_precision=10
        )
        return hashlib.sha256(hash_data.encode()).hexdigest()

    def save_version(
        self,
        data: pd.DataFrame,
        ticker: str,
        timeframe: str,
        description: Optional[str] = None,
    ) -> str:
        """
        Сохранить версию датасета.

        Args:
            data: DataFrame для сохранения
            ticker: Тикер
            timeframe: Таймфрейм
            description: Описание изменений

        Returns:
            Хэш версии
        """
        version_hash = self.compute_hash(data)
        version_dir = self.versions_dir / ticker / timeframe
        version_dir.mkdir(parents=True, exist_ok=True)

        version_file = version_dir / f"{version_hash}.parquet"

        # Сохранить данные если версии ещё нет
        if not version_file.exists():
            data.to_parquet(version_file, compression="snappy", index=False)

            # Сохранить метаданные версии
            metadata = {
                "hash": version_hash,
                "ticker": ticker,
                "timeframe": timeframe,
                "created_at": datetime.utcnow().isoformat(),
                "description": description or "",
                "total_bars": len(data),
            }

            metadata_file = version_dir / f"{version_hash}.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved version {version_hash[:8]} for {ticker}/{timeframe}")
        else:
            logger.debug(f"Version {version_hash[:8]} already exists")

        return version_hash

    def load_version(
        self, ticker: str, timeframe: str, version_hash: str
    ) -> pd.DataFrame:
        """
        Загрузить конкретную версию датасета.

        Args:
            ticker: Тикер
            timeframe: Таймфрейм
            version_hash: Хэш версии

        Returns:
            DataFrame с данными

        Raises:
            StorageError: Если версия не найдена
        """
        version_file = (
            self.versions_dir / ticker / timeframe / f"{version_hash}.parquet"
        )

        if not version_file.exists():
            raise StorageError(
                f"Version {version_hash} not found for {ticker}/{timeframe}"
            )

        return pd.read_parquet(version_file)

    def get_history(self, ticker: str, timeframe: str) -> list[dict]:
        """
        Получить историю версий датасета.

        Args:
            ticker: Тикер
            timeframe: Таймфрейм

        Returns:
            Список метаданных версий (отсортирован по дате)
        """
        version_dir = self.versions_dir / ticker / timeframe

        if not version_dir.exists():
            return []

        history = []
        for metadata_file in version_dir.glob("*.json"):
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                history.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load version metadata: {e}")

        # Сортировать по дате создания
        history.sort(key=lambda x: x["created_at"], reverse=True)
        return history

    def compare_versions(
        self, ticker: str, timeframe: str, hash1: str, hash2: str
    ) -> dict:
        """
        Сравнить две версии датасета.

        Args:
            ticker: Тикер
            timeframe: Таймфрейм
            hash1: Хэш первой версии
            hash2: Хэш второй версии

        Returns:
            Словарь с результатами сравнения
        """
        data1 = self.load_version(ticker, timeframe, hash1)
        data2 = self.load_version(ticker, timeframe, hash2)

        return {
            "hash1": hash1,
            "hash2": hash2,
            "bars_diff": len(data2) - len(data1),
            "date_range_1": (
                data1["timestamp"].min().isoformat(),
                data1["timestamp"].max().isoformat(),
            ),
            "date_range_2": (
                data2["timestamp"].min().isoformat(),
                data2["timestamp"].max().isoformat(),
            ),
            "are_equal": hash1 == hash2,
        }
