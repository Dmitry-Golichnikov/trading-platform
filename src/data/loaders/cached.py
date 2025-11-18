"""
Обёртка-декоратор для кэширования загруженных данных.
"""

import hashlib
import logging
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.common.exceptions import DataLoadError
from src.data.loaders.base import DataLoader

logger = logging.getLogger(__name__)


class CachedDataLoader:
    """
    Кэширующая обёртка над любым DataLoader.

    Предоставляет два уровня кэширования:
    1. In-memory LRU cache для быстрого доступа к часто используемым данным
    2. Disk cache (Parquet) для персистентного хранения

    Attributes:
        loader: Базовый загрузчик данных
        cache_dir: Директория для дискового кэша
        ttl: Time-to-live для кэша (секунды)
        max_memory_size: Максимальный размер LRU кэша в памяти
    """

    def __init__(
        self,
        loader: DataLoader,
        cache_dir: Optional[Path] = None,
        ttl: int = 86400,  # 24 часа
        max_memory_size: int = 128,
    ):
        """
        Инициализировать кэширующий загрузчик.

        Args:
            loader: Базовый загрузчик данных
            cache_dir: Директория для кэша (по умолчанию artifacts/cache/)
            ttl: Время жизни кэша в секундах (по умолчанию 24 часа)
            max_memory_size: Размер LRU кэша в памяти
        """
        self.loader = loader
        self.cache_dir = cache_dir or Path("artifacts/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(seconds=ttl)
        self.max_memory_size = max_memory_size

        # LRU in-memory cache для горячих данных
        self._memory_cache: OrderedDict[str, tuple[pd.DataFrame, datetime]] = OrderedDict()

        # Статистика кэша
        self.stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "disk_hits": 0,
            "disk_misses": 0,
        }

        logger.info(
            f"CachedDataLoader initialized: cache_dir={self.cache_dir}, "
            f"ttl={ttl}s, max_memory_size={max_memory_size}"
        )

    def load(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime,
        timeframe: str = "1m",
        use_cache: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Загрузить данные с использованием кэша.

        Args:
            ticker: Тикер инструмента
            from_date: Начальная дата
            to_date: Конечная дата
            timeframe: Временной интервал
            use_cache: Использовать кэш (можно отключить для принудительной загрузки)
            **kwargs: Дополнительные параметры

        Returns:
            DataFrame с OHLCV данными
        """
        if not use_cache:
            logger.debug("Cache disabled, loading directly")
            return self.loader.load(ticker, from_date, to_date, timeframe, **kwargs)

        # Генерировать ключ кэша
        cache_key = self._generate_cache_key(ticker, from_date, to_date, timeframe)

        # 1. Попытка загрузки из in-memory кэша (самый быстрый)
        memory_result = self._get_from_memory(cache_key)
        if memory_result is not None:
            logger.debug(f"Memory cache hit: {cache_key}")
            self.stats["memory_hits"] += 1
            return memory_result

        self.stats["memory_misses"] += 1

        # 2. Попытка загрузки из дискового кэша
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        if cache_path.exists():
            # Проверить TTL
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if datetime.now() - mtime < self.ttl:
                try:
                    logger.debug(f"Loading from disk cache: {cache_key}")
                    df = pd.read_parquet(cache_path)
                    self.stats["disk_hits"] += 1

                    # Сохранить в memory cache для последующих запросов
                    self._save_to_memory(cache_key, df)

                    return df
                except Exception as e:
                    logger.warning(f"Failed to load from cache: {e}")
            else:
                logger.debug(f"Cache expired for {cache_key}")

        # Кэш промах - загрузить из источника
        logger.debug(f"Cache miss for {cache_key}, loading from source")
        self.stats["disk_misses"] += 1

        try:
            df = self.loader.load(ticker, from_date, to_date, timeframe, **kwargs)

            # Сохранить в оба кэша
            self._save_to_cache(df, cache_path)
            self._save_to_memory(cache_key, df)

            return df

        except Exception as e:
            raise DataLoadError(f"Failed to load data: {e}") from e

    def _generate_cache_key(self, ticker: str, from_date: datetime, to_date: datetime, timeframe: str) -> str:
        """
        Генерировать уникальный ключ кэша.

        Args:
            ticker: Тикер
            from_date: Начальная дата
            to_date: Конечная дата
            timeframe: Таймфрейм

        Returns:
            MD5 хэш параметров
        """
        key_str = f"{ticker}_{from_date.isoformat()}_{to_date.isoformat()}_{timeframe}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_from_memory(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Получить данные из in-memory кэша.

        Args:
            cache_key: Ключ кэша

        Returns:
            DataFrame или None если нет в кэше или истёк TTL
        """
        if cache_key not in self._memory_cache:
            return None

        df, cached_at = self._memory_cache[cache_key]

        # Проверить TTL
        if datetime.now() - cached_at > self.ttl:
            # Удалить устаревший элемент
            del self._memory_cache[cache_key]
            return None

        # Переместить в конец (LRU: most recently used)
        self._memory_cache.move_to_end(cache_key)
        return df.copy()  # Вернуть копию для безопасности

    def _save_to_memory(self, cache_key: str, df: pd.DataFrame) -> None:
        """
        Сохранить данные в in-memory LRU кэш.

        Args:
            cache_key: Ключ кэша
            df: DataFrame для сохранения
        """
        # Проверить размер кэша и удалить самый старый элемент если переполнен
        if len(self._memory_cache) >= self.max_memory_size:
            # Удалить первый элемент (least recently used)
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
            logger.debug(f"Evicted LRU item: {oldest_key}")

        # Добавить новый элемент
        self._memory_cache[cache_key] = (df.copy(), datetime.now())

    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path) -> None:
        """
        Сохранить DataFrame в дисковый кэш.

        Args:
            df: DataFrame для сохранения
            cache_path: Путь к файлу кэша
        """
        try:
            df.to_parquet(cache_path, compression="snappy", index=False)
            logger.debug(f"Saved to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def clear_cache(self, older_than: Optional[timedelta] = None, clear_memory: bool = True) -> dict[str, int]:
        """
        Очистить кэш.

        Args:
            older_than: Удалить файлы старше указанного времени (если None, удалить все)
            clear_memory: Очистить также in-memory кэш

        Returns:
            Словарь с количеством удалённых элементов
        """
        disk_count = 0
        memory_count = 0
        now = datetime.now()

        # Очистить дисковый кэш
        for cache_file in self.cache_dir.glob("*.parquet"):
            if older_than is None:
                cache_file.unlink()
                disk_count += 1
            else:
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if now - mtime > older_than:
                    cache_file.unlink()
                    disk_count += 1

        # Очистить memory кэш
        if clear_memory:
            memory_count = len(self._memory_cache)
            self._memory_cache.clear()

        logger.info(f"Cleared cache: {disk_count} disk files, {memory_count} memory items")
        return {"disk": disk_count, "memory": memory_count}

    def get_stats(self) -> dict:
        """
        Получить статистику использования кэша.

        Returns:
            Словарь со статистикой
        """
        total_requests = (
            self.stats["memory_hits"]
            + self.stats["memory_misses"]
            + self.stats["disk_hits"]
            + self.stats["disk_misses"]
        )
        memory_hit_rate = self.stats["memory_hits"] / total_requests * 100 if total_requests > 0 else 0
        disk_hit_rate = (
            self.stats["disk_hits"] / (self.stats["disk_hits"] + self.stats["disk_misses"]) * 100
            if (self.stats["disk_hits"] + self.stats["disk_misses"]) > 0
            else 0
        )
        overall_hit_rate = (
            (self.stats["memory_hits"] + self.stats["disk_hits"]) / total_requests * 100 if total_requests > 0 else 0
        )

        return {
            **self.stats,
            "memory_hit_rate": f"{memory_hit_rate:.2f}%",
            "disk_hit_rate": f"{disk_hit_rate:.2f}%",
            "overall_hit_rate": f"{overall_hit_rate:.2f}%",
            "memory_cache_size": len(self._memory_cache),
            "disk_cache_size": len(list(self.cache_dir.glob("*.parquet"))),
        }

    def get_available_tickers(self) -> list[str]:
        """Делегировать к базовому загрузчику."""
        return self.loader.get_available_tickers()
