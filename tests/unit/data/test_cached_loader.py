"""Тесты для кэширующего загрузчика данных."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data.loaders.cached import CachedDataLoader


class MockLoader:
    """Мок-загрузчик для тестирования."""

    def __init__(self) -> None:
        self.call_count = 0

    def load(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime,
        timeframe: str = "1m",
        **kwargs: object,
    ) -> pd.DataFrame:
        """Генерировать тестовые данные."""
        self.call_count += 1

        # Генерировать простой DataFrame
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime([from_date, to_date], utc=True),
                "ticker": [ticker] * 2,
                "open": [100.0, 101.0],
                "high": [100.5, 101.5],
                "low": [99.5, 100.5],
                "close": [100.2, 101.2],
                "volume": [1000, 1100],
            }
        )

    def get_available_tickers(self) -> list[str]:
        """Вернуть список тикеров."""
        return ["TEST"]


class TestCachedDataLoader:
    """Тесты для двухуровневого кэша."""

    @pytest.fixture
    def mock_loader(self) -> MockLoader:
        """Мок-загрузчик."""
        return MockLoader()

    @pytest.fixture
    def cached_loader(
        self, mock_loader: MockLoader, tmp_path: Path
    ) -> CachedDataLoader:
        """Кэширующий загрузчик с временной директорией."""
        return CachedDataLoader(
            loader=mock_loader,
            cache_dir=tmp_path / "cache",
            ttl=3600,  # 1 час
            max_memory_size=2,  # Маленький размер для тестирования LRU
        )

    def test_first_load_misses_both_caches(
        self, cached_loader: CachedDataLoader, mock_loader: MockLoader
    ) -> None:
        """Первая загрузка - промах обоих кэшей."""
        from_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 2, tzinfo=timezone.utc)

        result = cached_loader.load("TEST", from_date, to_date, "1m")

        assert len(result) == 2
        assert mock_loader.call_count == 1
        assert cached_loader.stats["memory_misses"] == 1
        assert cached_loader.stats["disk_misses"] == 1

    def test_second_load_hits_memory_cache(
        self, cached_loader: CachedDataLoader, mock_loader: MockLoader
    ) -> None:
        """Вторая загрузка тех же данных - попадание в memory cache."""
        from_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 2, tzinfo=timezone.utc)

        # Первая загрузка
        result1 = cached_loader.load("TEST", from_date, to_date, "1m")

        # Вторая загрузка
        result2 = cached_loader.load("TEST", from_date, to_date, "1m")

        assert len(result2) == 2
        assert mock_loader.call_count == 1  # Только один вызов источника
        assert cached_loader.stats["memory_hits"] == 1
        assert cached_loader.stats["memory_misses"] == 1

        pd.testing.assert_frame_equal(result1, result2)

    def test_lru_eviction(
        self, cached_loader: CachedDataLoader, mock_loader: MockLoader
    ) -> None:
        """Тест вытеснения из LRU кэша при переполнении."""
        # max_memory_size=2, загрузим 3 разных датасета

        date1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        date2 = datetime(2024, 1, 2, tzinfo=timezone.utc)
        date3 = datetime(2024, 1, 3, tzinfo=timezone.utc)
        date4 = datetime(2024, 1, 4, tzinfo=timezone.utc)

        # Загрузка 1
        cached_loader.load("TEST", date1, date2, "1m")
        # Загрузка 2
        cached_loader.load("TEST", date2, date3, "1m")
        # Загрузка 3 - должна вытеснить первую
        cached_loader.load("TEST", date3, date4, "1m")

        # Проверить размер memory cache
        assert cached_loader.get_stats()["memory_cache_size"] == 2

        # Повторная загрузка первого датасета должна быть из disk cache
        cached_loader.load("TEST", date1, date2, "1m")

        stats = cached_loader.get_stats()
        # Должен быть 1 disk hit (для первого датасета после вытеснения из памяти)
        assert stats["disk_hits"] == 1

    def test_disk_cache_hit_after_memory_clear(
        self, cached_loader: CachedDataLoader, mock_loader: MockLoader
    ) -> None:
        """Тест попадания в disk cache после очистки memory cache."""
        from_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 2, tzinfo=timezone.utc)

        # Первая загрузка - сохраняет в оба кэша
        cached_loader.load("TEST", from_date, to_date, "1m")

        # Очистить только memory cache напрямую (оставить disk cache)
        cached_loader._memory_cache.clear()

        # Вторая загрузка - должна взять из disk cache и сохранить обратно в memory
        result_data = cached_loader.load("TEST", from_date, to_date, "1m")

        assert len(result_data) == 2
        assert mock_loader.call_count == 1  # Только один вызов источника
        assert cached_loader.stats["disk_hits"] == 1
        assert (
            cached_loader.stats["memory_misses"] == 2
        )  # Обе загрузки промахнулись по памяти

    def test_ttl_expiration(
        self, cached_loader: CachedDataLoader, mock_loader: MockLoader, tmp_path: Path
    ) -> None:
        """Тест истечения TTL для memory cache."""
        # Создать новый loader с очень коротким TTL
        short_ttl_loader = CachedDataLoader(
            loader=mock_loader,
            cache_dir=tmp_path / "cache2",
            ttl=1,  # 1 секунда
            max_memory_size=10,
        )

        from_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 2, tzinfo=timezone.utc)

        # Первая загрузка
        short_ttl_loader.load("TEST", from_date, to_date, "1m")

        # Подождать немного больше TTL
        import time

        time.sleep(1.5)

        # Вторая загрузка - память истекла, должна быть из источника
        short_ttl_loader.load("TEST", from_date, to_date, "1m")

        # Должно быть 2 вызова источника
        assert mock_loader.call_count == 2

    def test_cache_disabled(
        self, cached_loader: CachedDataLoader, mock_loader: MockLoader
    ) -> None:
        """Тест отключения кэша."""
        from_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 2, tzinfo=timezone.utc)

        # Две загрузки с отключенным кэшем
        cached_loader.load("TEST", from_date, to_date, "1m", use_cache=False)
        cached_loader.load("TEST", from_date, to_date, "1m", use_cache=False)

        # Должно быть 2 вызова источника
        assert mock_loader.call_count == 2
        assert cached_loader.stats["memory_hits"] == 0
        assert cached_loader.stats["disk_hits"] == 0

    def test_get_stats(
        self, cached_loader: CachedDataLoader, mock_loader: MockLoader
    ) -> None:
        """Тест получения статистики."""
        from_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 2, tzinfo=timezone.utc)

        # Первая загрузка - промах
        cached_loader.load("TEST", from_date, to_date, "1m")

        # Вторая загрузка - попадание в memory
        cached_loader.load("TEST", from_date, to_date, "1m")

        stats = cached_loader.get_stats()

        assert "memory_hits" in stats
        assert "memory_misses" in stats
        assert "disk_hits" in stats
        assert "disk_misses" in stats
        assert "memory_hit_rate" in stats
        assert "disk_hit_rate" in stats
        assert "overall_hit_rate" in stats
        assert "memory_cache_size" in stats
        assert "disk_cache_size" in stats

        assert stats["memory_hits"] == 1
        assert stats["memory_misses"] == 1
        assert stats["disk_misses"] == 1

    def test_clear_cache(
        self, cached_loader: CachedDataLoader, mock_loader: MockLoader
    ) -> None:
        """Тест очистки кэша."""
        from_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 1, 2, tzinfo=timezone.utc)

        # Загрузить данные
        cached_loader.load("TEST", from_date, to_date, "1m")

        # Очистить оба кэша
        result = cached_loader.clear_cache()

        assert result["memory"] == 1
        assert result["disk"] == 1

        # Проверить что кэши пусты
        stats = cached_loader.get_stats()
        assert stats["memory_cache_size"] == 0
        assert stats["disk_cache_size"] == 0

    def test_get_available_tickers(
        self, cached_loader: CachedDataLoader, mock_loader: MockLoader
    ) -> None:
        """Тест делегирования метода get_available_tickers."""
        tickers = cached_loader.get_available_tickers()
        assert tickers == ["TEST"]
