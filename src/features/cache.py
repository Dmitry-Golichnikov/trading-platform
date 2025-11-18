"""Кэширование вычисленных признаков."""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class FeatureCache:
    """Кэш вычисленных признаков с хранилищем на основе Parquet."""

    def __init__(
        self,
        cache_dir: Path = Path("artifacts/features"),
        catalog_path: Optional[Path] = None,
    ):
        """
        Инициализация кэша.

        Args:
            cache_dir: Директория для хранения кэшированных признаков
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Создаём каталог для метаданных (SQLite)
        if catalog_path is not None:
            self.catalog_path = Path(catalog_path)
        elif self.cache_dir == Path("artifacts/features"):
            self.catalog_path = self.cache_dir.parent / "db" / "feature_catalog.db"
        else:
            self.catalog_path = self.cache_dir / "feature_catalog.db"
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_catalog()

    def _init_catalog(self):
        """Инициализация каталога (SQLite базы)."""
        conn = sqlite3.connect(self.catalog_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                cache_key TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                file_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                num_features INTEGER,
                num_rows INTEGER,
                metadata TEXT
            )
        """
        )

        conn.commit()
        conn.close()

    def _compute_cache_key(self, dataset_id: str, feature_config: dict) -> str:
        """
        Вычислить ключ кэша на основе dataset_id и конфигурации.

        Args:
            dataset_id: Идентификатор датасета
            feature_config: Конфигурация признаков

        Returns:
            Хеш-ключ для кэша
        """
        # Сериализуем конфиг в JSON (детерминированно)
        config_str = json.dumps(feature_config, sort_keys=True)

        # Вычисляем хеш
        hash_input = f"{dataset_id}_{config_str}"
        cache_key = hashlib.sha256(hash_input.encode()).hexdigest()

        return cache_key

    def get(self, dataset_id: str, feature_config: dict) -> Optional[pd.DataFrame]:
        """
        Получить признаки из кэша.

        Args:
            dataset_id: Идентификатор датасета
            feature_config: Конфигурация признаков

        Returns:
            DataFrame с признаками или None если не найдено
        """
        cache_key = self._compute_cache_key(dataset_id, feature_config)

        # Ищем в каталоге
        conn = sqlite3.connect(self.catalog_path)
        cursor = conn.cursor()

        cursor.execute("SELECT file_path FROM features WHERE cache_key = ?", (cache_key,))
        result = cursor.fetchone()

        conn.close()

        if result is None:
            return None

        file_path = Path(result[0])

        if not file_path.exists():
            # Файл удалён, очищаем запись
            self._remove_from_catalog(cache_key)
            return None

        # Загружаем из Parquet
        try:
            features = pd.read_parquet(file_path)
            return features
        except Exception as e:
            print(f"Ошибка загрузки кэша: {e}")
            return None

    def save(
        self,
        dataset_id: str,
        feature_config: dict,
        features: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Сохранить признаки в кэш.

        Args:
            dataset_id: Идентификатор датасета
            feature_config: Конфигурация признаков
            features: DataFrame с признаками
            metadata: Дополнительные метаданные
        """
        cache_key = self._compute_cache_key(dataset_id, feature_config)
        config_hash = hashlib.sha256(json.dumps(feature_config, sort_keys=True).encode()).hexdigest()[:8]

        # Создаём поддиректорию для датасета
        dataset_dir = self.cache_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Путь к файлу
        file_path = dataset_dir / f"{config_hash}.parquet"

        # Сохраняем в Parquet
        features.to_parquet(file_path, compression="snappy", index=True)

        # Обновляем каталог
        conn = sqlite3.connect(self.catalog_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO features
            (cache_key, dataset_id, config_hash, file_path, created_at,
             num_features, num_rows, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                cache_key,
                dataset_id,
                config_hash,
                str(file_path),
                datetime.now().isoformat(),
                features.shape[1],
                features.shape[0],
                json.dumps(metadata) if metadata else None,
            ),
        )

        conn.commit()
        conn.close()

    def invalidate(self, dataset_id: Optional[str] = None):
        """
        Инвалидировать кэш.

        Args:
            dataset_id: Если указан, инвалидирует только этот датасет.
                       Иначе инвалидирует весь кэш.
        """
        conn = sqlite3.connect(self.catalog_path)
        cursor = conn.cursor()

        if dataset_id:
            # Получаем файлы для удаления
            cursor.execute(
                "SELECT file_path FROM features WHERE dataset_id = ?",
                (dataset_id,),
            )
            files = cursor.fetchall()

            # Удаляем записи из каталога
            cursor.execute("DELETE FROM features WHERE dataset_id = ?", (dataset_id,))

            # Удаляем файлы
            for file_path in files:
                path = Path(file_path[0])
                if path.exists():
                    path.unlink()

            # Удаляем директорию если пустая
            dataset_dir = self.cache_dir / dataset_id
            if dataset_dir.exists() and not any(dataset_dir.iterdir()):
                dataset_dir.rmdir()
        else:
            # Очищаем всё
            cursor.execute("DELETE FROM features")

            # Удаляем все файлы
            for dataset_dir in self.cache_dir.iterdir():
                if dataset_dir.is_dir():
                    for file in dataset_dir.iterdir():
                        file.unlink()
                    dataset_dir.rmdir()

        conn.commit()
        conn.close()

    def list_cached(self, dataset_id: Optional[str] = None) -> pd.DataFrame:
        """
        Список кэшированных признаков.

        Args:
            dataset_id: Фильтр по датасету (опционально)

        Returns:
            DataFrame со списком кэшированных признаков
        """
        conn = sqlite3.connect(self.catalog_path)

        if dataset_id:
            query = "SELECT * FROM features WHERE dataset_id = ?"
            df = pd.read_sql_query(query, conn, params=(dataset_id,))
        else:
            query = "SELECT * FROM features"
            df = pd.read_sql_query(query, conn)

        conn.close()

        return df

    def _remove_from_catalog(self, cache_key: str):
        """Удалить запись из каталога."""
        conn = sqlite3.connect(self.catalog_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM features WHERE cache_key = ?", (cache_key,))

        conn.commit()
        conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """
        Получить статистику кэша.

        Returns:
            Словарь со статистикой
        """
        conn = sqlite3.connect(self.catalog_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM features")
        total_entries = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT dataset_id) FROM features")
        total_datasets = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(num_features) FROM features")
        total_features = cursor.fetchone()[0] or 0

        cursor.execute("SELECT SUM(num_rows) FROM features")
        total_rows = cursor.fetchone()[0] or 0

        conn.close()

        # Размер на диске
        total_size = sum(f.stat().st_size for f in self.cache_dir.rglob("*.parquet") if f.is_file())
        size_mb = total_size / (1024 * 1024)

        return {
            "total_entries": total_entries,
            "total_datasets": total_datasets,
            "total_features": total_features,
            "total_rows": total_rows,
            "size_mb": size_mb,
        }
