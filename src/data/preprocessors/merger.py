"""
Объединение данных из разных источников.
"""

import logging
from typing import Literal, Sequence

import pandas as pd

from src.common.exceptions import PreprocessingError
from src.data.validators.integrity import IntegrityValidator
from src.data.validators.schema import SchemaValidator

logger = logging.getLogger(__name__)

MergeStrategy = Literal["latest", "priority", "average"]
FillMethod = Literal["forward_fill", "backward_fill", "interpolate", "drop"]


class DataMerger:
    """
    Класс для объединения данных из разных источников.

    Поддерживает:
    - Merge по timestamp
    - Разрешение конфликтов по приоритету источника
    - Заполнение пропусков
    - Валидация после объединения
    """

    def __init__(self) -> None:
        """Инициализировать merger."""
        self.schema_validator = SchemaValidator()
        self.integrity_validator = IntegrityValidator()

    def merge_sources(
        self,
        sources: Sequence[pd.DataFrame],
        strategy: MergeStrategy = "latest",
        priority_order: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Объединить данные из нескольких источников.

        Args:
            sources: Список DataFrame'ов с OHLCV данными
            strategy: Стратегия разрешения конфликтов:
                - 'latest': брать самые свежие данные по created_at/modified
                - 'priority': по явному приоритету источников
                - 'average': усреднять конфликтующие значения
            priority_order: Порядок приоритета источников (для strategy='priority')
                Например: ['tinkoff', 'local', 'manual']

        Returns:
            Объединённый DataFrame

        Raises:
            PreprocessingError: При ошибке объединения
        """
        if not sources:
            raise PreprocessingError("No sources provided for merging")

        # Проверить priority_order для стратегии priority
        if strategy == "priority" and priority_order is None:
            raise PreprocessingError(
                "priority_order must be specified for 'priority' strategy"
            )

        if len(sources) == 1:
            logger.debug("Only one source, returning as-is")
            return sources[0].copy()

        try:
            # Валидация всех источников
            for idx, source in enumerate(sources):
                result = self.schema_validator.validate_columns(source)
                if not result.is_valid:
                    raise PreprocessingError(
                        f"Source {idx} failed schema validation: {result.errors}"
                    )

            # Объединить все источники
            combined = pd.concat(sources, ignore_index=True)

            # Разрешить конфликты
            if strategy == "priority":
                assert priority_order is not None  # Already checked above
                merged = self._resolve_priority(combined, priority_order)
            elif strategy == "latest":
                merged = self._resolve_latest(combined)
            elif strategy == "average":
                merged = self._resolve_average(combined)
            else:
                raise PreprocessingError(f"Unknown merge strategy: {strategy}")

            logger.info(
                f"Merged {len(sources)} sources: {len(combined)} → {len(merged)} bars"
            )
            return merged

        except Exception as e:
            raise PreprocessingError(f"Failed to merge sources: {e}") from e

    def resolve_conflicts(
        self, data: pd.DataFrame, priority_order: Sequence[str]
    ) -> pd.DataFrame:
        """
        Разрешить конфликты по приоритету источников.

        Args:
            data: DataFrame с возможными дубликатами timestamp
            priority_order: Порядок приоритета по колонке 'source'

        Returns:
            DataFrame без дубликатов
        """
        return self._resolve_priority(data, priority_order)

    def fill_gaps(
        self,
        data: pd.DataFrame,
        method: FillMethod = "forward_fill",
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Заполнить пропуски в данных.

        Args:
            data: DataFrame с пропусками
            method: Метод заполнения:
                - 'forward_fill': заполнить предыдущим значением
                - 'backward_fill': заполнить следующим значением
                - 'interpolate': линейная интерполяция
                - 'drop': удалить строки с пропусками
            limit: Максимальное количество последовательных пропусков для заполнения

        Returns:
            DataFrame с заполненными пропусками
        """
        if data.empty:
            return data

        result = data.copy()

        if method == "drop":
            result = result.dropna()
            logger.debug(f"Dropped {len(data) - len(result)} rows with NaN values")
            return result

        # Колонки для заполнения (не timestamp и ticker)
        fill_cols = ["open", "high", "low", "close", "volume"]
        fill_cols = [col for col in fill_cols if col in result.columns]

        if method == "forward_fill":
            result[fill_cols] = result[fill_cols].ffill(limit=limit)
        elif method == "backward_fill":
            result[fill_cols] = result[fill_cols].bfill(limit=limit)
        elif method == "interpolate":
            result[fill_cols] = result[fill_cols].interpolate(
                method="linear", limit=limit
            )
        else:
            raise PreprocessingError(f"Unknown fill method: {method}")

        # Удалить оставшиеся NaN
        result = result.dropna()

        logger.info(
            f"Filled gaps using {method}: {len(data)} → {len(result)} bars remaining"
        )
        return result

    def validate_after_merge(self, data: pd.DataFrame, timeframe: str) -> None:
        """
        Валидировать данные после объединения.

        Args:
            data: Объединённые данные
            timeframe: Таймфрейм для проверки

        Raises:
            PreprocessingError: Если валидация не прошла
        """
        # Проверка схемы
        schema_result = self.schema_validator.validate_all(data)
        if not schema_result.is_valid:
            raise PreprocessingError(
                f"Merged data failed schema validation: {schema_result.errors}"
            )

        # Проверка целостности
        integrity_result = self.integrity_validator.validate_all(data, timeframe)
        if not integrity_result.is_valid:
            raise PreprocessingError(
                f"Merged data failed integrity validation: {integrity_result.errors}"
            )

        logger.info("Merged data passed validation")

    def _resolve_latest(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Разрешить конфликты, оставляя самые свежие записи.

        Предполагается, что последние записи в DataFrame - самые свежие.
        """
        # Сортировать по timestamp и оставить последнее значение для каждого timestamp
        result = data.sort_values("timestamp").drop_duplicates(
            subset=["timestamp", "ticker"], keep="last"
        )
        return result.reset_index(drop=True)

    def _resolve_priority(
        self, data: pd.DataFrame, priority_order: Sequence[str]
    ) -> pd.DataFrame:
        """
        Разрешить конфликты по приоритету источников.

        Args:
            data: DataFrame с колонкой 'source'
            priority_order: Список источников от высокого к низкому приоритету
        """
        if "source" not in data.columns:
            logger.warning("No 'source' column, using 'latest' strategy")
            return self._resolve_latest(data)

        # Создать маппинг приоритетов
        priority_map = {source: idx for idx, source in enumerate(priority_order)}

        # Добавить колонку приоритета
        data = data.copy()
        data["_priority"] = data["source"].map(priority_map)

        # Заменить неизвестные источники на низкий приоритет
        data["_priority"] = data["_priority"].fillna(len(priority_order))

        # Сортировать по timestamp и приоритету, оставить первую запись
        result = (
            data.sort_values(["timestamp", "_priority"])
            .drop_duplicates(subset=["timestamp", "ticker"], keep="first")
            .drop(columns=["_priority"])
        )

        return result.reset_index(drop=True)

    def _resolve_average(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Разрешить конфликты усреднением значений OHLCV.

        Для каждого timestamp:
        - open, high, low, close: среднее
        - volume: сумма
        """
        if data.empty:
            return data

        # Группировать по timestamp и ticker
        agg_dict = {
            "open": "mean",
            "high": "mean",
            "low": "mean",
            "close": "mean",
            "volume": "sum",
        }

        # Добавить колонки, которые есть в данных
        group_cols = ["timestamp"]
        if "ticker" in data.columns:
            group_cols.append("ticker")

        result = data.groupby(group_cols, as_index=False).agg(agg_dict)

        # Округлить volume до целого
        result["volume"] = result["volume"].round().astype("int64")

        logger.debug(f"Averaged conflicts: {len(data)} → {len(result)} bars")
        return result
