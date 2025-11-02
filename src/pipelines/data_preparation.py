"""Асинхронный пайплайн подготовки исторических данных."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, cast

import pandas as pd

from src.common.exceptions import DataLoadError, DataValidationError, StorageError
from src.data.loaders import TinkoffDataLoader
from src.data.loaders.base import DataLoader
from src.data.loaders.local_file import LocalFileLoader
from src.data.preprocessors.resampler import TimeframeResampler
from src.data.preprocessors.timezone import convert_to_utc
from src.data.schemas import DatasetConfig, DatasetMetadata
from src.data.storage.catalog import DatasetCatalog
from src.data.storage.parquet_storage import (
    ParquetStorage,
    SourceLiteral,
    TimeframeLiteral,
)
from src.data.storage.versioning import DataVersioning
from src.data.validators import IntegrityValidator, QualityValidator, SchemaValidator

logger = logging.getLogger(__name__)

UTC = timezone.utc
DEFAULT_TARGET_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h", "1d")


@dataclass
class PipelineResult:
    """Результат обработки тикера."""

    ticker: str
    metadata: list[DatasetMetadata]
    missing_years: list[int]
    warnings: list[str]


class DataPreparationPipeline:
    """
    Асинхронный пайплайн подготовки данных.

    Последовательность этапов:
    1. Формирование списка тикеров (конфиг или файл)
    2. Планирование загрузки по годам (учёт backfill и update latest)
    3. Асинхронная загрузка данных и предобработка
    4. Валидация (схема, целостность, качество)
    5. Ресэмплинг в старшие таймфреймы
    6. Сохранение в Parquet и обновление каталога
    7. Версионирование и отчёт о полноте
    """

    def __init__(
        self,
        config: DatasetConfig,
        *,
        loader: Optional[DataLoader] = None,
        storage: Optional[ParquetStorage] = None,
        catalog: Optional[DatasetCatalog] = None,
        versioning: Optional[DataVersioning] = None,
        target_timeframes: Sequence[str] = DEFAULT_TARGET_TIMEFRAMES,
        concurrency: int = 1,
        progress_callback: Optional[Callable[[str, str, int, int], None]] = None,
    ) -> None:
        self.config = config
        self.loader = loader or self._create_loader()
        self.storage = storage or ParquetStorage()
        self.catalog = catalog or DatasetCatalog()
        self.versioning = versioning or DataVersioning()
        self.resampler = TimeframeResampler()
        self.schema_validator = SchemaValidator()
        self.integrity_validator = IntegrityValidator()
        self.quality_validator = QualityValidator()

        merged_timeframes = list(dict.fromkeys(target_timeframes))
        if self.config.timeframe not in merged_timeframes:
            merged_timeframes.append(self.config.timeframe)
        self.target_timeframes = tuple(dict.fromkeys(merged_timeframes))
        self.base_timeframe = self._detect_base_timeframe()
        self.concurrency = max(1, concurrency)
        self._semaphore = asyncio.Semaphore(self.concurrency)
        self.progress_callback = progress_callback
        self._warning_buffer: dict[str, list[str]] = defaultdict(list)

    def _notify_progress(
        self, ticker: str, stage: str, completed: int, total: int
    ) -> None:
        if not self.progress_callback:
            return
        try:
            self.progress_callback(ticker, stage, completed, total)
        except Exception as exc:  # pragma: no cover - fallback path
            logger.debug("Progress callback failed: %s", exc)

    def _collect_warnings(self, ticker: str) -> list[str]:
        return self._warning_buffer.pop(ticker, [])

    async def run(self) -> list[PipelineResult]:
        """Запустить пайплайн для всех тикеров."""
        tickers = self._resolve_tickers()
        if not tickers:
            raise ValueError("No tickers provided in config")

        logger.info(
            "Starting data preparation for tickers=%s timeframe=%s",
            tickers,
            self.config.timeframe,
        )

        tasks = [self._process_ticker(ticker) for ticker in tickers]
        results: list[PipelineResult] = []

        for future in asyncio.as_completed(tasks):
            result = await future
            results.append(result)

        return results

    async def _process_ticker(self, ticker: str) -> PipelineResult:
        """Обработать данные для одного тикера."""
        self._warning_buffer.setdefault(ticker, [])
        (
            expected_years,
            existing_years,
            missing_years_initial,
        ) = self._detect_missing_years(ticker)

        years_to_process = self._plan_years(set(missing_years_initial), existing_years)
        total_years = len(years_to_process)

        logger.info(
            "Ticker %s: expected_years=%s, existing_years=%s, years_to_process=%s",
            ticker,
            sorted(expected_years),
            sorted(existing_years),
            years_to_process,
        )

        if not years_to_process:
            logger.info("Ticker %s already up to date", ticker)
            self._notify_progress(ticker, "done", 1, 1)
            return PipelineResult(
                ticker=ticker,
                metadata=[],
                missing_years=[],
                warnings=self._collect_warnings(ticker),
            )

        self._notify_progress(ticker, "start", 0, total_years)

        load_tasks = [
            asyncio.create_task(self._load_year_data(ticker, year))
            for year in years_to_process
        ]

        year_data: list[pd.DataFrame] = []
        completed = 0

        for task in asyncio.as_completed(load_tasks):
            try:
                df = await task
            except Exception:
                for pending in load_tasks:
                    if not pending.done():
                        pending.cancel()
                raise

            year_data.append(df)
            completed += 1
            self._notify_progress(ticker, "progress", completed, total_years)

        # Объединить данные
        frames = [df for df in year_data if not df.empty]
        if not frames:
            logger.warning("No data loaded for ticker %s (all years empty)", ticker)
            self._notify_progress(ticker, "done", completed, total_years or 1)
            return PipelineResult(
                ticker=ticker,
                metadata=[],
                missing_years=list(expected_years),
                warnings=self._collect_warnings(ticker),
            )

        combined = pd.concat(frames, ignore_index=True)

        metadata_records = self._persist_timeframes(ticker, combined)

        _, _, missing_years_set = self._detect_missing_years(ticker)
        missing_years = sorted(missing_years_set)

        if missing_years and self.config.backfill_missing:
            backfill_metadata = await self._backfill_missing(ticker, missing_years)
            if backfill_metadata:
                metadata_records.extend(backfill_metadata)
                _, _, missing_years_set = self._detect_missing_years(ticker)
                missing_years = sorted(missing_years_set)
        if missing_years and not self.config.backfill_missing:
            logger.warning(
                "Ticker %s still has missing years after processing "
                "(backfill disabled): %s",
                ticker,
                missing_years,
            )
        elif missing_years:
            logger.warning(
                "Ticker %s still has missing years after backfill attempt: %s",
                ticker,
                missing_years,
            )

        self._notify_progress(ticker, "done", completed, total_years or 1)

        return PipelineResult(
            ticker=ticker,
            metadata=metadata_records,
            missing_years=missing_years,
            warnings=self._collect_warnings(ticker),
        )

    async def _load_year_data(self, ticker: str, year: int) -> pd.DataFrame:
        """Асинхронно загрузить данные за конкретный год."""
        start = datetime(year, 1, 1, tzinfo=UTC)
        end = datetime(year, 12, 31, 23, 59, 59, tzinfo=UTC)

        async with self._semaphore:
            logger.debug(
                "Loading data for %s %s (%s-%s)",
                ticker,
                self.base_timeframe,
                start,
                end,
            )

            try:
                df = await asyncio.to_thread(
                    self.loader.load,
                    ticker,
                    start,
                    end,
                    self.base_timeframe,
                )
            except Exception as exc:  # noqa: BLE001
                raise DataLoadError(
                    f"Failed to load data for {ticker} {year}: {exc}"
                ) from exc

            if df is None:
                return pd.DataFrame()

            if df.empty:
                logger.warning("No data returned for %s %s", ticker, year)
                return pd.DataFrame()

            df = self._postprocess_dataframe(df, ticker)
            self._run_validations(
                df, self.base_timeframe, ticker, context=f"{ticker}:{year}"
            )
            return df

    def _persist_timeframes(
        self, ticker: str, base_data: pd.DataFrame
    ) -> list[DatasetMetadata]:
        """Сохранить данные базового и производных таймфреймов."""
        standardized_base = self._postprocess_dataframe(base_data, ticker)

        metadata_records: list[DatasetMetadata] = []

        if self.config.validate_data:
            self._run_validations(standardized_base, self.base_timeframe, ticker)

        base_metadata = self._save_dataset(
            data=standardized_base,
            ticker=ticker,
            timeframe=self.base_timeframe,
        )
        metadata_records.append(base_metadata)

        target_timeframes = [
            tf for tf in self.target_timeframes if tf != self.base_timeframe
        ]
        resampled_map = self.resampler.resample_multiple_timeframes(
            standardized_base,
            source_tf=self.base_timeframe,
            target_tfs=target_timeframes,
        )

        for timeframe, dataset in resampled_map.items():
            if timeframe == self.base_timeframe:
                continue
            if dataset.empty:
                logger.warning(
                    "Ticker %s timeframe %s produced empty dataset", ticker, timeframe
                )
                continue

            standardized_dataset = self._postprocess_dataframe(dataset, ticker)
            if self.config.validate_data:
                self._run_validations(
                    standardized_dataset,
                    timeframe,
                    ticker,
                    context=f"{ticker}:{timeframe}",
                )
            metadata = self._save_dataset(
                data=standardized_dataset,
                ticker=ticker,
                timeframe=timeframe,
            )
            metadata_records.append(metadata)

        return metadata_records

    def _postprocess_dataframe(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Привести DataFrame к стандартизованной форме."""
        if "timestamp" not in df.columns:
            raise DataValidationError("DataFrame must contain 'timestamp' column")

        df = df.copy()
        df = convert_to_utc(df)

        df["ticker"] = ticker
        columns = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise DataValidationError(f"Missing columns after load: {missing}")

        df = df[columns]
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df = df.reset_index(drop=True)
        return df

    def _run_validations(
        self,
        data: pd.DataFrame,
        timeframe: str,
        ticker: str,
        *,
        context: str | None = None,
    ) -> list[str]:
        """Запустить все валидаторы для датасета."""
        if not self.config.validate_data:
            return []
        context_info = f"[{context}]" if context else ""
        schema_result = self.schema_validator.validate_all(data)
        integrity_result = self.integrity_validator.validate_all(data, timeframe)
        price_anomalies = self.quality_validator.detect_price_anomalies(data)
        volume_quality = self.quality_validator.check_volume_sanity(data)
        spread_quality = self.quality_validator.check_spread(data)

        errors = schema_result.errors + integrity_result.errors
        if errors:
            raise DataValidationError(f"Validation failed {context_info}: {errors}")

        warnings = (
            schema_result.warnings
            + integrity_result.warnings
            + price_anomalies.warnings
            + volume_quality.warnings
            + spread_quality.warnings
        )
        for warning in warnings:
            base = "Validation warning"
            if context_info:
                base = f"{base} {context_info}"
            message = f"{base}: {warning}"
            logger.warning(message)
            self._warning_buffer[ticker].append(message)

        return warnings

    def _save_dataset(
        self, data: pd.DataFrame, ticker: str, timeframe: str
    ) -> DatasetMetadata:
        """Сохранить датасет, обновить каталог и версионирование."""
        source = self._resolve_source()

        timeframe_literal = cast(TimeframeLiteral, timeframe)

        try:
            self.storage.get_metadata(ticker, timeframe)
            metadata = self.storage.append_data(
                data,
                ticker,
                timeframe_literal,
                source=source,
            )
        except StorageError:
            metadata = self.storage.save_dataset(
                data=data,
                ticker=ticker,
                timeframe=timeframe_literal,
                source=source,
            )

        self.catalog.add_dataset(metadata)
        version_hash = self.versioning.save_version(
            data=data,
            ticker=ticker,
            timeframe=timeframe_literal,
            description=f"Pipeline run {datetime.utcnow().isoformat()}",
        )
        logger.info(
            "Saved dataset %s/%s -> %s (hash %s)",
            ticker,
            timeframe,
            metadata.dataset_id,
            version_hash[:8],
        )
        return metadata

    def _resolve_tickers(self) -> list[str]:
        """Получить список тикеров из конфигурации."""
        tickers: set[str] = set()

        if isinstance(self.config.ticker, str):
            tickers.add(self.config.ticker)
        elif isinstance(self.config.ticker, Iterable):
            tickers.update(self.config.ticker)

        if self.config.tickers_file:
            file_path = Path(self.config.tickers_file)
            if not file_path.exists():
                raise FileNotFoundError(f"Tickers file not found: {file_path}")
            for line in file_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    tickers.add(stripped)

        return sorted(map(str.upper, tickers))

    def _determine_expected_years(self, ticker: str) -> set[int]:
        """Определить набор годов, которые должны быть загружены."""
        start_year = self.config.from_date.year
        end_year = self.config.to_date.year
        listing_year: Optional[int] = None
        if hasattr(self.loader, "get_listing_year"):
            try:
                listing_year = getattr(self.loader, "get_listing_year")(ticker)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to resolve listing year for %s: %s", ticker, exc)

        if listing_year is not None:
            start_year = max(start_year, listing_year)

        if end_year < start_year:
            end_year = start_year

        return set(range(start_year, end_year + 1))

    def _list_existing_years(self, ticker: str, timeframe: str) -> set[int]:
        """Получить список годов, для которых уже есть данные."""
        dataset_dir = self.storage.data_dir / ticker / timeframe
        if not dataset_dir.exists():
            return set()

        years: set[int] = set()
        for file in dataset_dir.glob("*.parquet"):
            try:
                years.add(int(file.stem))
            except ValueError:
                continue
        return years

    def _detect_missing_years(self, ticker: str) -> tuple[set[int], set[int], set[int]]:
        """Определить ожидаемые, существующие и отсутствующие годы."""
        expected = self._determine_expected_years(ticker)
        existing = self._list_existing_years(ticker, self.base_timeframe)
        missing = expected - existing
        return expected, existing, missing

    def _plan_years(self, missing: set[int], _existing: set[int]) -> list[int]:
        """Определить список годов для загрузки."""
        planned = set(missing)
        planned = self._refresh_latest_year(planned)

        if not self.config.backfill_missing:
            latest_year = self.config.to_date.year
            planned = {year for year in planned if year == latest_year}

        return sorted(planned)

    def _refresh_latest_year(self, planned: set[int]) -> set[int]:
        """Добавить текущий год для обновления, если требуется."""
        if not self.config.update_latest_year:
            return planned
        current_year = self.config.to_date.year
        planned.add(current_year)
        return planned

    async def _backfill_missing(
        self, ticker: str, missing_years: Sequence[int]
    ) -> list[DatasetMetadata]:
        """Докачать отсутствующие годы и сохранить результаты."""
        if not missing_years:
            return []

        logger.info("Backfilling missing years for %s: %s", ticker, missing_years)

        frames: list[pd.DataFrame] = []
        for year in missing_years:
            df = await self._load_year_data(ticker, year)
            if df.empty:
                warning = (
                    f"Backfill skipped for {ticker} {year}: data unavailable or empty"
                )
                logger.warning(warning)
                self._warning_buffer[ticker].append(warning)
                continue
            frames.append(df)

        if not frames:
            return []

        combined = pd.concat(frames, ignore_index=True)
        return self._persist_timeframes(ticker, combined)

    def _detect_base_timeframe(self) -> str:
        """Определить исходный таймфрейм для загрузки."""
        if self.config.source_type == "api":
            if self.config.resample_from:
                return self.config.resample_from
            # REST API возвращает минутные свечи, поэтому используем
            # 1m как базовый период
            return "1m"

        if self.config.resample_from:
            return self.config.resample_from
        return self.config.timeframe

    def _resolve_source(self) -> SourceLiteral:
        """Определить источник для метаданных."""
        mapping: dict[str, SourceLiteral] = {
            "local": "local",
            "api": "tinkoff",
        }
        return mapping.get(self.config.source_type, "manual")

    def _create_loader(self) -> DataLoader:
        """Создать загрузчик на основе конфигурации."""
        if self.config.source_type == "local":
            base_path = None
            if self.config.file_path:
                base_path = Path(self.config.file_path).parent
            return LocalFileLoader(base_path=base_path)
        return TinkoffDataLoader(token=self.config.api_token)
