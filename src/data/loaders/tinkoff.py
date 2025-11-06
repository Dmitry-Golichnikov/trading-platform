"""Загрузчик исторических данных через Tinkoff Invest API (REST getHistory)."""

from __future__ import annotations

import io
import logging
import os
import threading
import time
import zipfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import httpx
import pandas as pd
from tinkoff.invest import Client
from tinkoff.invest.exceptions import RequestError

from src.common.exceptions import DataLoadError
from src.data.loaders.base import DataLoader

logger = logging.getLogger(__name__)

UTC = timezone.utc
HISTORY_ENDPOINT = "https://invest-public-api.tinkoff.ru/history-data"


@dataclass(frozen=True)
class InstrumentInfo:
    """Краткая информация об инструменте."""

    ticker: str
    figi: str
    instrument_id: str
    first_1min_candle: Optional[datetime] = None


class _RateLimiter:
    """Простая реализация rate limiter для ограничений API."""

    def __init__(self, max_calls: int, period_seconds: float = 60.0) -> None:
        self._max_calls = max_calls
        self._period = period_seconds
        self._lock = threading.Lock()
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self._period:
                    self._calls.popleft()

                if len(self._calls) < self._max_calls:
                    self._calls.append(now)
                    return

                wait_time = self._period - (now - self._calls[0])

            time.sleep(max(wait_time, 0.0))


class TinkoffDataLoader(DataLoader):
    """Загрузчик исторических данных через API Tinkoff Invest."""

    def __init__(
        self,
        token: Optional[str] = None,
        *,
        base_url: str = HISTORY_ENDPOINT,
        downloads_dir: Optional[Path] = None,
        extracted_dir: Optional[Path] = None,
        timeout: float = 60.0,
        cache_ttl_hours: int = 24,
        rate_limit_per_minute: int = 30,
        max_attempts: int = 3,
        backoff_seconds: float = 2.0,
        instrument_resolver: Optional[Callable[[str], InstrumentInfo]] = None,
        app_name: str = "trading-platform",
    ) -> None:
        env_token = os.getenv("TINKOFF_API_TOKEN") or os.getenv("TINKOFF_INVEST_TOKEN")
        self.token = token or env_token
        if not self.token:
            raise ValueError(
                "Tinkoff API token is required. Provide it explicitly or set "
                "the TINKOFF_INVEST_TOKEN environment variable."
            )

        self.base_url = base_url
        self.downloads_dir = (downloads_dir or Path("artifacts/raw/downloads")).resolve()
        self.extracted_dir = (extracted_dir or Path("artifacts/raw/extracted")).resolve()
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)

        self.timeout = timeout
        self.cache_ttl_hours = cache_ttl_hours
        self._rate_limiter = _RateLimiter(rate_limit_per_minute)
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._instrument_cache: dict[str, InstrumentInfo] = {}
        self._instrument_lock = threading.Lock()
        self._instrument_resolver = instrument_resolver
        self._app_name = app_name

        self._http_client = httpx.Client(
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.token}"},
        )

    # ------------------------------------------------------------------
    # DataLoader interface
    # ------------------------------------------------------------------

    def load(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime,
        timeframe: str = "1m",
        **_: object,
    ) -> pd.DataFrame:
        if timeframe != "1m":
            logger.warning(
                (
                    "TinkoffDataLoader возвращает только 1m свечи. "
                    "Запрошенный таймфрейм %s будет обработан на верхнем "
                    "уровне (resample)."
                ),
                timeframe,
            )

        instrument = self._get_instrument_info(ticker)

        start_year = from_date.year
        end_year = to_date.year
        if instrument.first_1min_candle:
            start_year = max(start_year, instrument.first_1min_candle.year)

        data_frames: list[pd.DataFrame] = []
        for year in range(start_year, end_year + 1):
            df = self._load_year(instrument, year, ticker)
            if not df.empty:
                data_frames.append(df)

        if not data_frames:
            logger.warning("No data downloaded for %s in %s-%s", ticker, start_year, end_year)
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "ticker",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            )

        data = pd.concat(data_frames, ignore_index=True)
        data.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)

        # Фильтрация по диапазону дат
        if from_date.tzinfo is None:
            from_date = from_date.replace(tzinfo=UTC)
        if to_date.tzinfo is None:
            to_date = to_date.replace(tzinfo=UTC)

        mask = (data["timestamp"] >= from_date) & (data["timestamp"] <= to_date)
        filtered = data.loc[mask].copy()
        filtered.sort_values("timestamp", inplace=True)
        filtered.reset_index(drop=True, inplace=True)
        return filtered

    def get_available_tickers(self) -> list[str]:
        if not self.downloads_dir.exists():
            return []
        return [p.name for p in self.downloads_dir.iterdir() if p.is_dir()]

    def get_listing_year(self, ticker: str) -> Optional[int]:
        info = self._get_instrument_info(ticker)
        if info.first_1min_candle:
            return info.first_1min_candle.year
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._http_client.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def _get_instrument_info(self, ticker: str) -> InstrumentInfo:
        with self._instrument_lock:
            cached = self._instrument_cache.get(ticker)
        if cached:
            return cached

        resolver = self._instrument_resolver or self._resolve_instrument_via_api
        info = resolver(ticker)
        if not isinstance(info, InstrumentInfo):
            raise DataLoadError("Instrument resolver must return InstrumentInfo instance")

        with self._instrument_lock:
            self._instrument_cache[ticker] = info
        return info

    def _resolve_instrument_via_api(self, ticker: str) -> InstrumentInfo:
        logger.info("Resolving instrument info for %s via Tinkoff API", ticker)
        try:
            assert self.token is not None
            with Client(self.token, app_name=self._app_name) as client:
                response = client.instruments.find_instrument(query=ticker)
        except RequestError as exc:  # pragma: no cover - network error path
            raise DataLoadError(f"Failed to resolve instrument '{ticker}': {exc}") from exc

        if not response.instruments:
            raise DataLoadError(f"Instrument '{ticker}' not found in Tinkoff catalog")

        # Предпочитаем инструменты MOEX (class_code начинается с TQ)
        candidates = [
            ins
            for ins in response.instruments
            if ins.ticker.upper() == ticker.upper() and ins.class_code.upper().startswith("TQ")
        ]

        if not candidates:
            # Fallback: любые совпадения по тикеру
            candidates = [ins for ins in response.instruments if ins.ticker.upper() == ticker.upper()]

        if not candidates:
            raise DataLoadError(f"No matching instruments found for ticker '{ticker}'")

        instrument = candidates[0]
        logger.debug(
            "Resolved %s to FIGI %s (class_code: %s)",
            ticker,
            instrument.figi,
            instrument.class_code,
        )

        first_candle = None

        first_min_ts = getattr(instrument, "first_1min_candle_date", None)
        if first_min_ts:
            try:
                first_candle = first_min_ts.ToDatetime().replace(tzinfo=UTC)
            except AttributeError:  # When already datetime
                first_candle = first_min_ts.replace(tzinfo=UTC)

        if not first_candle:
            first_day_ts = getattr(instrument, "first_1day_candle_date", None)
            if first_day_ts:
                try:
                    first_candle = first_day_ts.ToDatetime().replace(tzinfo=UTC)
                except AttributeError:
                    first_candle = first_day_ts.replace(tzinfo=UTC)

        return InstrumentInfo(
            ticker=ticker,
            figi=instrument.figi,
            instrument_id=instrument.uid or instrument.figi,
            first_1min_candle=first_candle,
        )

    def _load_year(self, instrument: InstrumentInfo, year: int, ticker: str) -> pd.DataFrame:
        try:
            archive_path = self._ensure_archive(instrument, year, ticker)
        except DataLoadError as exc:
            logger.warning("Failed to download archive for %s %s: %s", ticker, year, exc)
            return pd.DataFrame()
        if not archive_path.exists():
            return pd.DataFrame()

        try:
            return self._parse_archive(archive_path, ticker, year)
        except zipfile.BadZipFile as exc:
            raise DataLoadError(f"Corrupted archive {archive_path}") from exc

    def _ensure_archive(self, instrument: InstrumentInfo, year: int, ticker: str) -> Path:
        ticker_dir = self.downloads_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        archive_path = ticker_dir / f"{year}.zip"

        if archive_path.exists():
            age_hours = (time.time() - archive_path.stat().st_mtime) / 3600.0
            if age_hours <= self.cache_ttl_hours:
                logger.debug("Using cached archive %s", archive_path)
                return archive_path

        self._download_archive(instrument, year, archive_path)
        return archive_path

    def _download_archive(self, instrument: InstrumentInfo, year: int, target_path: Path) -> None:
        logger.info(
            "Downloading %s (%s) history for %s",
            instrument.ticker,
            instrument.figi,
            year,
        )
        # ВАЖНО: API Tinkoff ожидает параметр 'figi', а не 'instrument_id'
        params = {
            "figi": instrument.figi,
            "year": str(year),
        }

        attempt = 1
        while attempt <= self._max_attempts:
            self._rate_limiter.acquire()
            response = self._http_client.get(self.base_url, params=params)

            if response.status_code == 200 and response.content:
                target_path.write_bytes(response.content)
                return

            if response.status_code == 404:
                logger.warning("Archive not found for %s %s", instrument.ticker, year)
                return

            logger.warning(
                "Attempt %s/%s: failed to download %s %s (status %s)",
                attempt,
                self._max_attempts,
                instrument.ticker,
                year,
                response.status_code,
            )

            attempt += 1
            if attempt <= self._max_attempts:
                time.sleep(self._backoff_seconds * (attempt - 1))

        raise DataLoadError(
            "Failed to download history for %s %s: %s %s"
            % (instrument.ticker, year, response.status_code, response.text)
        )

    def _parse_archive(self, archive_path: Path, ticker: str, year: int) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue

                data = zf.read(info)
                if info.filename.endswith(".zip"):
                    frames.extend(self._parse_nested_zip(data, ticker, year, info.filename))
                    continue

                frames.append(self._process_csv_bytes(data, ticker, year, info.filename))

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _parse_nested_zip(self, payload: bytes, ticker: str, year: int, filename: str) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        with zipfile.ZipFile(io.BytesIO(payload)) as nested:
            for info in nested.infolist():
                if info.is_dir():
                    continue
                data = nested.read(info)
                frames.append(self._process_csv_bytes(data, ticker, year, info.filename))
        return frames

    def _process_csv_bytes(self, payload: bytes, ticker: str, year: int, source_name: str) -> pd.DataFrame:
        if not payload:
            return pd.DataFrame()

        # Сохранить распакованный файл для отладки
        safe_name = Path(source_name).name or f"{year}.csv"
        extract_dir = self.extracted_dir / ticker / str(year)
        extract_dir.mkdir(parents=True, exist_ok=True)
        extract_path = extract_dir / safe_name
        if not extract_path.exists():
            extract_path.write_bytes(payload)

        # Читаем с запасом по столбцам (8 из-за возможного лишнего пустого столбца)
        # и сначала все как строки для надежности
        buffer = io.StringIO(payload.decode("utf-8"))
        df = pd.read_csv(
            buffer,
            sep=";",
            header=None,
            names=list(range(8)),
            usecols=list(range(7)),  # берём первые 7 полей
            dtype=str,
            engine="python",
        )

        if df.empty:
            return pd.DataFrame()

        # Присваиваем понятные имена
        df.columns = ["figi", "timestamp", "open", "close", "high", "low", "volume"]

        # Приведение типов
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        # Переставляем в правильный порядок OHLC: open, high, low, close
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Volume может быть дробным, преобразуем в Int64 (с поддержкой NaN)
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["volume"] = df["volume"].astype("Int64")

        # Удаляем строки с пропущенными значениями
        df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).reset_index(drop=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Добавляем ticker и переименовываем timestamp
        df["ticker"] = ticker
        df = df.rename(columns={"timestamp": "timestamp"})

        # Возвращаем в нужном порядке (без figi)
        df = df[["timestamp", "ticker", "open", "high", "low", "close", "volume"]]
        return df
