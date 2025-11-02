"""
Загрузчик данных из локальных файлов (CSV, Parquet).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.common.exceptions import DataLoadError, DataValidationError

logger = logging.getLogger(__name__)


class LocalFileLoader:
    """
    Загрузчик данных из локальных файлов.

    Поддерживает форматы:
    - Parquet
    - CSV (включая формат из Tinkoff getHistory)

    Автоматически определяет формат файла по расширению.
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Инициализировать загрузчик.

        Args:
            base_path: Базовый путь для поиска файлов (опционально)
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        logger.info(f"LocalFileLoader initialized with base_path: {self.base_path}")

    def load(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime,
        timeframe: str = "1m",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Загрузить данные из локального файла.

        Args:
            ticker: Тикер инструмента
            from_date: Начальная дата
            to_date: Конечная дата
            timeframe: Временной интервал
            file_path: Путь к файлу (если None, пытается найти автоматически)
            **kwargs: Дополнительные параметры

        Returns:
            DataFrame с OHLCV данными

        Raises:
            DataLoadError: При ошибке загрузки
        """
        file_path_arg = kwargs.pop("file_path", None)
        file_path: Optional[Path]
        if isinstance(file_path_arg, (str, Path)):
            file_path = Path(file_path_arg)
        else:
            file_path = None

        if file_path is None:
            file_path = self._find_file(ticker, timeframe)
        else:
            file_path = (
                self.base_path / file_path if not file_path.is_absolute() else file_path
            )

        if not file_path.exists():
            raise DataLoadError(f"File not found: {file_path}")

        logger.info(f"Loading data from {file_path}")

        # Определить формат по расширению
        suffix = file_path.suffix.lower()

        try:
            if suffix == ".parquet":
                df = self._load_parquet(file_path)
            elif suffix == ".csv":
                df = self._load_csv(file_path)
            else:
                raise DataLoadError(f"Unsupported file format: {suffix}")

            # Фильтровать по датам
            df = self._filter_by_dates(df, from_date, to_date)

            # Валидировать схему
            self._validate_schema(df)

            # Добавить ticker если отсутствует
            if "ticker" not in df.columns:
                df["ticker"] = ticker

            logger.info(f"Loaded {len(df)} bars for {ticker}")
            return df

        except Exception as e:
            raise DataLoadError(f"Failed to load data from {file_path}: {e}") from e

    def _load_parquet(self, file_path: Path) -> pd.DataFrame:
        """Загрузить данные из Parquet файла."""
        try:
            df = pd.read_parquet(file_path)

            # Убедиться что timestamp - это DatetimeIndex
            if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp").sort_index()
            elif isinstance(df.index, pd.DatetimeIndex):
                if df.index.tzinfo is None:
                    df.index = df.index.tz_localize("UTC")

            # Сбросить индекс чтобы timestamp была колонкой
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                if "index" in df.columns:
                    df = df.rename(columns={"index": "timestamp"})

            return df

        except Exception as e:
            raise DataLoadError(f"Failed to read Parquet file: {e}") from e

    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Загрузить данные из CSV файла.

        Поддерживает два формата:
        1. Стандартный: timestamp,ticker,open,high,low,close,volume
        2. Tinkoff getHistory: figi;timestamp;open;high;low;close;volume;
        """
        try:
            # Попробовать определить формат по первой строке
            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline()

            # Tinkoff getHistory формат (разделитель ;)
            if ";" in first_line and first_line.count(";") >= 6:
                df = self._load_tinkoff_csv(file_path)
            else:
                # Стандартный CSV
                df = pd.read_csv(
                    file_path, parse_dates=["timestamp"], date_format="ISO8601"
                )

            # Конвертировать timestamp в UTC
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            return df

        except Exception as e:
            raise DataLoadError(f"Failed to read CSV file: {e}") from e

    def _load_tinkoff_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Загрузить CSV в формате Tinkoff getHistory.

        Формат: figi;timestamp;open;high;low;close;volume;
        Пример:
            cbdf1d32-5758-490e-a2b1-780eaa79bdf7;2020-01-03T07:04:00Z;13.37;13.37;
            13.37;13.37;2;
        """
        try:
            df = pd.read_csv(
                file_path,
                sep=";",
                names=[
                    "figi",
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "extra",
                ],
                parse_dates=["timestamp"],
                date_format="ISO8601",
            )

            # Удалить пустую колонку от завершающего разделителя
            if "extra" in df.columns:
                df = df.drop(columns=["extra"])

            # Удалить пустые строки
            df = df.dropna(subset=["figi", "timestamp"])

            # Переименовать figi в ticker
            df = df.rename(columns={"figi": "ticker"})

            # Конвертировать типы
            df["open"] = pd.to_numeric(df["open"], errors="coerce")
            df["high"] = pd.to_numeric(df["high"], errors="coerce")
            df["low"] = pd.to_numeric(df["low"], errors="coerce")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

            # Удалить строки с NaN
            df = df.dropna(subset=["open", "high", "low", "close", "volume"])

            logger.debug(f"Loaded {len(df)} bars from Tinkoff CSV format")
            return df

        except Exception as e:
            raise DataLoadError(f"Failed to parse Tinkoff CSV: {e}") from e

    def _filter_by_dates(
        self, df: pd.DataFrame, from_date: datetime, to_date: datetime
    ) -> pd.DataFrame:
        """Фильтровать DataFrame по диапазону дат."""
        if "timestamp" not in df.columns:
            raise DataValidationError("DataFrame must have 'timestamp' column")

        # Убедиться что timestamp timezone-aware
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Конвертировать from_date и to_date в UTC если нет timezone
        if from_date.tzinfo is None:
            from_date = from_date.replace(tzinfo=pd.Timestamp("now", tz="UTC").tzinfo)
        if to_date.tzinfo is None:
            to_date = to_date.replace(tzinfo=pd.Timestamp("now", tz="UTC").tzinfo)

        mask = (df["timestamp"] >= from_date) & (df["timestamp"] <= to_date)
        return df[mask].copy()

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """
        Валидировать схему DataFrame.

        Raises:
            DataValidationError: При невалидной схеме
        """
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = set(required_columns) - set(df.columns)

        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")

        # Проверить типы
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            raise DataValidationError("'timestamp' must be datetime type")

        for col in ["open", "high", "low", "close"]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise DataValidationError(f"'{col}' must be numeric type")

        if not pd.api.types.is_integer_dtype(
            df["volume"]
        ) and not pd.api.types.is_numeric_dtype(df["volume"]):
            raise DataValidationError("'volume' must be numeric type")

    def _find_file(self, ticker: str, timeframe: str) -> Path:
        """
        Автоматически найти файл для тикера и таймфрейма.

        Ищет в:
        1. artifacts/data/{ticker}/{timeframe}/
        2. data/{ticker}/
        3. {base_path}/
        """
        search_paths = [
            self.base_path / "artifacts" / "data" / ticker / timeframe,
            self.base_path / "data" / ticker,
            self.base_path,
        ]

        for search_dir in search_paths:
            if not search_dir.exists():
                continue

            # Искать .parquet и .csv файлы
            for ext in [".parquet", ".csv"]:
                matches = list(search_dir.glob(f"*{ext}"))
                if matches:
                    return matches[0]

        raise DataLoadError(
            f"Could not find data file for ticker={ticker}, timeframe={timeframe}"
        )

    def get_available_tickers(self) -> list[str]:
        """
        Получить список доступных тикеров.

        Returns:
            Список тикеров из artifacts/data/
        """
        data_dir = self.base_path / "artifacts" / "data"
        if not data_dir.exists():
            return []

        tickers = [d.name for d in data_dir.iterdir() if d.is_dir()]
        logger.debug(f"Found {len(tickers)} tickers in {data_dir}")
        return tickers
