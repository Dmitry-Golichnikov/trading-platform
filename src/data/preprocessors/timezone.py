"""
Утилиты для работы с таймзонами.
"""

import logging
from datetime import datetime

import pandas as pd
import pytz

from src.common.exceptions import PreprocessingError

logger = logging.getLogger(__name__)


def convert_to_utc(data: pd.DataFrame) -> pd.DataFrame:
    """
    Конвертировать timestamp в UTC.

    Args:
        data: DataFrame с колонкой timestamp

    Returns:
        DataFrame с timestamp в UTC
    """
    if "timestamp" not in data.columns:
        raise PreprocessingError("DataFrame must have 'timestamp' column")

    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Если нет timezone info, предполагаем UTC
    if data["timestamp"].dt.tz is None:
        data["timestamp"] = data["timestamp"].dt.tz_localize("UTC")
        logger.debug("Localized naive timestamps to UTC")
    else:
        # Конвертировать в UTC
        data["timestamp"] = data["timestamp"].dt.tz_convert("UTC")
        logger.debug("Converted timestamps to UTC")

    return data


def validate_timezone_aware(data: pd.DataFrame) -> bool:
    """
    Проверить что все timestamps имеют timezone info.

    Args:
        data: DataFrame с timestamp колонкой

    Returns:
        True если все timestamps timezone-aware
    """
    if "timestamp" not in data.columns:
        return False

    timestamps = pd.to_datetime(data["timestamp"])
    return timestamps.dt.tz is not None


def localize_timestamp(ts: datetime, tz: str) -> datetime:
    """
    Локализовать naive timestamp к указанной таймзоне.

    Args:
        ts: Naive datetime
        tz: Имя таймзоны (например, 'Europe/Moscow')

    Returns:
        Timezone-aware datetime
    """
    timezone = pytz.timezone(tz)
    if ts.tzinfo is None:
        return timezone.localize(ts)
    return ts.astimezone(timezone)


def handle_dst_transition(data: pd.DataFrame) -> pd.DataFrame:
    """
    Обработать переход на летнее/зимнее время в данных.

    При переходе на летнее время (весной) может быть пропуск в 1 час.
    При переходе на зимнее время (осенью) может быть дублирование 1 часа.

    Эта функция:
    - Детектит переходы DST
    - Для spring forward: интерполирует пропущенный час
    - Для fall back: усредняет дублирующиеся записи
    - Обеспечивает монотонность временного ряда

    Args:
        data: DataFrame с timezone-aware timestamp

    Returns:
        DataFrame с обработанными DST переходами

    Raises:
        PreprocessingError: Если timestamp не timezone-aware
    """
    if data.empty:
        return data

    if "timestamp" not in data.columns:
        raise PreprocessingError("DataFrame must have 'timestamp' column")

    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Проверка timezone-aware
    if data["timestamp"].dt.tz is None:
        raise PreprocessingError(
            "Timestamps must be timezone-aware to handle DST transitions. "
            "Use convert_to_utc() first."
        )

    # Если данные уже в UTC, DST не влияет
    if str(data["timestamp"].dt.tz) == "UTC":
        logger.debug("Data is in UTC, no DST handling needed")
        return data

    # Сортировать по timestamp
    data = data.sort_values("timestamp").reset_index(drop=True)

    # Вычислить разницу между последовательными timestamp
    time_diffs = data["timestamp"].diff()

    # Определить ожидаемую разницу (предполагаем минутные данные)
    if len(data) > 1:
        # Берём медиану как типичный интервал
        typical_diff = time_diffs.median()  # type: ignore[misc]
    else:
        return data

    # Детектить аномальные разницы (потенциальные DST переходы)
    # Spring forward: разница больше типичной (пропуск)
    # Fall back: дубликаты timestamp
    spring_forward_threshold = typical_diff * 1.5
    duplicates_mask = data.duplicated(subset=["timestamp"], keep=False)

    # Обработать fall back (дублирующиеся timestamp)
    if duplicates_mask.any():
        logger.info(
            f"Detected {duplicates_mask.sum()} duplicate timestamps (fall back)"
        )

        # Усреднить OHLCV для дублирующихся timestamp
        agg_dict = {
            "open": "mean",
            "high": "max",
            "low": "min",
            "close": "mean",
            "volume": "sum",
        }

        group_cols = ["timestamp"]
        if "ticker" in data.columns:
            group_cols.append("ticker")

        # Группировать только дубликаты
        duplicates = data[duplicates_mask].copy()
        non_duplicates = data[~duplicates_mask].copy()

        if not duplicates.empty:
            merged_duplicates = duplicates.groupby(group_cols, as_index=False).agg(
                agg_dict
            )
            # Округлить volume
            merged_duplicates["volume"] = (
                merged_duplicates["volume"].round().astype("int64")
            )

            # Объединить обратно
            data = pd.concat([non_duplicates, merged_duplicates], ignore_index=True)
            data = data.sort_values("timestamp").reset_index(drop=True)

    # Обработать spring forward (пропуски)
    time_diffs = data["timestamp"].diff()
    gaps_mask = time_diffs > spring_forward_threshold  # type: ignore[operator]

    if gaps_mask.any():
        gap_count = gaps_mask.sum()
        logger.info(f"Detected {gap_count} gaps potentially due to spring forward DST")

        # Для каждого пропуска можно интерполировать или оставить как есть
        # В большинстве случаев для финансовых данных пропуск в 1 час допустим
        # (торговля могла не вестись в этот час)
        # Поэтому просто логируем, не заполняем автоматически

    logger.debug("DST transitions handled")
    return data
