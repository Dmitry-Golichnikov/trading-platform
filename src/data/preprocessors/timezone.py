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
