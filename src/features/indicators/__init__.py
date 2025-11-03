"""
Библиотека технических индикаторов для построения признаков.

Все индикаторы каузальные (не используют будущие данные).
"""

# Импорт индикаторов для автоматической регистрации
from src.features.indicators import advanced  # noqa: F401
from src.features.indicators import momentum  # noqa: F401
from src.features.indicators import trend  # noqa: F401
from src.features.indicators import volatility  # noqa: F401
from src.features.indicators import volume  # noqa: F401
from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry

__all__ = [
    "Indicator",
    "IndicatorRegistry",
]
