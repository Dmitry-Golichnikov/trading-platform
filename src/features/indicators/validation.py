"""
Валидация каузальности индикаторов.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.features.indicators.base import Indicator


def validate_causality(
    indicator: Indicator,
    test_data: pd.DataFrame,
    n_future_bars: int = 10,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """
    Проверить что индикатор каузальный (не использует будущие данные).

    Метод: добавляем будущие данные и проверяем что
    прошлые значения индикатора не изменились.

    Args:
        indicator: Индикатор для проверки
        test_data: Тестовые OHLCV данные
        n_future_bars: Количество будущих баров для добавления
        rtol: Относительная толерантность для сравнения
        atol: Абсолютная толерантность для сравнения

    Returns:
        True если индикатор каузальный, False иначе

    Example:
        >>> sma = SMA(window=20)
        >>> data = generate_random_ohlcv(1000)
        >>> assert validate_causality(sma, data)
    """
    # Рассчитать на исходных данных
    result1 = indicator.calculate(test_data.copy())

    # Добавить будущие данные
    extended_data = append_future_bars(test_data.copy(), n=n_future_bars)
    result2 = indicator.calculate(extended_data)

    # Сравнить на overlap period (исключая NaN значения)
    overlap = result2.iloc[: len(result1)]

    # Проверить что все колонки совпадают
    for col in result1.columns:
        if col not in overlap.columns:
            return False

        # Получить значения (только непустые)
        vals1 = result1[col].values
        vals2 = overlap[col].values

        # Проверить что массивы имеют одинаковую длину
        if len(vals1) != len(vals2):
            return False

        # Сравнить с учётом NaN
        is_equal = np.allclose(vals1, vals2, rtol=rtol, atol=atol, equal_nan=True)  # type: ignore
        if not is_equal:
            return False

    return True


def append_future_bars(data: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Добавить будущие бары к данным для проверки каузальности.

    Генерирует случайные OHLCV данные на основе статистики исходных данных.

    Args:
        data: Исходные OHLCV данные
        n: Количество баров для добавления

    Returns:
        DataFrame с добавленными будущими барами
    """
    # Копируем данные
    extended = data.copy()

    # Получаем последнюю цену закрытия
    last_close = data["close"].iloc[-1]
    last_timestamp = data["timestamp"].iloc[-1] if "timestamp" in data.columns else None

    # Вычисляем статистику для генерации реалистичных данных
    returns = data["close"].pct_change().dropna()
    mean_return = returns.mean()
    std_return = returns.std()

    # Генерируем будущие бары
    future_data = []
    current_price = last_close

    for i in range(n):
        # Генерируем случайное изменение цены
        price_change = np.random.normal(mean_return, std_return)
        new_close = current_price * (1 + price_change)

        # Генерируем OHLC относительно close
        volatility = std_return * np.abs(np.random.normal(0, 1))
        high = new_close * (1 + volatility)
        low = new_close * (1 - volatility)
        open_price = current_price  # Open равен предыдущему close

        # Генерируем объём (на основе среднего)
        avg_volume = data["volume"].mean()
        volume = avg_volume * np.random.uniform(0.5, 1.5)

        # Создаём строку данных
        row = {
            "open": open_price,
            "high": max(high, open_price, new_close),
            "low": min(low, open_price, new_close),
            "close": new_close,
            "volume": volume,
        }

        # Добавляем timestamp если есть
        if last_timestamp is not None:
            # Определяем временной шаг (по разнице последних двух баров)
            if len(data) > 1:
                time_step = data["timestamp"].iloc[-1] - data["timestamp"].iloc[-2]
            else:
                time_step = pd.Timedelta(days=1)
            row["timestamp"] = last_timestamp + time_step * (i + 1)

        future_data.append(row)
        current_price = new_close

    # Добавляем будущие данные
    future_df = pd.DataFrame(future_data)
    extended = pd.concat([extended, future_df], ignore_index=True)

    return extended


def generate_random_ohlcv(
    n_bars: int = 1000,
    start_price: float = 100.0,
    volatility: float = 0.02,
    start_date: Optional[pd.Timestamp] = None,
    freq: str = "1D",
) -> pd.DataFrame:
    """
    Генерировать случайные OHLCV данные для тестирования.

    Args:
        n_bars: Количество баров
        start_price: Начальная цена
        volatility: Волатильность (стандартное отклонение дневных изменений)
        start_date: Начальная дата (по умолчанию сегодня - n_bars дней)
        freq: Частота данных (pandas frequency string)

    Returns:
        DataFrame с OHLCV данными
    """
    np.random.seed(42)  # Для воспроизводимости

    # Генерируем временные метки
    if start_date is None:
        start_date = pd.Timestamp.now() - pd.Timedelta(days=n_bars)

    timestamps = pd.date_range(start=start_date, periods=n_bars, freq=freq)

    # Генерируем цены закрытия (случайное блуждание)
    returns = np.random.normal(0, volatility, n_bars)
    close_prices = start_price * np.exp(np.cumsum(returns))

    # Генерируем OHLC
    data = []
    for i, close in enumerate(close_prices):
        # Генерируем внутрибарную волатильность
        intrabar_vol = volatility * np.random.uniform(0.5, 1.5)

        high = close * (1 + abs(np.random.normal(0, intrabar_vol)))
        low = close * (1 - abs(np.random.normal(0, intrabar_vol)))

        if i == 0:
            open_price = start_price
        else:
            open_price = close_prices[i - 1]

        # Корректируем high/low чтобы они включали open и close
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Генерируем объём
        volume = np.random.uniform(100000, 1000000)

        data.append(
            {
                "timestamp": timestamps[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    return pd.DataFrame(data)
