"""
Демонстрация работы библиотеки технических индикаторов.
"""

import sys
from pathlib import Path

# Добавляем путь к src ПЕРЕД импортами
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

from src.features.indicators.registry import IndicatorRegistry  # noqa: E402
from src.features.indicators.validation import generate_random_ohlcv  # noqa: E402


def main():
    """Демонстрация работы индикаторов."""
    print("=" * 80)
    print("Демонстрация библиотеки технических индикаторов")
    print("=" * 80)

    # Генерируем тестовые данные
    print("\n1. Генерация тестовых OHLCV данных...")
    data = generate_random_ohlcv(n_bars=100, start_price=100.0)
    print(f"   Сгенерировано {len(data)} баров")
    print(f"   Диапазон цен: {data['close'].min():.2f} - {data['close'].max():.2f}")

    # Список всех зарегистрированных индикаторов
    print("\n2. Зарегистрированные индикаторы:")
    all_indicators = IndicatorRegistry.list_all()
    print(f"   Всего индикаторов: {len(all_indicators)}")
    for i, name in enumerate(all_indicators, 1):
        print(f"   {i:2d}. {name}")

    # Демонстрация различных индикаторов
    print("\n3. Расчёт индикаторов:")

    # SMA
    print("\n   a) SMA (Simple Moving Average):")
    sma = IndicatorRegistry.get("sma", window=20)
    sma_result = sma.calculate(data)
    print(f"      Последнее значение SMA_20: {sma_result['SMA_20'].iloc[-1]:.2f}")

    # EMA
    print("\n   b) EMA (Exponential Moving Average):")
    ema = IndicatorRegistry.get("ema", window=20)
    ema_result = ema.calculate(data)
    print(f"      Последнее значение EMA_20: {ema_result['EMA_20'].iloc[-1]:.2f}")

    # RSI
    print("\n   c) RSI (Relative Strength Index):")
    rsi = IndicatorRegistry.get("rsi", window=14)
    rsi_result = rsi.calculate(data)
    rsi_value = rsi_result["RSI_14"].iloc[-1]
    print(f"      Последнее значение RSI_14: {rsi_value:.2f}")
    if rsi_value > 70:
        print("      Статус: Перекупленность")
    elif rsi_value < 30:
        print("      Статус: Перепроданность")
    else:
        print("      Статус: Нормальный диапазон")

    # MACD
    print("\n   d) MACD:")
    macd = IndicatorRegistry.get("macd", fast=12, slow=26, signal=9)
    macd_result = macd.calculate(data)
    print(f"      MACD Line: {macd_result['MACD'].iloc[-1]:.4f}")
    print(f"      Signal Line: {macd_result['MACD_signal'].iloc[-1]:.4f}")
    print(f"      Histogram: {macd_result['MACD_hist'].iloc[-1]:.4f}")

    # ATR
    print("\n   e) ATR (Average True Range):")
    atr = IndicatorRegistry.get("atr", window=14)
    atr_result = atr.calculate(data)
    print(f"      Последнее значение ATR_14: {atr_result['ATR_14'].iloc[-1]:.2f}")

    # Bollinger Bands
    print("\n   f) Bollinger Bands:")
    bb = IndicatorRegistry.get("bollinger_bands", window=20, std_dev=2.0)
    bb_result = bb.calculate(data)
    current_price = data["close"].iloc[-1]
    print(f"      Верхняя полоса: {bb_result['BB_upper'].iloc[-1]:.2f}")
    print(f"      Средняя линия: {bb_result['BB_middle'].iloc[-1]:.2f}")
    print(f"      Нижняя полоса: {bb_result['BB_lower'].iloc[-1]:.2f}")
    print(f"      Текущая цена: {current_price:.2f}")
    print(f"      %B: {bb_result['BB_pct'].iloc[-1]:.2f}")

    # ADX
    print("\n   g) ADX (Average Directional Index):")
    adx = IndicatorRegistry.get("adx", window=14)
    adx_result = adx.calculate(data)
    adx_value = adx_result["ADX_14"].iloc[-1]
    print(f"      ADX: {adx_value:.2f}")
    print(f"      +DI: {adx_result['DI_plus'].iloc[-1]:.2f}")
    print(f"      -DI: {adx_result['DI_minus'].iloc[-1]:.2f}")
    if adx_value > 25:
        print("      Статус: Сильный тренд")
    else:
        print("      Статус: Слабый тренд / Флэт")

    # Объединение нескольких индикаторов
    print("\n4. Объединение индикаторов в один DataFrame:")
    combined = pd.concat(
        [
            data[["timestamp", "close"]],
            sma_result,
            ema_result,
            rsi_result,
            atr_result,
        ],
        axis=1,
    )
    print("\n   Последние 5 строк:")
    print(combined.tail().to_string())

    print("\n" + "=" * 80)
    print("Демонстрация завершена!")
    print("=" * 80)


if __name__ == "__main__":
    main()
