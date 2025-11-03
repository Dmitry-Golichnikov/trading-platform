# Библиотека технических индикаторов

Полная библиотека каузальных технических индикаторов для построения признаков торговых моделей.

## Особенности

- ✅ **30+ индикаторов** всех категорий (трендовые, моментум, волатильность, объёмные, продвинутые)
- ✅ **Все индикаторы каузальные** - не используют будущие данные
- ✅ **Единый API** - все индикаторы наследуются от базового класса `Indicator`
- ✅ **Реестр индикаторов** - удобный доступ через `IndicatorRegistry`
- ✅ **Валидация каузальности** - автоматическая проверка корректности реализации
- ✅ **Полностью протестировано** - unit тесты для всех индикаторов

## Установка

```bash
# Установите зависимости
pip install -r requirements.txt
```

## Быстрый старт

### Базовое использование

```python
from src.features.indicators import IndicatorRegistry
from src.features.indicators.validation import generate_random_ohlcv

# Генерируем тестовые данные
data = generate_random_ohlcv(n_bars=100, start_price=100.0)

# Создаём индикатор через реестр
sma = IndicatorRegistry.get("sma", window=20)

# Рассчитываем
result = sma.calculate(data)
print(result["SMA_20"])
```

### Прямое создание индикатора

```python
from src.features.indicators.trend.sma import SMA

sma = SMA(window=20)
result = sma.calculate(data)
```

### Список всех индикаторов

```python
from src.features.indicators import IndicatorRegistry

all_indicators = IndicatorRegistry.list_all()
print(f"Доступно индикаторов: {len(all_indicators)}")
print(all_indicators)
```

## Категории индикаторов

### Трендовые (Trend)

- **SMA** - Simple Moving Average
- **EMA** - Exponential Moving Average
- **WMA** - Weighted Moving Average
- **MACD** - Moving Average Convergence Divergence
- **Parabolic SAR** - Stop and Reverse
- **Ichimoku** - Ichimoku Cloud (5 линий)

```python
# Пример: MACD
macd = IndicatorRegistry.get("macd", fast=12, slow=26, signal=9)
result = macd.calculate(data)
# result содержит: MACD, MACD_signal, MACD_hist
```

### Моментум (Momentum)

- **RSI** - Relative Strength Index
- **Stochastic** - Stochastic Oscillator
- **Stochastic RSI** - Stochastic RSI
- **CCI** - Commodity Channel Index
- **Williams %R**

```python
# Пример: RSI
rsi = IndicatorRegistry.get("rsi", window=14)
result = rsi.calculate(data)
# result содержит: RSI_14 (значения 0-100)
```

### Волатильность (Volatility)

- **ATR** - Average True Range
- **Bollinger Bands** - Полосы Боллинджера
- **Keltner Channels** - Каналы Кельтнера
- **Donchian Channels** - Каналы Дончиана

```python
# Пример: Bollinger Bands
bb = IndicatorRegistry.get("bollinger_bands", window=20, std_dev=2.0)
result = bb.calculate(data)
# result содержит: BB_upper, BB_middle, BB_lower, BB_width, BB_pct
```

### Объёмные (Volume)

- **OBV** - On Balance Volume
- **VWAP** - Volume Weighted Average Price
- **MFI** - Money Flow Index
- **Chaikin Money Flow**
- **Accumulation/Distribution**
- **Volume Profile**

```python
# Пример: OBV
obv = IndicatorRegistry.get("obv")
result = obv.calculate(data)
# result содержит: OBV
```

### Продвинутые (Advanced)

- **ADX** - Average Directional Index
- **Elder Force Index**
- **TRIX** - Triple Exponential Average
- **DPO** - Detrended Price Oscillator
- **FDI** - Fractal Dimension Index
- **Nadaraya-Watson Envelope**
- **Heikin-Ashi** - Японские свечи
- **Pivot Points** - Classic/Fibonacci/Camarilla

```python
# Пример: ADX
adx = IndicatorRegistry.get("adx", window=14)
result = adx.calculate(data)
# result содержит: ADX_14, DI_plus, DI_minus
```

## Валидация каузальности

Все индикаторы проверяются на каузальность автоматически:

```python
from src.features.indicators.validation import validate_causality
from src.features.indicators import IndicatorRegistry

sma = IndicatorRegistry.get("sma", window=20)
data = generate_random_ohlcv(n_bars=200)

is_causal = validate_causality(sma, data)
print(f"Индикатор каузальный: {is_causal}")  # True
```

## Создание собственного индикатора

```python
from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry
import pandas as pd

@IndicatorRegistry.register("my_indicator")
class MyIndicator(Indicator):
    """Мой кастомный индикатор."""

    def __init__(self, window: int = 10):
        super().__init__(window=window)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Рассчитать индикатор."""
        self._validate_data(data)

        window = self.params["window"]
        result = data[["close"]].rolling(window=window).mean()

        return result

    def get_required_columns(self):
        return ["close"]

    def get_lookback_period(self):
        return self.params["window"]

# Использование
my_ind = IndicatorRegistry.get("my_indicator", window=15)
result = my_ind.calculate(data)
```

## Запуск демонстрации

```bash
python examples/indicators_demo.py
```

Это покажет работу всех основных индикаторов на реальных данных.

## Запуск тестов

```bash
# Все тесты
pytest tests/unit/features/indicators/

# Конкретный индикатор
pytest tests/unit/features/indicators/test_sma.py

# Тесты каузальности
pytest tests/unit/features/indicators/test_causality.py

# С выводом
pytest tests/unit/features/indicators/ -v
```

## Требования к данным

Все индикаторы ожидают DataFrame со следующими колонками:

- `timestamp` - временная метка (pandas Timestamp)
- `open` - цена открытия
- `high` - максимальная цена
- `low` - минимальная цена
- `close` - цена закрытия
- `volume` - объём торгов

Некоторые индикаторы используют только часть колонок (например, SMA использует только `close`).

## API индикатора

Все индикаторы реализуют следующий интерфейс:

```python
class Indicator(ABC):
    def __init__(self, **params):
        """Инициализация с параметрами."""

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Рассчитать индикатор. Возвращает DataFrame с результатами."""

    def get_required_columns(self) -> List[str]:
        """Список необходимых колонок."""

    def get_lookback_period(self) -> int:
        """Период разогрева (количество баров с NaN)."""

    def validate_params(self) -> None:
        """Валидация параметров."""

    @property
    def name(self) -> str:
        """Имя индикатора."""
```

## Производительность

Все индикаторы оптимизированы для производительности:

- Векторизация с numpy/pandas
- Избегание циклов где возможно
- Эффективное использование памяти
- Расчёт на 10K барах < 1 сек для каждого индикатора

## Дальнейшие шаги

После реализации библиотеки индикаторов:

1. **Этап 04**: Генерация признаков - использование индикаторов для создания фич
2. **Этап 05**: Разметка таргетов
3. **Этап 06-10**: Моделирование

## Документация

Каждый индикатор содержит подробную документацию:

```python
from src.features.indicators import IndicatorRegistry

sma = IndicatorRegistry.get("sma", window=20)
help(sma)  # Показать документацию
```

## Лицензия

MIT

---

**Статус**: ✅ Реализован полностью

**Дата**: 03.11.2025

**Версия**: 1.0
