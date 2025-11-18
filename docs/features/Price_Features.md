# Ценовые признаки

## Назначение
Базируются на OHLC-данных и отражают динамику цены, структуру свечей и цену относительно исторических уровней.

## Категории
- **OHLC агрегаты**: open, high, low, close, hl2, hlc3, ohlc4.
- **Свечные паттерны**: body size, wick size, candle type (hammer, doji, engulfing).
- **Спреды и относительные величины**: high-low, body/spread, wick/body, close-open.
- **Gap/Jump признаки**: разрыв между текущим open и предыдущим close.
- **Волатильность**: rolling std, Parkinson volatility, intraday volatility.
- **Trendlines/Regressions**: slope, R² по окну.

## Микроструктурные признаки
- `MICRO_BODY`, `MICRO_WICK`, `MICRO_SPREAD`.
- Отношение к ATR, ATR-normalized свечи.
- Intraday volatility (std внутри бара).

## Параметры конфигурации
- `window`, `lag`, `normalize`.
- `pattern_library`: набор паттернов и порогов.
- `volatility_period`: окно для rolling std/ATR.

## Рекомендации
- Учитывать timezone и корректность OHLC при resample.
- Нормализовать величины на ATR/EMA для сравнимости между тикерами.
- Проверять связь с будущими таргетами (feature importance, shap).
