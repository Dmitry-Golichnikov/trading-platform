# Технические индикаторы текущего таймфрейма

## Назначение
Отражают состояние рынка на рабочем таймфрейме. Используются для оценки тренда, импульса, волатильности и объёмных характеристик.

## Категории индикаторов
- **Трендовые**: SMA/EMA/WMA, Ichimoku, Parabolic SAR.
- **Осцилляторы**: RSI, Stochastic, Williams %R, CCI.
- **Импульс/ROC**: Momentum, Rate of Change, TRIX, Elder Force Index.
- **Каналы**: Bollinger Bands, Keltner Channels, Donchian, Nadaraya-Watson Envelope.
- **Объёмные**: OBV, Chaikin Money Flow, Volume Profile, Volume metrics.
- **Волатильность**: ATR, Fractal Dimension Index, Stochastic RSI.

## Параметры конфигурации
- `window`: длина скользящего окна.
- `source`: цена (close, typical price, hl2, hlc3).
- `smooth`: параметры сглаживания (EMA vs SMA).
- `normalize`: нормализация (z-score, min/max, scaling.
- `shift`: лаг для моделей, исключающих look-ahead.

## Рекомендации
- Хранить описание индикаторов в конфиге (название, параметры, идентификатор).
- Использовать каузальные реализации (не заглядывать в будущее).
- При большом наборе признаков применять отбор/регуляризацию или проверять мультиколлинеарность.
