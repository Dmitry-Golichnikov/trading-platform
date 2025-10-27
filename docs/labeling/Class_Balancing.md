# Балансировка классов в разметке

## Назначение
Компенсирует дисбаланс таргетов (редкие long/short сигналы) для корректного обучения моделей.

## Методы
- **Веса классов**: `class_weight` (balanced/custom), `pos_weight` для BCE.
- **Oversampling/undersampling**: повторение редких классов или удаление частых (Random, SMOTE, TimeSeries-SMOTE).
- **Sequence-aware weighting**: увеличение веса последовательностей редких сигналов (например, long-long-long подряд).
- **Cost-sensitive подходы**: разные штрафы за FP/FN в loss-функции.
- **Batch-level балансировка**: формирование mini-batch с равным числом примеров каждого класса.

## Параметры
- `method`: `weights`, `oversample`, `undersample`, `sequence_weighting`, `cost_sensitive`.
- `weights`: словарь `class → weight`.
- `oversample_ratio`, `undersample_ratio`.
- `sequence_window`: размер окна для sequence weighting.
- `loss_adjustment`: коэффициенты для loss (например, в Focal/Tversky).

## Рекомендации
- Вести лог итогового распределения классов после балансировки.
- Проверять влияние на метрики (особенно Precision/Recall).
- Для временных рядов избегать «перемешивания» последовательности — oversampling проводить по блокам.
