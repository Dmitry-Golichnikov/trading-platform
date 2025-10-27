# Регуляризация и Dropout

## Назначение
Предотвращают переобучение, улучшают обобщающую способность моделей, особенно нейросетей и ансамблей.

## Методы регуляризации
- **L1/L2**: штрафы на веса (lasso, ridge). В деревьях — `lambda_l1`, `lambda_l2`; в нейросетях — `weight_decay`.
- **ElasticNet**: комбинация L1 и L2.
- **Early stopping**: остановка обучения при отсутствии улучшений.
- **Label smoothing**: смягчение меток в классификации.
- **DropConnect**: зануление весов (не только активаций).
- **Stochastic depth**: случайное отключение слоёв (ResNet, Transformer).

## Dropout и варианты
- **Dropout**: зануление доли активаций (обычно 0.1–0.5).
- **Variational dropout**: одинаковая маска на всём времени (RNN).
- **Spatial dropout**: зануление целых каналов (CNN).
- **Embedding dropout**: dropout на входных embedding.

## Параметры
- `dropout_rate`: доля зануления.
- `weight_decay`: коэффициент L2 (например, 1e-5).
- `label_smoothing`: `ε` (обычно 0.05–0.2).
- `early_stopping`: `patience`, `metric`, `min_delta`.
- `stochastic_depth_rate`: вероятность отключения слоя.

## Рекомендации
- Использовать комбинации методов (dropout + weight_decay + early stopping).
- Настраивать dropout отдельно для разных слоёв (embedding, attention, FFN).
- Для деревьев балансировать регуляризацию и сложность (num_leaves, depth).
- Следить за тем, чтобы регуляризация не привела к недообучению (мониторить метрики train/val).
