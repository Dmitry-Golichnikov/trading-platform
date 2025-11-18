# LightGBM Built-in Optimizer

## Назначение
LightGBM использует градиентный бустинг по деревьям с оптимизацией по листьям (`leaf-wise`). Для повышения устойчивости и скорости реализован специальный алгоритм построения гистограмм.

## Ключевые параметры
- `boosting_type`: gbdt, dart, rf.
- `tree_learner`: serial, feature, data, voting (распределённое обучение).
- `max_depth`, `num_leaves`: контролируют сложность дерева.
- `learning_rate`: шаг градиентного бустинга.
- `feature_fraction`, `bagging_fraction`, `bagging_freq`: стохастизация.
- `min_data_in_leaf`, `lambda_l1`, `lambda_l2`: регуляризация.

## Особенности
- Поддержка GPU (`device_type=gpu`), оптимизация памяти через гистограммы.
- Использует `leaf-wise` стратегию роста деревьев: более агрессивный поиск лучших сплитов.
- Требует контроль `num_leaves` и `min_data_in_leaf` для предотвращения переобучения.

## Рекомендации по применению
- Для табличных признаков с большим объёмом данных — основной вариант.
- Для задач long-only/short-only можно задавать `scale_pos_weight` или `class_weight`.
- Следите за ранней остановкой (`early_stopping_rounds`) и логированием через callbacks.
