# CatBoost Built-in Optimizer

## Назначение
CatBoost использует градиентный бустинг на симметричных деревьях с ordered boosting и встроенной обработкой категориальных признаков. Оптимизация основана на адаптивном градиентном шаге.

## Ключевые параметры
- `iterations`, `depth`, `learning_rate`.
- `l2_leaf_reg`: регуляризация листьев.
- `border_count`: количество бинов для числовых признаков.
- `bagging_temperature`: стохастический контроль переобучения.
- `loss_function`, `eval_metric`: выбор цели и метрики.
- `sampling_frequency`, `rsm`: частота и размер случайных подмножества признаков.

## Особенности
- Ordered boosting предотвращает таргет leakage при категориальных фичах.
- Поддержка GPU (`task_type="GPU"`).
- Автоматический подбор шага обучения (`learning_rate`) при необходимости.

## Рекомендации
- Указывать вес классов (`class_weights` или `scale_pos_weight`) при дисбалансе long/short.
- Использовать встроенный cross-validation (`cv`) для подбора итераций.
- Логировать важность признаков (`PredictionValuesChange`, `LossFunctionChange`).
