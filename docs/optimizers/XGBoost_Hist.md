# XGBoost Histogram Optimizer

## Назначение
Варианты `hist` и `gpu_hist` ускоряют обучение XGBoost за счёт гистограммного представления признаков и блокового построения деревьев. Подходит для больших табличных датасетов.

## Ключевые параметры
- `tree_method`: hist, gpu_hist (для GPU), exact, approx.
- `max_depth`, `min_child_weight`, `gamma` — контроль структуры деревьев.
- `subsample`, `colsample_bytree`, `colsample_bylevel` — стохастизация.
- `eta` (learning_rate), `lambda`, `alpha` — шаг и регуляризация.
- `grow_policy`: depthwise или lossguide.

## Особенности
- `hist` уменьшает потребление памяти и ускоряет обучение.
- `gpu_hist` использует параллелизм GPU, обеспечивает значительное ускорение.
- Lossguide growth строит деревья, начиная с лучшего листа (аналог LightGBM leaf-wise).

## Рекомендации
- При дисбалансе классов использовать `scale_pos_weight`.
- Контролировать `max_bin` для скорости/точности.
- Сохранять модели в бинарном формате (`.json`/`.model`) и фиксировать seed для воспроизводимости.
