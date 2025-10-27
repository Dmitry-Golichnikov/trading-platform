# Nadam (Nesterov Adam)

## Назначение
Сочетает Adam и Nesterov momentum, что обеспечивает более плавную траекторию обновлений и ускоренную сходимость в некоторых задачах.

## Особенности
- Градиент оценивается с учётом Nesterov-поправки, добавляя предвидение в шаг Adam.
- Поддерживает параметры `learning_rate`, `betas`, `eps`, `weight_decay`.
- Реализован в `torch.optim.NAdam` (PyTorch) и `tf.keras.optimizers.Nadam`.

## Рекомендации
- Использовать, если SGD+Nesterov показал хорошие результаты, но требуется адаптивность Adam.
- Тщательно подбирать learning rate; можно применять warmup и cosine decay.
