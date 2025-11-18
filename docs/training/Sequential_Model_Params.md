# Гиперпараметры последовательных моделей

## Назначение
Определяют архитектуру и поведение рекуррентных сетей (LSTM/GRU), TCN, CNN-Seq и трансформеров. Влияют на способность модели улавливать временные зависимости.

## Общие параметры
- `sequence_length`: длина входного окна.
- `stride`: шаг сдвига окна.
- `target_shift`: насколько таргет отстоит от входного окна (horizon).
- `batch_first`: формат батча (`batch, time, features`).

## RNN (LSTM/GRU)
- `hidden_size`: размер скрытого состояния.
- `num_layers`: количество рекуррентных слоёв.
- `bidirectional`: использовать двунаправленные слои.
- `dropout`: dropout между слоями (не на последнем).
- `teacher_forcing_ratio`: доля teacher forcing в Seq2Seq.
- `scheduled_sampling`: постепенное уменьшение teacher forcing.

## Temporal Convolutional Networks (TCN)
- `kernel_size`: размер свёрточного ядра.
- `dilation`: множитель растяжения (2^i).
- `num_levels`: число блоков.
- `channels`: ширина каналов на каждом уровне.

## Transformer / Attention
- `d_model`: размер скрытого представления.
- `num_heads`: количество attention голов.
- `num_layers`: количество энкодер/декодер слоёв.
- `d_ff`: размер feed-forward слоя.
- `dropout`, `attention_dropout`.
- `positional_encoding`: тип (sinusoidal, learned).
- `layer_norm_eps`: стабилизатор layer norm.

## Регуляризация и стабилизация
- `gradient_clipping`: ограничение нормы градиента.
- `label_smoothing`: для soft таргетов.
- `warmup_steps`: warmup LR для трансформеров.
- `mixed_precision`: FP16/AMP для ускорения.

## Рекомендации
- Подбирать длину окна и stride в связке с horizon.
- Для длинных зависимостей использовать attention/Transformer.
- Применять gradient clipping для стабильности.
- Логировать параметры (d_model, num_layers) и итоговые размеры модели.
