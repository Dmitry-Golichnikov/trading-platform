# Batching и оптимизация

## Назначение
Настройки batching контролируют размер батча, частоту обновлений и использование ресурсов. Дополнительные параметры оптимизации влияют на градиентные шаги и эффективность обучения.

## Параметры batching
- `batch_size`: количество примеров в батче.
- `dynamic_batching`: изменение размера батча на лету (например, больше на GPU, меньше на CPU).
- `gradient_accumulation_steps`: накапливание градиентов для эффективного увеличения batch size.
- `shuffle`: перемешивание данных (учитывать, что для временных рядов использовать `rolling` shuffle).
- `drop_last`: удаление неполного батча.
- `bucket_by_length`: группировка последовательностей по длине (для RNN/Transformer).

## Mixed precision и вычисления
- `amp`: режим автоматической смешанной точности (AMP, FP16).
- `loss_scaling`: статический/динамический scale для FP16.
- `cpu_offload`: перенос части вычислений на CPU (DeepSpeed, ZeRO).
- `distributed`: параметры DDP (world_size, backend, gradient_sync).

## Доппараметры оптимизации
- `optimizer_params`: betas, eps, weight_decay.
- `clip_grad_norm`/`clip_grad_value`: ограничение градиентов.
- `ema_decay`: EMA/Polyak averaging.
- `sam_rho`: радиус SAM.
- `lookahead_k`, `lookahead_alpha`.

## Рекомендации
- Подбирать batch size с учётом памяти и статистики (batch_norm/LayerNorm).
- Использовать gradient accumulation для имитации большого батча на ограниченной памяти.
- Включать AMP на современных GPU (с соблюдением стабильности).
- Логировать фактический batch size и acc steps для воспроизводимости.
