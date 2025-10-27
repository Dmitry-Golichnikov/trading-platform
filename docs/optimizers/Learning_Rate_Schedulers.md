# Learning Rate Schedulers

## Cosine Annealing
- Плавно уменьшает learning rate по косинусной кривой.
- Параметры: `T_max`, `eta_min`.
- Часто используется с warmup и restart (SGDR).

## Warmup + Decay
- Линейный или экспоненциальный рост learning rate в начале, затем — выбранный scheduler.
- Обязателен для трансформеров и больших batch.

## Cyclical Learning Rate (CLR)
- Меняет learning rate между нижней и верхней границей (triangular, exp, cosine).
- Помогает выходить из локальных минимумов.

## ReduceLROnPlateau
- Автоматически уменьшает learning rate при отсутствии улучшений метрики.
- Параметры: `factor`, `patience`, `threshold`.

## OneCycle / Super-Convergence
- Увеличение LR до максимума и затем снижение до минимума за одну эпоху.
- Хорошо работает с SGD + momentum.

## Hyperband / BOHB (Auto LR Tuning)
- Байесовская/рандомизированная оптимизация гиперпараметров, включая learning rate.
- Используется для автоматизированного поиска конфигураций.
