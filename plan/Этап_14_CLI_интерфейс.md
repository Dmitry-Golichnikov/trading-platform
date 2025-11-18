# Этап 14: CLI интерфейс

## Цель
Полнофункциональный CLI для всех операций платформы.

## Зависимости
Этапы 00-13

## Структура

```
src/interfaces/cli/
├── __main__.py                   # Entry point
├── data_commands.py
├── feature_commands.py
├── model_commands.py
├── backtest_commands.py
├── experiment_commands.py
└── utils.py                      # CLI utilities
```

## Группы команд

### 1. Data
```bash
trading-cli data load --ticker SBER --from 2020-01-01 --to 2023-12-31
trading-cli data list
trading-cli data info --dataset SBER/5m
trading-cli data resample --input SBER/1m --output SBER/5m
trading-cli data quality-report --dataset SBER/5m
```

### 2. Features
```bash
trading-cli features generate --config configs/features/default.yaml
trading-cli features list --dataset SBER/5m
trading-cli features select --method tree --top-k 50
```

### 3. Labels
```bash
trading-cli labels generate --config configs/labeling/long_only.yaml
trading-cli labels analyze --dataset SBER/5m_labeled
```

### 4. Models
```bash
trading-cli model train --config configs/models/lightgbm.yaml
trading-cli model list
trading-cli model evaluate --model-id abc123
trading-cli model predict --model-id abc123 --data test.parquet
```

### 5. Hyperopt
```bash
trading-cli hyperopt run --config configs/hyperopt/search.yaml
trading-cli hyperopt status --run-id xyz789
trading-cli hyperopt best --metric roc_auc
```

### 6. Backtest
```bash
trading-cli backtest run --model-id abc123 --config configs/backtest/default.yaml
trading-cli backtest report --backtest-id def456
trading-cli backtest compare --ids def456,ghi789
```

### 7. Experiments
```bash
trading-cli experiment create --config configs/experiments/full.yaml
trading-cli experiment run --id exp001
trading-cli experiment list --filter "metric:roc_auc>0.8"
trading-cli experiment compare --ids exp001,exp002
trading-cli experiment best --metric sharpe_ratio
```

### 8. Pipeline
```bash
trading-cli pipeline run --config configs/pipelines/full.yaml
trading-cli pipeline resume --id pipe001 --from-step training
trading-cli pipeline status --id pipe001
```

## Utilities

- Progress bars (tqdm)
- Colored output (rich/colorama)
- Tables (tabulate)
- Interactive prompts (click/inquirer)
- Logging to console

## Критерии готовности

- [ ] Все команды реализованы
- [ ] Help messages для всех команд
- [ ] Autocomplete (bash/zsh)
- [ ] Config файл для CLI (~/.trading-cli/config)
- [ ] Красивый вывод (таблицы, прогресс-бары)
- [ ] Error handling с понятными сообщениями

## Промпты

```
Реализуй полный CLI в src/interfaces/cli/:

1. Используй click для CLI framework
2. Организуй команды по группам
3. Каждая группа - отдельный файл
4. __main__.py - entry point с группами

Все команды должны:
- Иметь подробный help
- Валидировать аргументы
- Показывать progress bars для долгих операций
- Красивый вывод результатов (таблицы, цвета)
- Обработка ошибок с понятными сообщениями

Добавь autocomplete генератор для bash/zsh.
```

## Следующий этап
[Этап 15: GUI интерфейс](Этап_15_GUI_интерфейс.md)
