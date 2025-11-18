# Hyperparameter Optimization Module

Модуль для оптимизации гиперпараметров моделей машинного обучения.

## Компоненты

### 1. Базовые Оптимизаторы

#### Grid Search
Полный перебор всех комбинаций параметров.

```python
from src.modeling.hyperopt import GridSearchOptimizer

optimizer = GridSearchOptimizer(
    objective_func=my_objective,
    search_space={
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "n_points": 10},
        "max_depth": {"type": "int", "low": 3, "high": 10, "step": 1},
    },
    metric_name="roc_auc",
    direction="maximize",
)

result = optimizer.optimize()
```

#### Random Search
Случайная выборка из пространства параметров.

```python
from src.modeling.hyperopt import RandomSearchOptimizer

optimizer = RandomSearchOptimizer(
    objective_func=my_objective,
    search_space=search_space,
    n_trials=100,
    timeout=3600,
)

result = optimizer.optimize()
```

#### Bayesian Optimization
Оптимизация с использованием Гауссовских процессов.

```python
from src.modeling.hyperopt import BayesianOptimizer

optimizer = BayesianOptimizer(
    objective_func=my_objective,
    search_space=search_space,
    n_trials=100,
    n_initial_points=10,
    acq_func="EI",  # Expected Improvement
)

result = optimizer.optimize()
```

#### Optuna Backend
Продвинутая оптимизация с использованием Optuna.

```python
from src.modeling.hyperopt import OptunaOptimizer

optimizer = OptunaOptimizer(
    objective_func=my_objective,
    search_space=search_space,
    n_trials=100,
    pruning_enabled=True,
    sampler="TPE",  # Tree-structured Parzen Estimator
)

result = optimizer.optimize()
```

### 2. Multi-Level Optimization

Иерархическая оптимизация с "заморозкой" предыдущих уровней.

```python
from src.modeling.hyperopt import MultiLevelOptimizer, OptimizationLevel

levels = [
    OptimizationLevel(
        name="data_prep",
        search_space=data_space,
        optimizer_type="random",
        optimizer_config={"n_trials": 20},
    ),
    OptimizationLevel(
        name="model",
        search_space=model_space,
        optimizer_type="bayesian",
        optimizer_config={"n_trials": 50},
    ),
]

optimizer = MultiLevelOptimizer(
    objective_func=my_objective,
    levels=levels,
    metric_name="roc_auc",
    direction="maximize",
)

best_config = optimizer.optimize_hierarchical()
```

Или из конфигурационного файла:

```python
optimizer = MultiLevelOptimizer.from_config(
    config_path="configs/hyperopt/multi_level_example.yaml",
    objective_func=my_objective,
)

best_config = optimizer.optimize_hierarchical()
```

### 3. Threshold Optimization

Оптимизация порога классификации по ожидаемому PnL.

```python
from src.modeling.hyperopt import ThresholdOptimizer

optimizer = ThresholdOptimizer(
    tp=0.02,  # Take profit 2%
    sl=0.01,  # Stop loss 1%
    commission=0.001,  # Commission 0.1%
    constraints={
        "min_trades": 50,
        "max_drawdown": 0.15,
        "min_sharpe": 0.5,
    },
)

result = optimizer.optimize_threshold(y_proba, y_true)

print(f"Optimal threshold: {result.optimal_threshold}")
print(f"Expected PnL: {result.expected_pnl}")
print(f"Win rate: {result.win_rate}")

# Визуализация
optimizer.plot_threshold_curve(result, save_path="threshold_curve.png")
```

### 4. AutoML Pipeline

Автоматический поиск лучшей модели и гиперпараметров.

```python
from src.modeling.hyperopt import AutoMLPipeline

automl = AutoMLPipeline(
    model_types=["lightgbm", "xgboost", "catboost"],
    time_budget=3600,  # 1 hour
    n_trials_per_model=50,
    feature_selection=True,
)

automl.fit(X_train, y_train, X_val, y_val)

# Получить leaderboard
leaderboard = automl.get_leaderboard()
print(leaderboard)

# Предсказания
predictions = automl.predict(X_test)
```

### 5. Meta-Learning

Warm-start оптимизации на основе предыдущих экспериментов.

```python
from src.modeling.hyperopt import MetaLearning

meta = MetaLearning(history_db_path="artifacts/meta_learning/history.json")

# Предложить стартовую точку
suggested_params = meta.suggest_starting_point(
    X=X_train,
    y=y_train,
    model_type="lightgbm",
)

# Добавить новый эксперимент
meta.add_experiment(
    model_type="lightgbm",
    best_params={"num_leaves": 31, "learning_rate": 0.1},
    best_score=0.85,
    X=X_train,
    y=y_train,
)

# Статистика
stats = meta.get_statistics()
print(stats)
```

## Experiment Tracking

Централизованное отслеживание экспериментов.

```python
from src.orchestration.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker(
    experiment_name="my_experiment",
    use_mlflow=True,
)

# Логирование эксперимента
run_id = tracker.log_experiment(
    run_name="baseline",
    config={"learning_rate": 0.1, "max_depth": 5},
    metrics={"roc_auc": 0.85, "accuracy": 0.82},
    tags={"version": "v1"},
)

# Сравнение экспериментов
comparison_df = tracker.compare_experiments(
    run_ids=["run_1", "run_2", "run_3"],
    metrics=["roc_auc", "accuracy"],
)

# Лучший эксперимент
best_exp = tracker.get_best_experiment(
    metric="roc_auc",
    direction="maximize",
)
```

## CLI Commands

### Hyperparameter Search

```bash
# Run hyperparameter optimization
python -m src.interfaces.cli hyperopt run \
    --config configs/hyperopt/lightgbm_search_space.yaml \
    --data-path artifacts/data/train.parquet \
    --target-col target \
    --output-dir artifacts/hyperopt/lightgbm

# Run AutoML
python -m src.interfaces.cli hyperopt automl \
    --data-path artifacts/data/train.parquet \
    --target-col target \
    --time-budget 3600 \
    --output-dir artifacts/automl

# Compare experiments
python -m src.interfaces.cli hyperopt compare \
    --run-ids "run1,run2,run3" \
    --metrics "roc_auc,accuracy" \
    --output comparison.csv

# Get best experiment
python -m src.interfaces.cli hyperopt best \
    --metric roc_auc \
    --direction maximize

# List experiments
python -m src.interfaces.cli hyperopt list --limit 10

# Export experiments
python -m src.interfaces.cli hyperopt export --output experiments.csv
```

## Конфигурационные файлы

### Search Space

Пример конфигурации для LightGBM:

```yaml
# configs/hyperopt/lightgbm_search_space.yaml

model_type: lightgbm
metric_name: roc_auc
direction: maximize

optimizer_type: bayesian
optimizer_config:
  n_trials: 100
  timeout: 3600

search_space:
  num_leaves:
    type: int
    low: 20
    high: 150

  learning_rate:
    type: float
    low: 0.001
    high: 0.3
    log: true

  feature_fraction:
    type: float
    low: 0.5
    high: 1.0

fixed_params:
  n_estimators: 1000
  random_state: 42
```

### Multi-Level Optimization

```yaml
# configs/hyperopt/multi_level_example.yaml

metric_name: roc_auc
direction: maximize

levels:
  - name: data_prep
    optimizer_type: random
    optimizer_config:
      n_trials: 20
    search_space:
      outlier_threshold:
        type: float
        low: 2.0
        high: 5.0

  - name: model
    optimizer_type: bayesian
    optimizer_config:
      n_trials: 50
    search_space:
      learning_rate:
        type: float
        low: 0.001
        high: 0.3
        log: true
```

## Примеры использования

### Полный цикл оптимизации

```python
import pandas as pd
from src.modeling.hyperopt import OptunaOptimizer
from src.modeling.registry import ModelRegistry
from src.orchestration.experiment_tracker import ExperimentTracker

# Загрузка данных
df = pd.read_parquet("artifacts/data/train.parquet")
X_train, y_train = df.drop(columns=["target"]), df["target"]

# Создание objective function
def objective(params):
    model_cls = ModelRegistry.get_model("lightgbm")
    model = model_cls(**params)
    model.fit(X_train, y_train)

    # Оценка на валидации
    from sklearn.metrics import roc_auc_score
    y_pred = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_pred)

# Оптимизация
optimizer = OptunaOptimizer(
    objective_func=objective,
    search_space=search_space,
    n_trials=100,
)

result = optimizer.optimize()

# Логирование
tracker = ExperimentTracker(experiment_name="hyperopt")
tracker.log_experiment(
    run_name="lightgbm_optuna",
    config=result.best_params,
    metrics={"roc_auc": result.best_value},
)
```

## Лучшие практики

1. **Выбор оптимизатора:**
   - Grid Search: для небольших пространств поиска
   - Random Search: для быстрого baseline
   - Bayesian/Optuna: для эффективной оптимизации

2. **Search Space:**
   - Используйте логарифмическую шкалу для learning rate
   - Ограничивайте диапазоны разумными значениями
   - Используйте categorical для дискретных выборов

3. **Ресурсы:**
   - Устанавливайте timeout для ограничения времени
   - Включайте pruning для ранней остановки
   - Используйте meta-learning для warm-start

4. **Воспроизводимость:**
   - Фиксируйте random_state
   - Логируйте все конфигурации
   - Версионируйте данные и код

5. **Multi-Level Optimization:**
   - Оптимизируйте данные/признаки отдельно от модели
   - Замораживайте лучшие конфиги на каждом уровне
   - Начинайте с грубого поиска, затем точная настройка

## Зависимости

```bash
pip install optuna scikit-optimize mlflow
```

## См. также

- [Этап 06: Базовая инфраструктура моделирования](../../plan/Этап_06_Базовая_инфраструктура_моделирования.md)
- [Этап 07: Классические ML модели](../../plan/Этап_07_Классические_ML_модели.md)
- [Этап 10: Оценка моделей](../../plan/Этап_10_Оценка_моделей.md)
