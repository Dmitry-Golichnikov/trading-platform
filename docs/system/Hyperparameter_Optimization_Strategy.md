# Стратегия оптимизации гиперпараметров и вариативных элементов

## 1. Обзор вариативных элементов

### 1.1 Категории параметров

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List

class ParameterCategory(Enum):
    """Категории оптимизируемых параметров"""
    DATA = "data"                    # Тикеры, таймфреймы, фильтры
    FEATURES = "features"            # Признаки, окна индикаторов
    LABELING = "labeling"            # Horizon, барьеры, режимы
    MODEL = "model"                  # Архитектура, гиперпараметры
    TRAINING = "training"            # LR, batch size, оптимизаторы
    STRATEGY = "strategy"            # Пороги, стопы, тейк-профиты
    RISK = "risk"                    # Размер позиции, лимиты

@dataclass
class SearchSpace:
    """Определение пространства поиска"""
    category: ParameterCategory
    parameters: Dict[str, Any]
    frozen: bool = False  # Заморожен ли блок
    priority: int = 1     # Приоритет оптимизации
```

### 1.2 Пространство поиска

```python
# Пример полного пространства поиска
FULL_SEARCH_SPACE = {
    # Данные
    ParameterCategory.DATA: {
        'tickers': ['SBER', 'GAZP', 'LKOH', 'YNDX'],
        'timeframe': ['5m', '15m', '1h', '4h'],
        'direction': ['long_only', 'short_only', 'long_short'],
        'min_volume': (1000, 1000000, 'log-uniform'),
    },

    # Признаки
    ParameterCategory.FEATURES: {
        'sma_windows': [(5, 200), (10, 100), (20, 50)],
        'rsi_period': (7, 21, 'int'),
        'macd_fast': (8, 16, 'int'),
        'macd_slow': (20, 30, 'int'),
        'use_volume_features': [True, False],
        'use_higher_tf': [True, False],
    },

    # Разметка
    ParameterCategory.LABELING: {
        'target_type': ['horizon', 'triple_barrier'],
        'horizon': (5, 50, 'int'),
        'upper_barrier': (0.01, 0.05, 'float'),
        'lower_barrier': (0.01, 0.05, 'float'),
        'barrier_type': ['fixed', 'atr', 'volatility'],
        'min_return': (0.001, 0.02, 'float'),
    },

    # Модель
    ParameterCategory.MODEL: {
        'model_type': ['lightgbm', 'catboost', 'xgboost', 'lstm'],
        'n_estimators': (50, 500, 'int'),
        'max_depth': (3, 12, 'int'),
        'learning_rate': (0.001, 0.3, 'log-uniform'),
        'min_child_weight': (1, 10, 'int'),
        'subsample': (0.5, 1.0, 'float'),
        'colsample_bytree': (0.5, 1.0, 'float'),
        # LSTM specific
        'hidden_size': (64, 512, 'int'),
        'num_layers': (1, 4, 'int'),
        'dropout': (0.0, 0.5, 'float'),
    },

    # Обучение
    ParameterCategory.TRAINING: {
        'optimizer': ['adam', 'adamw', 'sgd'],
        'batch_size': [32, 64, 128, 256],
        'epochs': (50, 300, 'int'),
        'early_stopping_patience': (10, 50, 'int'),
        'weight_decay': (1e-6, 1e-2, 'log-uniform'),
    },

    # Стратегия
    ParameterCategory.STRATEGY: {
        'entry_threshold': (0.5, 0.9, 'float'),
        'exit_threshold': (0.3, 0.7, 'float'),
        'take_profit': (0.01, 0.1, 'float'),
        'stop_loss': (0.005, 0.05, 'float'),
        'trailing_stop': (0.005, 0.03, 'float'),
        'max_holding_bars': (5, 100, 'int'),
        'use_trailing': [True, False],
    },

    # Риск-менеджмент
    ParameterCategory.RISK: {
        'position_size': (0.1, 1.0, 'float'),
        'max_positions': (1, 5, 'int'),
        'max_daily_loss': (0.02, 0.1, 'float'),
        'max_portfolio_risk': (0.05, 0.2, 'float'),
    }
}
```

## 2. Методы оптимизации

### 2.1 Grid Search

```python
from itertools import product
from typing import Iterator

class GridSearchOptimizer:
    """Полный перебор сетки параметров"""

    def __init__(self, search_space: dict, n_splits: int = 3):
        self.search_space = search_space
        self.n_splits = n_splits

    def generate_configs(self) -> Iterator[dict]:
        """
        Генерировать все комбинации параметров

        Yields:
            Конфигурации для оценки
        """
        # Дискретизировать непрерывные параметры
        discretized = self._discretize_space(self.search_space)

        # Получить все комбинации
        keys = list(discretized.keys())
        values = [discretized[k] for k in keys]

        for combination in product(*values):
            config = dict(zip(keys, combination))
            yield config

    def _discretize_space(self, space: dict, n_points: int = 5) -> dict:
        """Дискретизировать непрерывные параметры"""
        discretized = {}

        for param_name, param_def in space.items():
            if isinstance(param_def, (list, tuple)) and len(param_def) == 3:
                # Непрерывный параметр: (min, max, distribution)
                low, high, dist = param_def

                if dist == 'int':
                    discretized[param_name] = list(range(low, high + 1,
                                                         max(1, (high - low) // n_points)))
                elif dist == 'log-uniform':
                    discretized[param_name] = np.logspace(
                        np.log10(low), np.log10(high), n_points
                    ).tolist()
                else:  # uniform
                    discretized[param_name] = np.linspace(
                        low, high, n_points
                    ).tolist()
            else:
                # Категориальный или дискретный
                discretized[param_name] = param_def

        return discretized

    def estimate_trials(self) -> int:
        """Оценить количество trials"""
        discretized = self._discretize_space(self.search_space)
        n_combinations = 1
        for values in discretized.values():
            n_combinations *= len(values)
        return n_combinations

# Использование
optimizer = GridSearchOptimizer(FULL_SEARCH_SPACE)
n_trials = optimizer.estimate_trials()
print(f"Grid search потребует {n_trials} trials")

# Для полного пространства это будет astronomical number!
# Поэтому grid search подходит только для небольших пространств
```

### 2.2 Random Search

```python
import numpy as np

class RandomSearchOptimizer:
    """Случайная выборка из пространства параметров"""

    def __init__(self, search_space: dict, n_trials: int = 100, seed: int = 42):
        self.search_space = search_space
        self.n_trials = n_trials
        self.rng = np.random.RandomState(seed)

    def generate_configs(self) -> Iterator[dict]:
        """
        Генерировать случайные конфигурации

        Yields:
            n_trials конфигураций
        """
        for _ in range(self.n_trials):
            config = {}
            for param_name, param_def in self.search_space.items():
                config[param_name] = self._sample_parameter(param_name, param_def)
            yield config

    def _sample_parameter(self, name: str, definition: Any) -> Any:
        """Сэмплировать один параметр"""
        if isinstance(definition, list):
            # Категориальный параметр
            return self.rng.choice(definition)

        elif isinstance(definition, tuple) and len(definition) == 3:
            # Непрерывный параметр: (min, max, distribution)
            low, high, dist = definition

            if dist == 'int':
                return int(self.rng.randint(low, high + 1))
            elif dist == 'log-uniform':
                log_low, log_high = np.log10(low), np.log10(high)
                return 10 ** self.rng.uniform(log_low, log_high)
            else:  # uniform
                return self.rng.uniform(low, high)

        else:
            raise ValueError(f"Unknown parameter definition: {definition}")

# Использование
optimizer = RandomSearchOptimizer(FULL_SEARCH_SPACE, n_trials=200)
for i, config in enumerate(optimizer.generate_configs()):
    print(f"Trial {i}: {config}")
    # Оценить конфигурацию
```

### 2.3 Bayesian Optimization (Optuna)

```python
import optuna
from optuna.samplers import TPESampler

class BayesianOptimizer:
    """Bayesian оптимизация с Optuna"""

    def __init__(
        self,
        search_space: dict,
        objective_function: callable,
        n_trials: int = 100,
        direction: str = 'maximize',
        seed: int = 42
    ):
        self.search_space = search_space
        self.objective_function = objective_function
        self.n_trials = n_trials
        self.direction = direction

        # Создать study
        self.study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=20,
                interval_steps=5
            )
        )

    def optimize(self) -> dict:
        """
        Запустить оптимизацию

        Returns:
            Лучшая найденная конфигурация
        """
        self.study.optimize(
            self._objective_wrapper,
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
            'best_trial': self.study.best_trial.number
        }

    def _objective_wrapper(self, trial: optuna.Trial) -> float:
        """Обёртка для objective function"""
        # Предложить параметры
        config = self._suggest_params(trial)

        # Оценить
        score = self.objective_function(trial, config)

        return score

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """Предложить параметры для trial"""
        config = {}

        for param_name, param_def in self.search_space.items():
            if isinstance(param_def, list):
                # Категориальный
                config[param_name] = trial.suggest_categorical(
                    param_name, param_def
                )

            elif isinstance(param_def, tuple) and len(param_def) == 3:
                low, high, dist = param_def

                if dist == 'int':
                    config[param_name] = trial.suggest_int(
                        param_name, low, high
                    )
                elif dist == 'log-uniform':
                    config[param_name] = trial.suggest_float(
                        param_name, low, high, log=True
                    )
                else:  # uniform
                    config[param_name] = trial.suggest_float(
                        param_name, low, high
                    )

        return config

    def plot_optimization_history(self):
        """Визуализация истории оптимизации"""
        fig = optuna.visualization.plot_optimization_history(self.study)
        fig.show()

    def plot_param_importances(self):
        """Важность параметров"""
        fig = optuna.visualization.plot_param_importances(self.study)
        fig.show()

# Пример objective function
def objective(trial: optuna.Trial, config: dict) -> float:
    """
    Objective function для оптимизации

    Args:
        trial: Optuna trial
        config: Конфигурация параметров

    Returns:
        Метрика для оптимизации (например, Sharpe ratio)
    """
    # 1. Подготовить данные
    data = prepare_data(config)

    # 2. Обучить модель
    model = train_model(config, data)

    # 3. Бэктест
    backtest_result = run_backtest(model, config, data)

    # 4. Вернуть целевую метрику
    sharpe = backtest_result['sharpe_ratio']

    # 5. Pruning для ранней остановки неперспективных trials
    if trial.should_prune():
        raise optuna.TrialPruned()

    return sharpe

# Использование
optimizer = BayesianOptimizer(
    search_space=FULL_SEARCH_SPACE[ParameterCategory.MODEL],
    objective_function=objective,
    n_trials=100,
    direction='maximize'
)

best_config = optimizer.optimize()
print(f"Best params: {best_config['best_params']}")
print(f"Best Sharpe: {best_config['best_value']}")
```

### 2.4 Генетические алгоритмы

```python
from deap import base, creator, tools, algorithms
import random

class GeneticAlgorithmOptimizer:
    """Генетический алгоритм для оптимизации"""

    def __init__(
        self,
        search_space: dict,
        objective_function: callable,
        population_size: int = 50,
        n_generations: int = 100,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        seed: int = 42
    ):
        self.search_space = search_space
        self.objective = objective_function
        self.pop_size = population_size
        self.n_gen = n_generations
        self.cx_prob = crossover_prob
        self.mut_prob = mutation_prob

        random.seed(seed)

        # Настроить DEAP
        self._setup_deap()

    def _setup_deap(self):
        """Настроить DEAP framework"""
        # Создать типы
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Регистрировать операторы
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat,
                             list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _create_individual(self):
        """Создать случайную особь (конфигурацию)"""
        individual = []
        for param_name, param_def in self.search_space.items():
            value = self._random_value(param_def)
            individual.append(value)
        return creator.Individual(individual)

    def _random_value(self, definition):
        """Случайное значение параметра"""
        if isinstance(definition, list):
            return random.choice(definition)
        elif isinstance(definition, tuple) and len(definition) == 3:
            low, high, dist = definition
            if dist == 'int':
                return random.randint(low, high)
            elif dist == 'log-uniform':
                return 10 ** random.uniform(np.log10(low), np.log10(high))
            else:
                return random.uniform(low, high)

    def _evaluate(self, individual):
        """Оценить особь"""
        # Преобразовать в конфигурацию
        config = self._individual_to_config(individual)

        # Оценить
        score = self.objective(config)

        return (score,)

    def _individual_to_config(self, individual) -> dict:
        """Преобразовать особь в конфигурацию"""
        config = {}
        param_names = list(self.search_space.keys())
        for i, param_name in enumerate(param_names):
            config[param_name] = individual[i]
        return config

    def _mutate(self, individual):
        """Мутация особи"""
        for i, (param_name, param_def) in enumerate(self.search_space.items()):
            if random.random() < 0.1:  # 10% шанс мутации каждого гена
                individual[i] = self._random_value(param_def)
        return (individual,)

    def optimize(self) -> dict:
        """
        Запустить генетический алгоритм

        Returns:
            Лучшая найденная конфигурация
        """
        # Создать начальную популяцию
        population = self.toolbox.population(n=self.pop_size)

        # Статистика
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Запустить эволюцию
        population, logbook = algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=self.cx_prob,
            mutpb=self.mut_prob,
            ngen=self.n_gen,
            stats=stats,
            verbose=True
        )

        # Лучшая особь
        best_individual = tools.selBest(population, k=1)[0]
        best_config = self._individual_to_config(best_individual)

        return {
            'best_params': best_config,
            'best_value': best_individual.fitness.values[0],
            'logbook': logbook
        }
```

## 3. Многоуровневая оптимизация

### 3.1 Последовательная оптимизация блоков

```python
class HierarchicalOptimizer:
    """Многоуровневая оптимизация с заморозкой блоков"""

    def __init__(self, full_search_space: dict):
        self.full_space = full_search_space
        self.frozen_params = {}
        self.optimization_history = []

    def optimize_sequential(self) -> dict:
        """
        Последовательная оптимизация блоков

        Порядок:
        1. Данные и разметка
        2. Признаки
        3. Модель (грубо)
        4. Обучение
        5. Модель (точно)
        6. Стратегия
        7. Риск-менеджмент
        8. Финальная валидация

        Returns:
            Оптимальная конфигурация
        """
        logger.info("Starting hierarchical optimization...")

        # Этап 1: Данные и разметка
        logger.info("Stage 1: Optimizing data and labeling...")
        best_data_config = self._optimize_block(
            categories=[ParameterCategory.DATA, ParameterCategory.LABELING],
            n_trials=50,
            name="data_labeling"
        )
        self._freeze_params(best_data_config)

        # Этап 2: Признаки
        logger.info("Stage 2: Optimizing features...")
        best_features = self._optimize_block(
            categories=[ParameterCategory.FEATURES],
            n_trials=100,
            name="features"
        )
        self._freeze_params(best_features)

        # Этап 3: Грубая оптимизация модели
        logger.info("Stage 3: Coarse model optimization...")
        coarse_model_space = self._get_coarse_model_space()
        best_model_coarse = self._optimize_block(
            categories=[ParameterCategory.MODEL],
            n_trials=50,
            name="model_coarse",
            custom_space=coarse_model_space
        )

        # Этап 4: Параметры обучения
        logger.info("Stage 4: Optimizing training parameters...")
        best_training = self._optimize_block(
            categories=[ParameterCategory.TRAINING],
            n_trials=30,
            name="training"
        )
        self._freeze_params(best_training)

        # Этап 5: Точная настройка модели
        logger.info("Stage 5: Fine-tuning model...")
        fine_model_space = self._get_fine_model_space(best_model_coarse)
        best_model_fine = self._optimize_block(
            categories=[ParameterCategory.MODEL],
            n_trials=100,
            name="model_fine",
            custom_space=fine_model_space
        )
        self._freeze_params(best_model_fine)

        # Этап 6: Параметры стратегии
        logger.info("Stage 6: Optimizing strategy parameters...")
        best_strategy = self._optimize_block(
            categories=[ParameterCategory.STRATEGY],
            n_trials=100,
            name="strategy"
        )
        self._freeze_params(best_strategy)

        # Этап 7: Риск-менеджмент
        logger.info("Stage 7: Optimizing risk management...")
        best_risk = self._optimize_block(
            categories=[ParameterCategory.RISK],
            n_trials=30,
            name="risk"
        )
        self._freeze_params(best_risk)

        # Этап 8: Финальная валидация
        logger.info("Stage 8: Final validation...")
        final_config = self.frozen_params.copy()
        final_score = self._validate_configuration(final_config)

        logger.info(f"Optimization complete. Final score: {final_score}")

        return {
            'config': final_config,
            'score': final_score,
            'history': self.optimization_history
        }

    def _optimize_block(
        self,
        categories: List[ParameterCategory],
        n_trials: int,
        name: str,
        custom_space: dict = None
    ) -> dict:
        """Оптимизировать один блок параметров"""
        # Составить пространство поиска для блока
        if custom_space:
            block_space = custom_space
        else:
            block_space = {}
            for cat in categories:
                if cat in self.full_space:
                    block_space.update(self.full_space[cat])

        # Создать objective function с замороженными параметрами
        def objective(trial, config):
            # Объединить с замороженными параметрами
            full_config = {**self.frozen_params, **config}

            # Оценить
            score = evaluate_configuration(full_config)

            return score

        # Оптимизировать
        optimizer = BayesianOptimizer(
            search_space=block_space,
            objective_function=objective,
            n_trials=n_trials,
            direction='maximize'
        )

        result = optimizer.optimize()

        # Сохранить в историю
        self.optimization_history.append({
            'name': name,
            'categories': [c.value for c in categories],
            'best_params': result['best_params'],
            'best_value': result['best_value'],
            'n_trials': n_trials
        })

        return result['best_params']

    def _freeze_params(self, params: dict):
        """Заморозить параметры"""
        self.frozen_params.update(params)
        logger.info(f"Frozen {len(params)} parameters: {list(params.keys())}")

    def _get_coarse_model_space(self) -> dict:
        """Грубое пространство модели (меньше вариантов)"""
        return {
            'model_type': ['lightgbm', 'catboost', 'xgboost'],
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.05, 0.1],
        }

    def _get_fine_model_space(self, coarse_best: dict) -> dict:
        """Точное пространство вокруг найденных значений"""
        model_type = coarse_best['model_type']

        # Сузить диапазон вокруг лучших значений
        fine_space = {
            'model_type': [model_type],  # Зафиксировать тип
        }

        # Для n_estimators
        n_est = coarse_best['n_estimators']
        fine_space['n_estimators'] = (
            max(50, n_est - 50),
            n_est + 50,
            'int'
        )

        # Для max_depth
        depth = coarse_best['max_depth']
        fine_space['max_depth'] = (
            max(3, depth - 2),
            min(12, depth + 2),
            'int'
        )

        # Для learning_rate - логарифмический диапазон
        lr = coarse_best['learning_rate']
        fine_space['learning_rate'] = (
            lr * 0.5,
            lr * 2.0,
            'log-uniform'
        )

        return fine_space

    def _validate_configuration(self, config: dict) -> float:
        """Финальная валидация на hold-out set"""
        # Оценить на тестовом наборе
        score = evaluate_configuration(config, test=True)
        return score
```

### 3.2 Параллельная оптимизация (Distributed)

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

class DistributedOptimizer:
    """Распределённая оптимизация на нескольких машинах"""

    def __init__(
        self,
        search_space: dict,
        n_workers: int = 4,
        trials_per_worker: int = 25
    ):
        self.search_space = search_space
        self.n_workers = n_workers
        self.trials_per_worker = trials_per_worker

    def optimize_distributed(self) -> dict:
        """
        Распределённая оптимизация

        Returns:
            Лучшая конфигурация со всех workers
        """
        # Создать shared study (PostgreSQL backend для Optuna)
        study = optuna.create_study(
            study_name='distributed_optimization',
            storage='postgresql://user:password@localhost/optuna',
            direction='maximize',
            load_if_exists=True
        )

        # Запустить workers параллельно
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(
                    self._worker_optimize,
                    worker_id=i,
                    study_name='distributed_optimization',
                    n_trials=self.trials_per_worker
                )
                for i in range(self.n_workers)
            ]

            # Собрать результаты
            for future in as_completed(futures):
                worker_result = future.result()
                logger.info(f"Worker completed: {worker_result}")

        # Получить лучший результат
        best_trial = study.best_trial

        return {
            'best_params': best_trial.params,
            'best_value': best_trial.value,
            'total_trials': len(study.trials)
        }

    def _worker_optimize(
        self,
        worker_id: int,
        study_name: str,
        n_trials: int
    ) -> dict:
        """Worker function для оптимизации"""
        logger.info(f"Worker {worker_id} starting {n_trials} trials...")

        # Подключиться к shared study
        study = optuna.load_study(
            study_name=study_name,
            storage='postgresql://user:password@localhost/optuna'
        )

        # Оптимизировать
        study.optimize(
            self._objective,
            n_trials=n_trials
        )

        return {
            'worker_id': worker_id,
            'trials_completed': n_trials
        }
```

## 4. Оптимизация порога предсказаний

### 4.1 Expected PnL Optimization

```python
class ThresholdOptimizer:
    """Оптимизация порога для максимизации ожидаемой прибыли"""

    def __init__(
        self,
        predictions: np.ndarray,
        true_returns: np.ndarray,
        take_profit: float,
        stop_loss: float,
        commission: float = 0.001
    ):
        """
        Args:
            predictions: Вероятности роста [0, 1]
            true_returns: Фактические доходности
            take_profit: Тейк-профит (например, 0.02 = 2%)
            stop_loss: Стоп-лосс (например, 0.01 = 1%)
            commission: Комиссия на сделку
        """
        self.predictions = predictions
        self.true_returns = true_returns
        self.tp = take_profit
        self.sl = stop_loss
        self.commission = commission

    def optimize_threshold(
        self,
        threshold_range: tuple = (0.5, 0.95),
        n_points: int = 50,
        min_trades: int = 10
    ) -> dict:
        """
        Найти оптимальный порог

        Args:
            threshold_range: Диапазон порогов для проверки
            n_points: Количество точек в диапазоне
            min_trades: Минимальное количество сделок

        Returns:
            Оптимальный порог и метрики
        """
        thresholds = np.linspace(*threshold_range, n_points)
        results = []

        for threshold in thresholds:
            # Рассчитать метрики для порога
            metrics = self._evaluate_threshold(threshold)

            # Проверить минимум сделок
            if metrics['n_trades'] < min_trades:
                metrics['expected_pnl'] = -np.inf

            results.append({
                'threshold': threshold,
                **metrics
            })

        # Найти лучший порог
        best_idx = np.argmax([r['expected_pnl'] for r in results])
        best_result = results[best_idx]

        return {
            'optimal_threshold': best_result['threshold'],
            'expected_pnl': best_result['expected_pnl'],
            'win_rate': best_result['win_rate'],
            'n_trades': best_result['n_trades'],
            'all_results': results
        }

    def _evaluate_threshold(self, threshold: float) -> dict:
        """Оценить порог"""
        # Сигналы выше порога
        signals = self.predictions >= threshold
        n_trades = signals.sum()

        if n_trades == 0:
            return {
                'expected_pnl': 0,
                'win_rate': 0,
                'n_trades': 0,
                'avg_win': 0,
                'avg_loss': 0
            }

        # Фактические доходности сигналов
        signal_returns = self.true_returns[signals]

        # Симулировать результаты с TP/SL
        trade_results = []
        for ret in signal_returns:
            if ret >= self.tp:
                # Достигнут тейк-профит
                trade_results.append(self.tp - self.commission)
            elif ret <= -self.sl:
                # Достигнут стоп-лосс
                trade_results.append(-self.sl - self.commission)
            else:
                # Выход по сигналу
                trade_results.append(ret - self.commission)

        trade_results = np.array(trade_results)

        # Метрики
        wins = trade_results > 0
        losses = trade_results < 0

        win_rate = wins.sum() / len(trade_results)
        avg_win = trade_results[wins].mean() if wins.any() else 0
        avg_loss = trade_results[losses].mean() if losses.any() else 0

        # Expected PnL
        expected_pnl = trade_results.mean()

        return {
            'expected_pnl': expected_pnl,
            'win_rate': win_rate,
            'n_trades': n_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }

    def plot_threshold_curve(self, results: list):
        """Визуализация зависимости метрик от порога"""
        import matplotlib.pyplot as plt

        thresholds = [r['threshold'] for r in results]
        expected_pnls = [r['expected_pnl'] for r in results]
        win_rates = [r['win_rate'] for r in results]
        n_trades = [r['n_trades'] for r in results]

        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        # Expected PnL
        axes[0].plot(thresholds, expected_pnls)
        axes[0].set_ylabel('Expected PnL')
        axes[0].set_title('Expected PnL vs Threshold')
        axes[0].grid(True)

        # Win Rate
        axes[1].plot(thresholds, win_rates)
        axes[1].set_ylabel('Win Rate')
        axes[1].set_title('Win Rate vs Threshold')
        axes[1].grid(True)

        # Number of Trades
        axes[2].plot(thresholds, n_trades)
        axes[2].set_ylabel('Number of Trades')
        axes[2].set_xlabel('Threshold')
        axes[2].set_title('Trade Count vs Threshold')
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

# Использование
optimizer = ThresholdOptimizer(
    predictions=model_predictions,
    true_returns=actual_returns,
    take_profit=0.02,
    stop_loss=0.01,
    commission=0.001
)

result = optimizer.optimize_threshold()
print(f"Optimal threshold: {result['optimal_threshold']:.3f}")
print(f"Expected PnL: {result['expected_pnl']:.4f}")
print(f"Win rate: {result['win_rate']:.2%}")
print(f"N trades: {result['n_trades']}")

optimizer.plot_threshold_curve(result['all_results'])
```

## 5. Budget Management

### 5.1 Adaptive Budget Allocation

```python
class BudgetManager:
    """Управление бюджетом trials для оптимизации"""

    def __init__(
        self,
        total_budget: int,
        categories: List[ParameterCategory],
        priorities: dict = None
    ):
        """
        Args:
            total_budget: Общий бюджет trials
            categories: Категории для оптимизации
            priorities: Приоритеты категорий (1-10)
        """
        self.total_budget = total_budget
        self.categories = categories
        self.priorities = priorities or {cat: 5 for cat in categories}
        self.spent_budget = {cat: 0 for cat in categories}

    def allocate_budget(self) -> dict:
        """
        Распределить бюджет между категориями

        Returns:
            Словарь {категория: количество trials}
        """
        # Нормализовать приоритеты
        total_priority = sum(self.priorities.values())

        allocation = {}
        for cat in self.categories:
            priority = self.priorities.get(cat, 5)
            allocated = int(self.total_budget * (priority / total_priority))
            allocation[cat] = max(10, allocated)  # Минимум 10 trials

        return allocation

    def update_spent(self, category: ParameterCategory, n_trials: int):
        """Обновить потраченный бюджет"""
        self.spent_budget[category] += n_trials

    def get_remaining_budget(self, category: ParameterCategory) -> int:
        """Получить оставшийся бюджет"""
        allocated = self.allocate_budget()[category]
        spent = self.spent_budget[category]
        return max(0, allocated - spent)

    def reallocate_unused(self):
        """Перераспределить неиспользованный бюджет"""
        allocation = self.allocate_budget()
        unused = {}

        for cat in self.categories:
            remaining = self.get_remaining_budget(cat)
            if remaining > 0:
                unused[cat] = remaining

        # Перераспределить пропорционально приоритетам
        total_unused = sum(unused.values())
        if total_unused == 0:
            return {}

        reallocation = {}
        for cat, priority in self.priorities.items():
            if cat not in unused:
                additional = int(total_unused * (priority / sum(self.priorities.values())))
                reallocation[cat] = allocation[cat] + additional

        return reallocation
```

### 5.2 Early Stopping

```python
class EarlyStoppingOptimizer:
    """Оптимизатор с ранней остановкой неперспективных trials"""

    def __init__(
        self,
        search_space: dict,
        objective_function: callable,
        patience: int = 10,
        min_improvement: float = 0.001
    ):
        self.search_space = search_space
        self.objective = objective_function
        self.patience = patience
        self.min_improvement = min_improvement

        self.best_score = -np.inf
        self.trials_without_improvement = 0
        self.trial_history = []

    def optimize(self, max_trials: int = 100) -> dict:
        """
        Оптимизация с early stopping

        Args:
            max_trials: Максимальное количество trials

        Returns:
            Лучшая конфигурация
        """
        for trial_num in range(max_trials):
            # Проверить условие early stopping
            if self.trials_without_improvement >= self.patience:
                logger.info(
                    f"Early stopping at trial {trial_num}: "
                    f"no improvement for {self.patience} trials"
                )
                break

            # Сгенерировать конфигурацию
            config = self._sample_config()

            # Оценить
            score = self.objective(config)

            # Обновить историю
            self.trial_history.append({
                'trial': trial_num,
                'config': config,
                'score': score
            })

            # Проверить улучшение
            if score > self.best_score + self.min_improvement:
                improvement = score - self.best_score
                logger.info(
                    f"Trial {trial_num}: New best score {score:.4f} "
                    f"(+{improvement:.4f})"
                )
                self.best_score = score
                self.trials_without_improvement = 0
            else:
                self.trials_without_improvement += 1

        # Вернуть лучшую конфигурацию
        best_trial = max(self.trial_history, key=lambda x: x['score'])

        return {
            'best_params': best_trial['config'],
            'best_score': best_trial['score'],
            'total_trials': len(self.trial_history)
        }

    def _sample_config(self) -> dict:
        """Сэмплировать случайную конфигурацию"""
        # Используем random search для простоты
        optimizer = RandomSearchOptimizer(self.search_space, n_trials=1)
        return next(optimizer.generate_configs())
```

## 6. Валидация и предотвращение overfitting

### 6.1 Walk-Forward Validation

```python
class WalkForwardOptimizer:
    """Оптимизация с walk-forward валидацией"""

    def __init__(
        self,
        data: pd.DataFrame,
        train_window: int = 252,  # 1 год торговых дней
        test_window: int = 63,    # 3 месяца
        step: int = 21            # 1 месяц
    ):
        self.data = data
        self.train_window = train_window
        self.test_window = test_window
        self.step = step

    def optimize_walk_forward(
        self,
        search_space: dict,
        objective_function: callable,
        n_trials_per_window: int = 50
    ) -> dict:
        """
        Walk-forward оптимизация

        Returns:
            Результаты для каждого окна + агрегированные метрики
        """
        results = []

        # Разделить на окна
        windows = self._create_windows()

        for i, (train_indices, test_indices) in enumerate(windows):
            logger.info(f"Window {i+1}/{len(windows)}")

            # Обучающая выборка
            train_data = self.data.iloc[train_indices]

            # Оптимизировать на обучающей выборке
            def train_objective(config):
                return objective_function(config, train_data)

            optimizer = BayesianOptimizer(
                search_space=search_space,
                objective_function=lambda trial, cfg: train_objective(cfg),
                n_trials=n_trials_per_window
            )

            opt_result = optimizer.optimize()
            best_config = opt_result['best_params']
            train_score = opt_result['best_value']

            # Тестировать на тестовой выборке
            test_data = self.data.iloc[test_indices]
            test_score = objective_function(best_config, test_data)

            results.append({
                'window': i,
                'train_period': (train_indices[0], train_indices[-1]),
                'test_period': (test_indices[0], test_indices[-1]),
                'best_config': best_config,
                'train_score': train_score,
                'test_score': test_score,
                'overfit': train_score - test_score
            })

            logger.info(
                f"Window {i}: Train={train_score:.4f}, "
                f"Test={test_score:.4f}, Overfit={train_score-test_score:.4f}"
            )

        # Агрегированные метрики
        avg_train = np.mean([r['train_score'] for r in results])
        avg_test = np.mean([r['test_score'] for r in results])
        avg_overfit = np.mean([r['overfit'] for r in results])

        return {
            'windows': results,
            'avg_train_score': avg_train,
            'avg_test_score': avg_test,
            'avg_overfit': avg_overfit,
            'consistency': np.std([r['test_score'] for r in results])
        }

    def _create_windows(self) -> list:
        """Создать окна для walk-forward"""
        windows = []
        start = 0

        while start + self.train_window + self.test_window <= len(self.data):
            train_end = start + self.train_window
            test_end = train_end + self.test_window

            train_indices = list(range(start, train_end))
            test_indices = list(range(train_end, test_end))

            windows.append((train_indices, test_indices))

            start += self.step

        return windows
```

## 7. Best Practices

### 7.1 Рекомендации
- ✅ Начинать с random search для exploration
- ✅ Использовать Bayesian optimization для exploitation
- ✅ Многоуровневая оптимизация с заморозкой блоков
- ✅ Walk-forward validation для предотвращения overfitting
- ✅ Ограничение бюджета с умным распределением
- ✅ Early stopping для неперспективных trials
- ✅ Оптимизация порога отдельно от модели
- ✅ Логирование всех trials в MLflow
- ✅ Визуализация процесса оптимизации

### 7.2 Антипаттерны
- ❌ Grid search на большом пространстве (комбинаторный взрыв)
- ❌ Оптимизация всех параметров одновременно
- ❌ Игнорирование overfitting
- ❌ Оптимизация на всех данных без валидации
- ❌ Отсутствие early stopping
- ❌ Неучёт вычислительных ограничений
- ❌ Оптимизация на тренировочном наборе и тестирование на нём же

### 7.3 Checklist
- [ ] Определено пространство поиска для каждой категории
- [ ] Выбран подходящий метод оптимизации
- [ ] Настроена walk-forward валидация
- [ ] Реализован early stopping
- [ ] Бюджет trials распределён разумно
- [ ] Логируются все эксперименты
- [ ] Проверяется устойчивость результатов
- [ ] Визуализируется процесс оптимизации
