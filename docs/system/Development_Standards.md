# Стандарты разработки

## 1. Code Style

### 1.1 Python Style Guide (PEP 8)

```python
# Хороший пример
class FeatureCalculator:
    """Calculate technical indicators from OHLCV data."""

    def __init__(self, config: dict):
        """
        Initialize calculator.

        Args:
            config: Configuration dictionary containing indicator parameters
        """
        self.config = config
        self.indicators = []

    def calculate_sma(
        self,
        data: pd.DataFrame,
        window: int = 20,
        column: str = 'close'
    ) -> pd.Series:
        """
        Calculate Simple Moving Average.

        Args:
            data: Input OHLCV dataframe
            window: Moving average window size
            column: Column to calculate SMA on

        Returns:
            Series with SMA values

        Raises:
            ValueError: If window is not positive
        """
        if window <= 0:
            raise ValueError(f"Window must be positive, got {window}")

        return data[column].rolling(window=window).mean()
```

### 1.2 Naming Conventions

```python
# Переменные и функции: snake_case
user_id = 123
def calculate_profit():
    pass

# Классы: PascalCase
class ModelTrainer:
    pass

# Константы: UPPER_SNAKE_CASE
MAX_RETRIES = 3
API_TIMEOUT = 30

# Приватные: префикс _
class MyClass:
    def __init__(self):
        self._private_var = 42

    def _private_method(self):
        pass

# Очень приватные: префикс __
class BaseClass:
    def __init__(self):
        self.__very_private = 100
```

### 1.3 Line Length and Formatting

```python
# Максимум 88-100 символов (Black default: 88)

# ✅ Хорошо
result = some_function_with_long_name(
    parameter_one=value_one,
    parameter_two=value_two,
    parameter_three=value_three,
)

# ✅ Длинные строки разбивать
error_message = (
    f"Failed to process {len(items)} items: "
    f"{error_details}. Please check the logs."
)

# ✅ Импорты группировать
# 1. Стандартная библиотека
import os
import sys
from datetime import datetime

# 2. Сторонние библиотеки
import numpy as np
import pandas as pd
import torch

# 3. Локальные модули
from src.data import DataLoader
from src.features import FeatureCalculator
```

## 2. Type Hints

### 2.1 Использование аннотаций типов

```python
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path

def load_data(
    file_path: Path,
    columns: Optional[List[str]] = None,
    chunksize: Optional[int] = None
) -> pd.DataFrame:
    """Load data from file."""
    ...

def calculate_metrics(
    predictions: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    return {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88
    }

class Model:
    def predict(self, X: np.ndarray) -> Union[np.ndarray, List[float]]:
        """Make predictions."""
        ...

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> Tuple[List[float], List[float]]:
        """Train model and return train/val losses."""
        ...
```

### 2.2 Mypy Configuration

```ini
# mypy.ini
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_any_unimported = False
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
check_untyped_defs = True
strict_equality = True

# Игнорировать для некоторых библиотек без type stubs
[mypy-sklearn.*]
ignore_missing_imports = True

[mypy-pandas.*]
ignore_missing_imports = True
```

## 3. Docstrings

### 3.1 Google Style Docstrings

```python
def train_model(
    data: pd.DataFrame,
    model_type: str,
    hyperparameters: dict,
    save_path: Optional[Path] = None
) -> Tuple[object, Dict[str, float]]:
    """
    Train a machine learning model on provided data.

    This function handles the complete training pipeline including
    data preparation, model initialization, training loop, and
    evaluation.

    Args:
        data: Training dataset with features and labels
        model_type: Type of model to train ('lightgbm', 'xgboost', etc.)
        hyperparameters: Dictionary of model hyperparameters
        save_path: Optional path to save trained model

    Returns:
        A tuple containing:
            - Trained model object
            - Dictionary of evaluation metrics (accuracy, loss, etc.)

    Raises:
        ValueError: If model_type is not supported
        RuntimeError: If training fails

    Example:
        >>> data = pd.read_parquet('training_data.parquet')
        >>> model, metrics = train_model(
        ...     data,
        ...     model_type='lightgbm',
        ...     hyperparameters={'learning_rate': 0.01}
        ... )
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        Accuracy: 0.856

    Note:
        Large datasets will be automatically batched to prevent OOM.
    """
    ...
```

### 3.2 Классы и модули

```python
"""
Data loading and preprocessing module.

This module provides classes and functions for loading financial data
from various sources (local files, APIs) and preprocessing it for
model training and backtesting.

Classes:
    DataLoader: Main class for loading data
    DataPreprocessor: Data cleaning and transformation
    DataValidator: Data quality checks

Functions:
    load_ohlcv: Load OHLCV data from file
    resample_timeframe: Resample data to different timeframe
"""

class DataLoader:
    """
    Load financial data from multiple sources.

    This class handles loading data from local files (Parquet, CSV) and
    external APIs (Tinkoff Investments). It provides caching, validation,
    and automatic retry logic.

    Attributes:
        source_type: Type of data source ('local', 'api')
        cache_dir: Directory for caching downloaded data
        api_client: API client instance if using API source

    Example:
        >>> loader = DataLoader(source_type='api', api_token='...')
        >>> data = loader.load(ticker='SBER', from_date='2020-01-01')
        >>> print(len(data))
        10000
    """
```

## 4. Code Organization

### 4.1 Структура файлов

```
src/
├── data/
│   ├── __init__.py
│   ├── loaders.py         # Загрузка данных
│   ├── preprocessors.py   # Препроцессинг
│   └── validators.py      # Валидация
├── features/
│   ├── __init__.py
│   ├── indicators.py      # Технические индикаторы
│   ├── extractors.py      # Feature extraction
│   └── selectors.py       # Feature selection
├── modeling/
│   ├── __init__.py
│   ├── models.py          # Модели
│   ├── trainers.py        # Тренировка
│   └── evaluators.py      # Оценка
└── common/
    ├── __init__.py
    ├── utils.py           # Утилиты
    ├── config.py          # Конфигурация
    └── logging.py         # Логирование
```

### 4.2 Размер файлов и функций

```python
# ✅ Функции: 20-50 строк (в среднем)
# ❌ Избегать функций >100 строк

# ✅ Файлы: 200-500 строк (в среднем)
# ❌ Избегать файлов >1000 строк

# Если файл/функция слишком большая - разбить на модули/функции
```

## 5. Pre-commit Hooks

### 5.1 Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.10
        args: ['--line-length=88']

  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ['--profile=black']

  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=88', '--extend-ignore=E203,W503']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-r', 'src/']
```

### 5.2 Black Configuration

```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py310']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''
```

### 5.3 isort Configuration

```toml
# pyproject.toml
[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
```

## 6. Git Workflow

### 6.1 Branch Naming

```bash
# Feature branches
feature/add-lstm-model
feature/tinkoff-api-integration

# Bug fixes
fix/memory-leak-in-backtest
fix/nan-in-rsi-calculation

# Hotfixes
hotfix/critical-data-corruption

# Documentation
docs/update-api-guide

# Refactoring
refactor/split-large-module
```

### 6.2 Commit Messages

```bash
# Формат: <type>: <subject>
#
# <body>
#
# <footer>

# Types:
# feat: новая функциональность
# fix: исправление бага
# docs: документация
# style: форматирование, пробелы, etc
# refactor: рефакторинг
# test: добавление тестов
# chore: обновление зависимостей, etc

# Примеры:

feat: add LSTM model for time series prediction

Implemented LSTM architecture with attention mechanism.
Includes training pipeline and inference optimization.

Closes #123

---

fix: handle NaN values in RSI calculation

Fixed issue where RSI returned inf when all price changes
were zero. Now correctly returns NaN for first N bars.

Fixes #456

---

docs: update API integration guide

Added examples for Tinkoff API rate limiting and error handling.

---

refactor: extract feature calculation to separate module

Moved all technical indicators from modeling.py to new
features/indicators.py module for better organization.
```

### 6.3 Pull Request Process

1. **Создать ветку** от `main`
2. **Разработка** с регулярными коммитами
3. **Тесты** - все должны проходить
4. **Pre-commit hooks** - автоматический запуск
5. **Push** и создание PR
6. **Code Review** - минимум 1 approval
7. **CI/CD** - все проверки должны пройти
8. **Merge** в `main`

```markdown
## PR Template

### Description
Brief description of changes

### Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

### Checklist
- [ ] Tests pass locally
- [ ] Added/updated tests
- [ ] Updated documentation
- [ ] Code follows style guidelines
- [ ] No new warnings
- [ ] Backwards compatible

### Related Issues
Closes #123
```

## 7. Code Review Guidelines

### 7.1 Что проверять

**Функциональность:**
- ✅ Код решает поставленную задачу
- ✅ Обработаны edge cases
- ✅ Нет очевидных багов

**Качество кода:**
- ✅ Читаемость и понятность
- ✅ Следование стандартам
- ✅ Отсутствие дублирования (DRY)
- ✅ Правильная архитектура

**Тесты:**
- ✅ Есть тесты для новой функциональности
- ✅ Тесты покрывают edge cases
- ✅ Тесты проходят

**Документация:**
- ✅ Docstrings для новых функций/классов
- ✅ Обновлена документация если нужно
- ✅ README обновлен при необходимости

### 7.2 Комментарии в review

```python
# ✅ Конструктивные комментарии
"""
Could we extract this into a separate function for better testability?

def _validate_input(data):
    if data is None:
        raise ValueError("Data cannot be None")
    ...
"""

# ✅ С примером
"""
Consider using a more descriptive variable name here:

# Instead of:
x = calculate(data)

# Use:
feature_values = calculate_features(data)
"""

# ❌ Неконструктивные
"""
This is bad.
"""

# ❌ Без объяснения
"""
Change this.
"""
```

## 8. Error Handling

### 8.1 Exception Handling

```python
# ✅ Специфичные исключения
try:
    data = load_data(file_path)
except FileNotFoundError:
    logger.error(f"File not found: {file_path}")
    raise
except pd.errors.ParserError:
    logger.error(f"Failed to parse file: {file_path}")
    raise

# ✅ Кастомные исключения
class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass

class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass

try:
    validate_data(df)
except DataValidationError as e:
    logger.error(f"Validation failed: {e}")
    # Обработка

# ❌ Общий Exception
try:
    risky_operation()
except Exception:
    pass  # Не делать так!

# ❌ Игнорирование исключений
try:
    important_operation()
except:  # Bare except
    pass
```

### 8.2 Assertions

```python
# ✅ Для проверки инвариантов
def calculate_sharpe_ratio(returns: pd.Series) -> float:
    assert len(returns) > 0, "Returns cannot be empty"
    assert returns.notnull().all(), "Returns contain NaN values"

    return returns.mean() / returns.std()

# ❌ Не использовать для валидации пользовательского ввода
# (assertions могут быть отключены с -O)
def process_user_data(data):
    assert data is not None  # Плохо! Использовать if + raise
```

## 9. Logging в коде

### 9.1 Правильное использование

```python
import logging

logger = logging.getLogger(__name__)

# ✅ Логирование с контекстом
logger.info(
    "Training completed",
    extra={
        'model_type': 'lightgbm',
        'epochs': 100,
        'final_loss': 0.234
    }
)

# ✅ Использование правильного уровня
logger.debug("Detailed diagnostic info")
logger.info("Important business event")
logger.warning("Unexpected but handled")
logger.error("Error occurred", exc_info=True)
logger.critical("System failure")

# ✅ Lazy evaluation для дорогих операций
logger.debug("Data shape: %s", lambda: expensive_format(data))

# ❌ Не использовать print()
print("Debug info")  # Плохо!

# ❌ Не форматировать заранее
logger.debug(f"Processing {len(data)} rows")  # Плохо
# Лучше:
logger.debug("Processing %d rows", len(data))
```

## 10. Performance Considerations

### 10.1 Профилирование

```python
# Использовать profiler для оптимизации
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Код для профилирования
expensive_operation()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Топ-20 функций

# Или line_profiler для построчного анализа
@profile  # Декоратор line_profiler
def slow_function():
    ...
```

### 10.2 Избегать преждевременной оптимизации

```python
# ❌ Преждевременная оптимизация
def calculate(data):
    # Сложная оптимизация которая делает код нечитаемым
    # но дает выигрыш 0.1%
    ...

# ✅ Сначала простая и понятная реализация
def calculate(data):
    # Простой читаемый код
    return data.apply(simple_function)

# Если профилирование показало что это узкое место:
# ✅ Тогда оптимизировать
def calculate_optimized(data):
    # Оптимизированная версия с комментариями
    # почему нужна оптимизация
    return fast_vectorized_operation(data)
```

## 11. Security

### 11.1 Не хардкодить секреты

```python
# ❌ Плохо
API_TOKEN = "t.xxxxxxxxxxxxxxxxxxx"
DATABASE_PASSWORD = "mypassword123"

# ✅ Хорошо - из переменных окружения
import os

API_TOKEN = os.getenv('TINKOFF_API_TOKEN')
if not API_TOKEN:
    raise ValueError("TINKOFF_API_TOKEN environment variable not set")

# ✅ Или из защищенного хранилища
from src.common.secrets import get_secret

API_TOKEN = get_secret('tinkoff_api_token')
```

### 11.2 Input Validation

```python
# ✅ Валидация всех внешних входов
def load_config(config_path: str):
    # Проверить путь
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Проверить что это файл в разрешённой директории
    allowed_dir = Path("configs/")
    if not Path(config_path).resolve().is_relative_to(allowed_dir.resolve()):
        raise SecurityError("Config path outside allowed directory")

    # Загрузить и валидировать схему
    config = yaml.safe_load(Path(config_path).read_text())
    validate_config_schema(config)

    return config
```

## 12. Best Practices Checklist

### 12.1 Before Committing
- [ ] Код соответствует style guide
- [ ] Все тесты проходят
- [ ] Добавлены docstrings
- [ ] Нет закомментированного кода
- [ ] Нет debug print statements
- [ ] Type hints добавлены
- [ ] Pre-commit hooks прошли

### 12.2 Before PR
- [ ] Rebase на latest main
- [ ] CI/CD проходит
- [ ] Документация обновлена
- [ ] CHANGELOG обновлен
- [ ] Нет merge conflicts
- [ ] Commit messages информативные

### 12.3 Code Quality Metrics
- **Complexity**: Cyclomatic complexity < 10
- **Line length**: < 88-100 символов
- **Function length**: < 50 строк (в среднем)
- **Test coverage**: > 80%
- **Documentation**: Все public API документированы
