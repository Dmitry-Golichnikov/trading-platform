# Модуль Моделирования

Базовая инфраструктура для обучения и работы с моделями машинного обучения.

## Структура

```
modeling/
├── base.py                    # Базовый интерфейс BaseModel
├── registry.py                # Реестр моделей
├── trainer.py                 # ModelTrainer для обучения
├── utils.py                   # Утилиты (seed, device management)
├── splitting.py               # Методы разделения данных
├── serialization.py           # Сохранение/загрузка моделей
├── sanity_checks.py           # Проверки перед обучением
├── callbacks/                 # Callbacks для обучения
│   ├── base.py
│   ├── early_stopping.py
│   ├── model_checkpoint.py
│   ├── mlflow_logger.py
│   └── progress_bar.py
├── loss_functions/            # Функции потерь
│   ├── base.py
│   ├── registry.py
│   ├── classification/        # BCE, Focal, etc.
│   ├── regression/            # MSE, MAE, Huber, etc.
│   └── trading_custom/        # Directional, Profit-based, etc.
└── models/                    # Реализации моделей (будет в Этапе 07)
```

## Основные компоненты

### BaseModel

Единый интерфейс для всех моделей:

```python
from src.modeling import BaseModel

class MyModel(BaseModel):
    def fit(self, X, y, **kwargs):
        # Обучение
        return self

    def predict(self, X):
        # Предсказание
        return predictions

    def save(self, path):
        # Сохранение
        pass

    @classmethod
    def load(cls, path):
        # Загрузка
        return model
```

### ModelRegistry

Регистрация и создание моделей:

```python
from src.modeling import ModelRegistry, BaseModel

@ModelRegistry.register("my_model", tags=["custom"])
class MyModel(BaseModel):
    pass

# Создание модели
model = ModelRegistry.create("my_model", param1=value1)
```

### ModelTrainer

Обучение с callbacks и tracking:

```python
from src.modeling import ModelTrainer, EarlyStopping, ModelCheckpoint

trainer = ModelTrainer(model, experiment_name="experiment_01")

result = trainer.train(
    X_train, y_train,
    X_val, y_val,
    callbacks=[
        EarlyStopping(patience=10),
        ModelCheckpoint("models/best.pt")
    ]
)

print(result.metrics)
```

### Loss Functions

Система функций потерь:

```python
from src.modeling.loss_functions import LossRegistry

# Получение loss function
loss_fn = LossRegistry.get('focal', alpha=0.25, gamma=2.0)

# Использование
loss = loss_fn(predictions, targets)

# Список доступных
losses = LossRegistry.list_losses(category='trading')
```

### Callbacks

Доступные callbacks:

- **EarlyStopping** - ранняя остановка при отсутствии улучшения
- **ModelCheckpoint** - сохранение лучшей модели
- **MLflowLogger** - логирование в MLflow
- **ProgressBar** - отображение прогресса обучения

```python
from src.modeling.callbacks import EarlyStopping, ProgressBar

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10),
    ProgressBar(total_epochs=100)
]
```

### Data Splitting

Методы разделения временных рядов:

```python
from src.modeling import DataSplitter

# Последовательное разделение
train, val, test = DataSplitter.split_sequential(data)

# Walk-forward
splits = DataSplitter.split_walk_forward(data, n_splits=5)

# С embargo (purged)
train, val, test = DataSplitter.split_purged(
    data,
    embargo_td=timedelta(days=1)
)
```

### Serialization

Сохранение и загрузка моделей:

```python
from src.modeling import ModelSerializer

# Сохранение
ModelSerializer.save(model, Path('model.pkl'))

# Загрузка
model = ModelSerializer.load(Path('model.pkl'))

# Информация о модели
info = ModelSerializer.get_model_info(Path('model.pkl'))
```

### Sanity Checks

Проверки перед обучением:

```python
from src.modeling import ModelSanityChecker

checker = ModelSanityChecker()
result = checker.check_all(X_train, y_train, X_val, y_val)

if not result.passed:
    print(result.summary())
```

## Утилиты

```python
from src.modeling import (
    set_seed,
    get_device,
    ensure_reproducibility,
    log_system_info
)

# Воспроизводимость
set_seed(42)
ensure_reproducibility(42)

# Управление устройствами
device = get_device()  # Автоопределение CPU/GPU
log_system_info()      # Информация о системе
```

## Loss Functions

### Classification

- `bce` - Binary Cross Entropy
- `bce_logits` - BCE with Logits (numerically stable)
- `weighted_bce` - Weighted BCE
- `focal` - Focal Loss
- `focal_multiclass` - Multi-class Focal Loss

### Regression

- `mse` - Mean Squared Error
- `mae` - Mean Absolute Error (L1)
- `rmse` - Root Mean Squared Error
- `huber` - Huber Loss
- `quantile` - Quantile Loss
- `logcosh` - Log-Cosh Loss

### Trading Custom

- `directional` - Directional Loss (оптимизация направления)
- `sign` - Sign Loss
- `asymmetric_directional` - Asymmetric Directional Loss
- `profit` - Profit-based Loss
- `sharpe` - Sharpe Ratio Loss
- `expected_pnl` - Expected PnL Loss
- `risk_adjusted_profit` - Risk-adjusted Profit Loss

## Примеры использования

### Полный пример обучения

```python
from src.modeling import (
    ModelTrainer,
    DataSplitter,
    ModelSanityChecker,
    EarlyStopping,
    ModelCheckpoint,
    set_seed
)
from src.modeling.loss_functions import LossRegistry

# Воспроизводимость
set_seed(42)

# Разделение данных
train, val, test = DataSplitter.split_sequential(data)

X_train, y_train = train.drop('target', axis=1), train['target']
X_val, y_val = val.drop('target', axis=1), val['target']

# Sanity checks
checker = ModelSanityChecker()
check_result = checker.check_all(X_train, y_train, X_val, y_val)

if not check_result.passed:
    print("⚠️ Проблемы с данными:")
    print(check_result.summary())

# Создание модели (будет реализовано в Этапе 07)
# model = ModelRegistry.create('lightgbm_classifier')

# Обучение
trainer = ModelTrainer(model, experiment_name='exp_001')

result = trainer.train(
    X_train, y_train,
    X_val, y_val,
    callbacks=[
        EarlyStopping(patience=10),
        ModelCheckpoint('models/best.pt'),
    ]
)

print(f"Обучение завершено: {result}")
```

## Тесты

Запуск тестов:

```bash
pytest tests/unit/modeling/ -v
```

## Следующие шаги

- **Этап 07**: Реализация классических ML моделей (LightGBM, XGBoost, etc.)
- **Этап 08**: Реализация нейросетевых моделей
- **Этап 09**: Система обучения и оптимизации гиперпараметров
