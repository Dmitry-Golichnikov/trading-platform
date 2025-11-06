# Модели машинного обучения

Этот модуль содержит реализации всех моделей машинного обучения для торговой платформы.

## Обзор

Все модели наследуются от `BaseModel` и предоставляют единый API:
- `fit(X, y, X_val, y_val)` - обучение модели
- `predict(X)` - предсказание
- `predict_proba(X)` - вероятности классов (для классификации)
- `save(path)` - сохранение модели
- `load(path)` - загрузка модели
- `feature_importances_` - важности признаков

## Категории моделей

### 1. Tree-based модели

Основаны на деревьях решений и gradient boosting.

#### LightGBM (`lightgbm`)
**Приоритет #1** - Основная модель для большинства задач.

**Особенности:**
- Быстрое обучение и предсказание
- GPU support
- Early stopping
- Feature importances (gain/split)

**Пример использования:**
```python
from src.modeling.registry import ModelRegistry

model = ModelRegistry.create(
    "lightgbm",
    task="classification",
    n_estimators=1000,
    learning_rate=0.05,
    device="cpu"
)

model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
```

**Конфигурация:** `configs/models/lightgbm_default.yaml`

---

#### XGBoost (`xgboost`)
**Особенности:**
- Мощный gradient boosting
- GPU support
- Различные objectives
- Регуляризация (L1/L2)

**Пример использования:**
```python
model = ModelRegistry.create(
    "xgboost",
    task="classification",
    max_depth=6,
    n_estimators=1000
)
```

**Конфигурация:** `configs/models/xgboost_default.yaml`

---

#### CatBoost (`catboost`)
**Особенности:**
- Нативная поддержка категориальных признаков
- Ordered boosting
- GPU support
- Меньше переобучение

**Пример использования:**
```python
model = ModelRegistry.create(
    "catboost",
    task="classification",
    iterations=1000,
    cat_features=["ticker", "day_of_week"]
)
```

**Конфигурация:** `configs/models/catboost_default.yaml`

---

#### RandomForest (`random_forest`)
**Особенности:**
- Простой и надёжный
- Parallel training (n_jobs=-1)
- Устойчив к переобучению
- Не требует тонкой настройки

**Пример использования:**
```python
model = ModelRegistry.create(
    "random_forest",
    task="classification",
    n_estimators=100,
    max_depth=None
)
```

**Конфигурация:** `configs/models/random_forest_default.yaml`

---

#### ExtraTrees (`extra_trees`)
**Особенности:**
- Более случайный чем RandomForest
- Быстрее обучается
- Меньше variance, больше bias

**Пример использования:**
```python
model = ModelRegistry.create(
    "extra_trees",
    task="classification",
    n_estimators=100
)
```

**Конфигурация:** `configs/models/extra_trees_default.yaml`

---

### 2. Linear модели

Линейные модели для базовых сценариев.

#### LogisticRegression (`logistic_regression`)
**Особенности:**
- Быстрое обучение
- Интерпретируемость (коэффициенты)
- Различные penalty (L1, L2, ElasticNet)
- Baseline модель

**Пример использования:**
```python
model = ModelRegistry.create(
    "logistic_regression",
    penalty="l2",
    C=1.0,
    max_iter=1000
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Коэффициенты
importance_df = model.get_feature_importance_df()
```

**Конфигурация:** `configs/models/logistic_regression_default.yaml`

---

#### ElasticNet (`elasticnet`)
**Особенности:**
- Комбинация L1 и L2 регуляризации
- Feature selection (L1 обнуляет коэффициенты)
- Устойчив к мультиколлинеарности
- Только регрессия

**Пример использования:**
```python
model = ModelRegistry.create(
    "elasticnet",
    alpha=1.0,
    l1_ratio=0.5  # 0=Ridge, 1=Lasso, 0.5=ElasticNet
)

model.fit(X_train, y_train)

# Ненулевые признаки
nonzero_features = model.get_nonzero_features()
```

**Конфигурация:** `configs/models/elasticnet_default.yaml`

---

### 3. Tabular Neural Networks

Нейросетевые модели для табличных данных.

#### TabNet (`tabnet`)
**Особенности:**
- Attention-based architecture
- Sequential feature selection
- Feature importances через attention masks
- GPU support
- Интерпретируемость

**Использует:** `pytorch-tabnet` библиотеку

**Пример использования:**
```python
model = ModelRegistry.create(
    "tabnet",
    task="classification",
    n_d=8,
    n_a=8,
    n_steps=3,
    device="cpu"
)

model.fit(
    X_train, y_train, X_val, y_val,
    max_epochs=100,
    batch_size=1024
)
```

**Конфигурация:** `configs/models/tabnet_default.yaml`

---

#### FT-Transformer (`ft_transformer`)
**Особенности:**
- Feature Tokenizer + Transformer encoder
- Attention mechanism для признаков
- Современная архитектура
- GPU support

**Реализация:** Custom PyTorch implementation

**Пример использования:**
```python
model = ModelRegistry.create(
    "ft_transformer",
    task="classification",
    d_token=192,
    n_blocks=3,
    attention_heads=8,
    device="cuda"
)

model.fit(X_train, y_train, X_val, y_val)
```

**Конфигурация:** `configs/models/ft_transformer_default.yaml`

---

#### NODE (`node`)
**Особенности:**
- Neural Oblivious Decision Ensembles
- Дифференцируемые деревья решений
- Ансамбль обливиусных деревьев
- GPU support

**Реализация:** Simplified PyTorch implementation

**Пример использования:**
```python
model = ModelRegistry.create(
    "node",
    task="classification",
    num_layers=4,
    num_trees=2048,
    depth=6,
    device="cpu"
)

model.fit(X_train, y_train, X_val, y_val)
```

**Конфигурация:** `configs/models/node_default.yaml`

---

## Выбор модели

### Для классификации:

**Лучший выбор (по приоритету):**
1. **LightGBM** - универсальный, быстрый, точный
2. **CatBoost** - если есть категориальные признаки
3. **XGBoost** - для сложных задач
4. **TabNet** - если нужна интерпретируемость
5. **RandomForest** - для baseline

**Когда использовать:**
- **LightGBM/XGBoost/CatBoost** - основные рабочие лошадки
- **RandomForest/ExtraTrees** - быстрый baseline, устойчивость
- **LogisticRegression** - простой baseline, быстрое прототипирование
- **TabNet/FT-Transformer/NODE** - когда tree-based не работают

### Для регрессии:

**Лучший выбор:**
1. **LightGBM** - универсальный
2. **XGBoost** - для сложных задач
3. **RandomForest** - baseline
4. **ElasticNet** - линейные зависимости

---

## GPU Support

**Модели с GPU support:**
- LightGBM: `device="gpu"`
- XGBoost: `tree_method="gpu_hist"`
- CatBoost: `task_type="GPU"`
- TabNet: `device="cuda"`
- FT-Transformer: `device="cuda"`
- NODE: `device="cuda"`

**Пример:**
```python
# LightGBM
model = ModelRegistry.create("lightgbm", device="gpu")

# XGBoost
model = ModelRegistry.create("xgboost", tree_method="gpu_hist")

# CatBoost
model = ModelRegistry.create("catboost", task_type="GPU")

# Neural networks
model = ModelRegistry.create("tabnet", device="cuda")
```

---

## Feature Importances

Все модели предоставляют `feature_importances_`:

```python
# Получить массив важностей
importances = model.feature_importances_

# Получить DataFrame с важностями
importance_df = model.get_feature_importance_df()
print(importance_df.head(10))  # топ-10 признаков
```

**Типы важностей:**
- **Tree-based**: gain-based (сколько улучшения дают признаки)
- **Linear**: абсолютные значения коэффициентов
- **TabNet**: attention masks (сколько внимания уделяется признаку)
- **FT-Transformer/NODE**: не поддерживают

---

## Сохранение/Загрузка

Все модели поддерживают save/load:

```python
# Сохранение
model.save(Path("artifacts/models/my_model"))

# Загрузка
from src.modeling.models import LightGBMModel
loaded_model = LightGBMModel.load(Path("artifacts/models/my_model"))
```

**Сохраняется:**
- Веса модели
- Гиперпараметры
- Метаданные (время обучения, количество примеров и т.д.)
- Имена признаков
- Классы (для классификации)

---

## Model Registry

Все модели регистрируются в `ModelRegistry`:

```python
from src.modeling.registry import ModelRegistry

# Список всех моделей
all_models = ModelRegistry.list_models()
print(all_models)  # ['lightgbm', 'xgboost', ...]

# Фильтр по тегам
tree_models = ModelRegistry.list_models(tags=["tree-based"])

# Информация о модели
metadata = ModelRegistry.get_metadata("lightgbm")
print(metadata)

# Создание модели
model = ModelRegistry.create("lightgbm", task="classification")
```

---

## Тестирование

**Unit тесты:** `tests/unit/test_models.py`
```bash
pytest tests/unit/test_models.py -v
```

**Интеграционные тесты:** `tests/integration/test_all_models_integration.py`
```bash
pytest tests/integration/test_all_models_integration.py -v
```

---

## Требования

**Обязательные:**
- pandas
- numpy
- scikit-learn

**Tree-based:**
- lightgbm
- xgboost
- catboost

**Tabular NN:**
- torch
- pytorch-tabnet (для TabNet)

**Установка всех зависимостей:**
```bash
pip install -r requirements.txt
```

---

## Roadmap

- [ ] Добавить ONNX export для production
- [ ] Добавить quantization для ускорения inference
- [ ] Добавить ensemble методы (stacking, blending)
- [ ] Добавить AutoML wrapper
- [ ] Добавить pruning для neural networks

---

**Создано:** 2025-11-06
**Версия:** 1.0
**Статус:** ✅ Готово к использованию
