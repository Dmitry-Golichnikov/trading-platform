# Quick Start: Модуль оценки моделей

Быстрое руководство по использованию модуля evaluation.

## Установка

```bash
# Основные зависимости уже установлены
# Для SHAP (опционально):
pip install shap

# Для визуализаций (опционально):
pip install plotly
```

## 5-минутный старт

### Шаг 1: Подготовка данных и модели

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Генерируем данные
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Обучаем модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### Шаг 2: Быстрая оценка

```python
from src.evaluation import MetricsCalculator

# Предсказания
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Все метрики одной строкой
metrics = MetricsCalculator.compute_metrics(
    y_test,
    y_pred,
    task_type='classification',
    y_proba=y_proba
)

# Выводим основные метрики
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
print(f"F1-score: {metrics['f1']:.3f}")
```

### Шаг 3: Полный отчёт

```python
from src.evaluation import ModelEvaluationReport

# Генерируем отчёт
report = ModelEvaluationReport(
    model=model,
    task_type='classification'
)

report_data = report.generate(
    X_test=X_test,
    y_test=y_test,
    X_train=X_train,
    y_train=y_train,
    output_path='model_report.html'
)

print("✓ Отчёт сохранён в model_report.html")
```

## Частые сценарии

### Сценарий 1: Проверка калибровки

```python
from src.evaluation import CalibrationMetrics

y_proba_pos = y_proba[:, 1]  # Вероятности положительного класса

# Вычисляем метрики калибровки
cal_metrics = CalibrationMetrics.compute_all(y_test, y_proba_pos)

print(f"Brier Score: {cal_metrics['brier_score']:.4f}")
print(f"ECE: {cal_metrics['ece']:.4f}")

# Если ECE > 0.1, модель плохо калибрована
if cal_metrics['ece'] > 0.1:
    print("⚠ Модель требует калибровки!")
```

### Сценарий 2: Топ важных признаков

```python
from src.evaluation import FeatureImportanceAnalyzer

feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

analyzer = FeatureImportanceAnalyzer(
    model=model,
    X=X_test,
    y=y_test,
    feature_names=feature_names
)

# Топ-10 признаков
top_features = analyzer.get_top_features(n_top=10)
print("Топ-10 признаков:")
for i, feat in enumerate(top_features, 1):
    print(f"  {i}. {feat}")
```

### Сценарий 3: Детекция дрейфа

```python
import pandas as pd
from src.evaluation import DriftDetector

# Преобразуем в DataFrame
df_train = pd.DataFrame(X_train, columns=feature_names)
df_test = pd.DataFrame(X_test, columns=feature_names)

# Детектор
detector = DriftDetector(df_train, df_test)

# Быстрая проверка PSI
psi_results = detector.detect_all(methods=['psi'])['psi']

# Признаки с дрейфом (PSI > 0.2)
drifted = psi_results[psi_results['psi'] > 0.2]
if not drifted.empty:
    print("⚠ Обнаружен дрейф в признаках:")
    print(drifted[['feature', 'psi', 'status']])
else:
    print("✓ Дрейфа не обнаружено")
```

## CLI: Оценка из терминала

### Быстрая оценка модели

```bash
# Сохраните модель
python -c "
from sklearn.ensemble import RandomForestClassifier
import joblib
# ... обучение ...
joblib.dump(model, 'my_model.pkl')
"

# Оцените через CLI
python -m src.interfaces.cli evaluate model \
    --model-path my_model.pkl \
    --test-data test_data.parquet \
    --output-dir reports/
```

### Важность признаков

```bash
python -m src.interfaces.cli evaluate importance \
    --model-path my_model.pkl \
    --data test_data.parquet \
    --top-n 10
```

### Детекция дрейфа

```bash
python -m src.interfaces.cli evaluate drift \
    --reference-data train.parquet \
    --current-data test.parquet
```

## Интеграция с пайплайном обучения

```python
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Полный пайплайн обучения и оценки."""
    from src.evaluation import (
        MetricsCalculator,
        ModelCalibrator,
        ModelEvaluationReport
    )

    # 1. Обучение
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 2. Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # 3. Метрики
    metrics = MetricsCalculator.compute_metrics(
        y_test, y_pred,
        task_type='classification',
        y_proba=y_proba
    )

    # 4. Калибровка (если нужно)
    cal_metrics = CalibrationMetrics.compute_all(y_test, y_proba[:, 1])
    if cal_metrics['ece'] > 0.1:
        calibrator = ModelCalibrator(method='isotonic')
        calibrator.fit(y_proba[:, 1], y_test)
        # Сохраняем калибратор вместе с моделью

    # 5. Генерация отчёта
    report = ModelEvaluationReport(model, task_type='classification')
    report.generate(
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
        output_path='model_report.html'
    )

    return model, metrics

# Использование
model, metrics = train_and_evaluate(X_train, X_test, y_train, y_test)
print(f"Model ROC-AUC: {metrics['roc_auc']:.3f}")
```

## Работа с регрессией

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from src.evaluation import RegressionMetrics

# Данные
X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Модель
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
metrics = RegressionMetrics.compute_all(y_test, y_pred, n_features=10)

print(f"RMSE: {metrics['rmse']:.3f}")
print(f"MAE: {metrics['mae']:.3f}")
print(f"R²: {metrics['r2']:.3f}")
```

## Полезные паттерны

### Паттерн 1: Сравнение моделей

```python
from src.evaluation import MetricsCalculator
import pandas as pd

models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42),
    # ... другие модели
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    metrics = MetricsCalculator.compute_metrics(
        y_test, y_pred,
        task_type='classification',
        y_proba=y_proba
    )

    results.append({
        'model': name,
        'accuracy': metrics['accuracy'],
        'roc_auc': metrics.get('roc_auc', None),
        'f1': metrics['f1']
    })

df_results = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
print(df_results)
```

### Паттерн 2: Мониторинг в продакшене

```python
from src.evaluation import DriftDetector, PopulationStabilityIndex
import pandas as pd

def monitor_production_data(train_data, prod_data, threshold=0.2):
    """Мониторинг дрейфа в продакшене."""

    detector = DriftDetector(train_data, prod_data)

    # PSI для всех признаков
    psi_results = detector.detect_all(methods=['psi'])['psi']

    # Алерты
    critical_drift = psi_results[psi_results['psi'] > threshold]

    if not critical_drift.empty:
        print("🚨 АЛЕРТ: Обнаружен критический дрейф!")
        print(critical_drift[['feature', 'psi', 'status']])
        return False
    else:
        print("✓ Данные стабильны")
        return True

# Использование
is_stable = monitor_production_data(df_train, df_production, threshold=0.2)
if not is_stable:
    # Отправить алерт, переобучить модель, etc.
    pass
```

### Паттерн 3: Автоматическая калибровка

```python
from src.evaluation import ModelCalibrator, compare_calibration_methods

def auto_calibrate(model, X_cal, y_cal):
    """Автоматический выбор лучшего метода калибровки."""

    # Получаем вероятности
    y_proba = model.predict_proba(X_cal)[:, 1]

    # Сравниваем методы
    comparison = compare_calibration_methods(y_proba, y_cal)

    # Выбираем лучший по Brier Score
    best_method = min(
        [m for m in comparison if m != 'uncalibrated'],
        key=lambda m: comparison[m].get('brier_score', float('inf'))
    )

    print(f"Лучший метод калибровки: {best_method}")

    # Обучаем лучший калибратор
    calibrator = ModelCalibrator(method=best_method)
    calibrator.fit(y_proba, y_cal)

    return calibrator

# Использование
calibrator = auto_calibrate(model, X_test, y_test)
```

## Советы и трюки

### 1. Экономия памяти для больших данных

```python
# Используйте подвыборку для SHAP
from src.evaluation import SHAPImportance

shap_df = SHAPImportance.compute(
    model, X_test,
    max_samples=100  # Только 100 сэмплов
)
```

### 2. Параллельное вычисление метрик

```python
from joblib import Parallel, delayed

def compute_metrics_parallel(models, X_test, y_test):
    def evaluate_model(name, model):
        y_pred = model.predict(X_test)
        metrics = MetricsCalculator.compute_metrics(y_test, y_pred)
        return name, metrics

    results = Parallel(n_jobs=-1)(
        delayed(evaluate_model)(name, model)
        for name, model in models.items()
    )

    return dict(results)
```

### 3. Кэширование для ускорения

```python
from functools import lru_cache
import hashlib
import pickle

@lru_cache(maxsize=32)
def cached_drift_detection(train_hash, test_hash):
    # Детекция дрейфа с кэшированием
    pass
```

## Что дальше?

- 📖 [Полная документация](README.md)
- 📊 [Примеры отчётов](../../examples/evaluation/)
- 🧪 [Юнит-тесты](../../tests/unit/test_evaluation_*.py)
- 🔧 [Расширение модуля](README.md#расширение-модуля)

## Поддержка

Вопросы? Создайте issue или посмотрите FAQ в основной документации.
