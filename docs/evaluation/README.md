# Модуль оценки моделей (Evaluation)

Модуль `src/evaluation` предоставляет комплексный набор инструментов для оценки качества моделей машинного обучения.

## Компоненты модуля

### 1. Метрики (metrics.py)

Вычисление метрик качества для задач классификации и регрессии.

#### Классификация

**Основные метрики:**
- **Accuracy** - доля правильных предсказаний
- **Balanced Accuracy** - средняя чувствительность по классам
- **Precision** - точность положительных предсказаний
- **Recall** - полнота (чувствительность)
- **F1-score** - гармоническое среднее Precision и Recall
- **Matthews Correlation Coefficient (MCC)** - корреляция между предсказаниями и реальностью
- **ROC-AUC** - площадь под ROC-кривой
- **PR-AUC** - площадь под Precision-Recall кривой
- **Confusion Matrix** - матрица ошибок

#### Регрессия

**Основные метрики:**
- **MSE** - средняя квадратичная ошибка
- **RMSE** - корень из MSE
- **MAE** - средняя абсолютная ошибка
- **MAPE** - средняя абсолютная процентная ошибка
- **R²** - коэффициент детерминации
- **Adjusted R²** - скорректированный R²
- **Quantile Loss** - функция потерь для квантильной регрессии

### 2. Калибровка (calibration.py)

Калибровка вероятностей для улучшения надежности предсказаний.

**Методы калибровки:**
- **Isotonic Regression** - монотонная калибровка
- **Platt Scaling** - логистическая калибровка
- **Beta Calibration** - параметрическая калибровка

**Метрики калибровки:**
- **Brier Score** - средний квадрат разности между вероятностями и реальностью
- **Expected Calibration Error (ECE)** - средневзвешенная разница между уверенностью и точностью
- **Maximum Calibration Error (MCE)** - максимальная разница в любом bin

### 3. Важность признаков (feature_importance.py)

Анализ влияния признаков на предсказания модели.

**Методы:**
- **Tree-based Importance** - важность из древовидных моделей (gain, split)
- **Permutation Importance** - важность через перемешивание
- **SHAP Values** - объяснения на основе теории игр
- **Partial Dependence** - зависимость предсказаний от признака

### 4. Детекция дрейфа (drift_detection.py)

Обнаружение изменений в распределении данных между обучением и продакшеном.

**Методы:**
- **Population Stability Index (PSI)** - индекс стабильности популяции
- **Kolmogorov-Smirnov Test** - статистический тест на равенство распределений
- **Chi-squared Test** - тест для категориальных переменных
- **Adversarial Validation** - обучение классификатора отличать train от test

### 5. Отчёты (reports.py)

Генерация комплексных HTML-отчётов с визуализациями.

**Содержание отчёта:**
- Все метрики модели
- Анализ калибровки (для классификации)
- Важность признаков
- Детекция дрейфа
- Интерактивные графики (Plotly)

## Примеры использования

### Пример 1: Оценка модели классификации

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from src.evaluation import ClassificationMetrics

# Генерируем данные
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Обучаем модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказания
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Вычисляем все метрики
metrics = ClassificationMetrics.compute_all(y_test, y_pred, y_proba)

print("Accuracy:", metrics['accuracy'])
print("ROC-AUC:", metrics['roc_auc'])
print("F1-score:", metrics['f1'])
```

### Пример 2: Калибровка модели

```python
from src.evaluation import ModelCalibrator, CalibrationMetrics

# Калибруем модель
calibrator = ModelCalibrator(method='isotonic')
y_proba_pos = y_proba[:, 1]
calibrator.fit(y_proba_pos, y_test)
y_proba_calibrated = calibrator.transform(y_proba_pos)

# Сравниваем метрики
metrics_before = CalibrationMetrics.compute_all(y_test, y_proba_pos)
metrics_after = CalibrationMetrics.compute_all(y_test, y_proba_calibrated)

print("Brier Score до:", metrics_before['brier_score'])
print("Brier Score после:", metrics_after['brier_score'])
print("ECE до:", metrics_before['ece'])
print("ECE после:", metrics_after['ece'])
```

### Пример 3: Анализ важности признаков

```python
from src.evaluation import FeatureImportanceAnalyzer

# Создаем анализатор
feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
analyzer = FeatureImportanceAnalyzer(
    model=model,
    X=X_test,
    y=y_test,
    feature_names=feature_names
)

# Вычисляем важность разными методами
importances = analyzer.compute_all_importances(
    methods=['tree', 'permutation'],
    n_repeats=10
)

# Получаем топ-10 признаков
top_features = analyzer.get_top_features(n_top=10, method='tree')
print("Топ-10 признаков:", top_features)

# Сравниваем методы
comparison = analyzer.compare_methods()
print(comparison)
```

### Пример 4: Детекция дрейфа

```python
import pandas as pd
from src.evaluation import DriftDetector

# Предположим у нас есть train и test данные
df_train = pd.DataFrame(X_train, columns=feature_names)
df_test = pd.DataFrame(X_test, columns=feature_names)

# Создаем детектор
detector = DriftDetector(df_train, df_test)

# Детекция всеми методами
drift_results = detector.detect_all(methods=['psi', 'ks', 'adversarial'])

# PSI результаты
print("\nPSI по признакам:")
print(drift_results['psi'].head(10))

# Сводка
summary = detector.summary()
print("\nСводка дрейфа:")
for key, value in summary.items():
    print(f"{key}: {value}")

# Признаки с дрейфом
drifted_features = detector.get_drifted_features(method='psi', threshold=0.2)
print(f"\nПризнаки с дрейфом: {drifted_features}")
```

### Пример 5: Генерация полного отчёта

```python
from src.evaluation import ModelEvaluationReport
from pathlib import Path

# Создаем генератор отчёта
report = ModelEvaluationReport(
    model=model,
    task_type='classification',
    model_name='RandomForest_v1'
)

# Генерируем отчёт
report_data = report.generate(
    X_test=X_test,
    y_test=y_test,
    X_train=X_train,
    y_train=y_train,
    feature_names=feature_names,
    output_path='artifacts/reports/model_evaluation.html',
    include_sections=['metrics', 'calibration', 'importance', 'drift']
)

# Сохраняем также в JSON
report.save_json('artifacts/reports/model_evaluation.json')

print("Отчёт сгенерирован!")
```

## CLI интерфейс

Модуль предоставляет удобные команды для работы через терминал:

### Оценка модели

```bash
python -m src.interfaces.cli evaluate model \
    --model-path artifacts/models/my_model.pkl \
    --test-data artifacts/data/test.parquet \
    --train-data artifacts/data/train.parquet \
    --task-type classification \
    --output-dir artifacts/reports \
    --sections "metrics,calibration,importance,drift"
```

### Анализ важности признаков

```bash
python -m src.interfaces.cli evaluate importance \
    --model-path artifacts/models/my_model.pkl \
    --data artifacts/data/test.parquet \
    --methods "tree,permutation,shap" \
    --top-n 20 \
    --output artifacts/reports/importance.csv
```

### Калибровка модели

```bash
python -m src.interfaces.cli evaluate calibrate \
    --model-path artifacts/models/my_model.pkl \
    --data artifacts/data/calibration.parquet \
    --method isotonic \
    --output artifacts/models/calibrator.pkl
```

### Детекция дрейфа

```bash
python -m src.interfaces.cli evaluate drift \
    --reference-data artifacts/data/train.parquet \
    --current-data artifacts/data/test.parquet \
    --methods "psi,ks,adversarial" \
    --output artifacts/reports/drift_analysis.json
```

## Интеграция с MLflow

Все метрики можно автоматически логировать в MLflow:

```python
import mlflow
from src.evaluation import MetricsCalculator

# Начинаем эксперимент
with mlflow.start_run():
    # Обучение модели
    model.fit(X_train, y_train)

    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    # Вычисляем метрики
    metrics = MetricsCalculator.compute_metrics(
        y_test, y_pred,
        task_type='classification',
        y_proba=y_proba
    )

    # Логируем в MLflow
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(metric_name, value)

    # Сохраняем модель
    mlflow.sklearn.log_model(model, "model")
```

## Best Practices

### 1. Выбор метрик

- **Несбалансированные классы:** используйте Balanced Accuracy, F1-score, PR-AUC вместо Accuracy
- **Финансовые приложения:** важен MCC, калибровка вероятностей
- **Регрессия с выбросами:** используйте MAE вместо MSE

### 2. Калибровка

- Isotonic Regression лучше для больших датасетов
- Platt Scaling подходит для малых данных
- Всегда проверяйте ECE после калибровки

### 3. Важность признаков

- Используйте несколько методов для надежности
- Tree-based importance быстрый, но может быть смещенным
- Permutation importance медленнее, но более надежный
- SHAP дает интерпретируемые объяснения

### 4. Детекция дрейфа

- PSI удобен для мониторинга в продакшене (быстрый)
- KS test дает статистическую значимость
- Adversarial validation полезен для понимания природы дрейфа
- Мониторьте дрейф регулярно (еженедельно/ежемесячно)

### 5. Отчёты

- Генерируйте отчёты после каждого обучения
- Сохраняйте JSON версии для программного доступа
- Включайте drift detection при сравнении с production данными

## Расширение модуля

### Добавление новой метрики

```python
# В src/evaluation/metrics.py
class CustomMetrics:
    @staticmethod
    def my_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Моя кастомная метрика."""
        # Ваша реализация
        return result
```

### Добавление метода калибровки

```python
# В src/evaluation/calibration.py
class MyCalibrator:
    def fit(self, y_proba, y_true):
        # Ваша реализация
        pass

    def transform(self, y_proba):
        # Ваша реализация
        return calibrated_proba
```

## Зависимости

- `numpy` - работа с массивами
- `pandas` - работа с табличными данными
- `scikit-learn` - базовые метрики и алгоритмы
- `scipy` - статистические тесты
- `plotly` - интерактивные визуализации (опционально)
- `shap` - SHAP values (опционально)

## Тестирование

Запуск тестов модуля:

```bash
pytest tests/unit/test_evaluation_*.py -v
```

## Дополнительные ресурсы

- [Документация по метрикам](../metrics/models/)
- [Примеры использования](../../examples/)
- [API документация](../../api/evaluation/)

## Поддержка

При возникновении проблем:
1. Проверьте версии зависимостей
2. Убедитесь что данные в правильном формате
3. Посмотрите примеры использования
4. Создайте issue в репозитории
