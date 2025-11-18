# Отбор и трансформация признаков

## Назначение
Сокращение размерности и удаление нерелевантных признаков, улучшение интерпретируемости и производительности моделей.

## Методы отбора
- **Filter methods**: корреляция, mutual information, chi-square, ANOVA.
- **Wrapper methods**: рекурсивный отбор (RFE), последовательный прямой/обратный отбор.
- **Embedded methods**: L1-регуляризация, feature importance в GBDT/RandomForest, SHAP-based.
- **Drift detection**: PSI, KL divergence для исключения drifted features.

## Методы трансформации
- **PCA/ICA/UMAP**: снижение размерности.
- **Autoencoder**: нелинейное сжатие.
- **Feature grouping/aggregation**: средние/максимумы по группам, создание метафичей.

## Конфигурация
- `selection_method`: список методов с параметрами.
- `top_k`/`threshold`: критерии отбора.
- `dim_reduction`: настройки PCA/UMAP (components, neighbors).
- `drift_threshold`: порог для PSI/kl.

## Артефакты
- Список выбранных признаков (`selected_features.json`).
- Модели трансформаций (PCA components, autoencoder weights).
- Логи с метриками отбора (importance, mutual info).

## Рекомендации
- Валидация: проверять качества моделей до/после отбора.
- Отдельно отслеживать группы признаков (технические, календарные) и их вклад.
- Обновлять отбор при появлении новых фичей или изменений в данных.
