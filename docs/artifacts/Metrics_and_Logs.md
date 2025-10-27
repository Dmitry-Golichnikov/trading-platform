# Метрики и логи

## Назначение
Фиксация метрик обучения, валидации и бэктестов, а также логов процесса обучения и инференса.

## Форматы
- MLflow metrics (временные ряды значений).
- JSON/CSV отчёты (сводные метрики, таблицы).
- TensorBoard event files (`.tfevents`).
- Логи (text/JSON) из тренеровщков и пайплайнов.

## Содержимое
- Train/validation/test метрики (Accuracy, F1, ROC-AUC, MSE, Sharpe и т.д.).
- Confusion matrices, calibration curves.
- Feature importance, SHAP values.
- Диагностика: время эпох, LR, значения loss.

## Хранение
- MLflow Tracking Server.
- Локальные каталоги (`logs/`, `reports/`).
- Системы мониторинга (Prometheus, Grafana, ELK/EFK).

## Рекомендации
- Логировать метрики по шагам и по эпохам, хранить историю.
- Создавать автоматические отчёты с визуализациями.
- Привязывать метрики к `run_id` и конфигам для воспроизводимости.
