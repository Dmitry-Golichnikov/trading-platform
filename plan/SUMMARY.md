# Краткое описание всех этапов

## Фаза 1: Фундамент (2-3 недели)

**Этап 00: Инициализация проекта**
- Создание структуры проекта
- Настройка инструментов разработки (black, mypy, flake8)
- Docker контейнеризация
- Базовые утилиты (config, logging, validation)

**Этап 01: Модуль данных**
- Загрузчики данных (локальные файлы, заглушка для Tinkoff API)
- Хранилище Parquet
- Каталогизация и версионирование датасетов
- Ресэмплинг таймфреймов
- CLI команды для работы с данными

**Этап 02: Обработка и валидация данных**
- Фильтры данных (аномалии, ликвидность, события, выбросы)
- Обработка пропусков и дубликатов
- Исправление ошибок в данных
- Метрики качества данных
- Quality reports (HTML)

## Фаза 2: Признаки и таргеты (2-3 недели)

**Этап 03: Библиотека индикаторов**
- 30+ технических индикаторов (все каузальные)
- Категории: трендовые, моментум, волатильность, объёмные, продвинутые
- Единый API для всех индикаторов
- Реестр индикаторов
- Валидация каузальности

**Этап 04: Генерация признаков**
- FeatureGenerator с YAML конфигами
- Extractors (price, volume, calendar, ticker, higher TF)
- Transformers (rolling, lags, differences, ratios)
- Feature selection методы
- Кэширование и версионирование признаков

**Этап 05: Разметка таргетов**
- Horizon labeling (fixed/adaptive)
- Triple barrier method
- Regression targets
- Постфильтры (smoothing, sequence filters)
- Балансировка классов
- Метаданные разметки

## Фаза 3: Моделирование (4-5 недель)

**Этап 06: Базовая инфраструктура моделирования**
- BaseModel единый интерфейс
- ModelRegistry
- ModelTrainer с MLflow integration
- Callbacks (EarlyStopping, ModelCheckpoint)
- Loss functions registry
- Data splitting методы
- Sanity checks перед обучением

**Этап 07: Классические ML модели**
- Tree-based: LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees
- Linear: LogisticRegression, ElasticNet
- Tabular NN: TabNet, FT-Transformer, NODE
- Дефолтные конфиги для всех моделей
- GPU support

**Этап 08: Нейросетевые модели**
- Sequential models: LSTM, GRU, Seq2Seq with Attention
- TCN, Temporal Fusion Transformer, Informer
- CNN+LSTM hybrid
- Sequence preparation
- Training utilities (schedulers, augmentation)

**Этап 09: Система обучения и оптимизации**
- Hyperparameter search (Grid, Random, Bayesian, Optuna)
- Multi-level optimization
- Threshold optimization по E[PnL]
- AutoML pipeline
- Meta-learning для warm-start
- Experiment tracking

**Этап 10: Оценка моделей**
- Все метрики классификации и регрессии
- Model calibration
- Feature importance (tree-based, permutation, SHAP)
- Drift detection (PSI, KS test)
- Evaluation reports (HTML)

## Фаза 4: Бэктестинг и пайплайны (3-4 недели)

**Этап 11: Движок бэктестинга**
- Event-driven backtesting engine
- BaseStrategy интерфейс
- Все exit rules (TP, SL, trailing stop, time stop, etc)
- Position и portfolio management
- Учёт комиссий и slippage
- Strategy metrics (Sharpe, Sortino, Drawdown, etc)
- Визуализация (equity curves, trades)

**Этап 12: Система пайплайнов**
- Все пайплайны (Data, Features, Training, Validation, Backtest)
- FullPipeline end-to-end
- Чекпоинты и кэширование
- Идемпотентность
- Конфигурация через YAML

**Этап 13: Оркестрация и управление экспериментами**
- ExperimentManager
- Task scheduler (cron-like)
- MLflow integration (полная)
- Monitoring hooks
- Сравнение экспериментов

## Фаза 5: Интерфейсы (3-4 недели)

**Этап 14: CLI интерфейс**
- Полный набор команд для всех операций
- Группы: data, features, labels, models, hyperopt, backtest, experiments, pipelines
- Progress bars, colored output, tables
- Autocomplete для bash/zsh
- Help и документация

**Этап 15: GUI интерфейс**
- Web-based GUI (FastAPI + React)
- Dashboard, data management, model training, backtesting
- Real-time updates через WebSocket
- Interactive charts (Plotly)
- Preset management
- Dark/Light theme

**Этап 16: Интеграция Tinkoff API**
- TinkoffAPIClient с rate limiting
- TinkoffDataLoader
- Загрузка исторических данных (годовые архивы для 1m)
- Автоматический resample в higher timeframes
- Поиск инструментов
- Retry logic и error handling

## Фаза 6: Инфраструктура и завершение (2-3 недели)

**Этап 17: Мониторинг и логирование**
- Structured logging (JSON)
- Log rotation и retention
- Metrics collection (system + application)
- Health checks
- Alerting система
- Prometheus + Grafana integration
- PII masking

**Этап 18: Распределённые вычисления**
- Task queue (Redis)
- Worker для стационарного ПК с GPU
- Client для ноутбука
- Git sync service (автоматический pull)
- Artifact sync в shared storage
- Fallback to local execution
- Systemd services

**Этап 19: CI/CD и тестирование**
- GitHub Actions workflows (CI, tests, docker, docs, nightly)
- Unit, integration, regression, E2E tests
- Coverage tracking (Codecov)
- Docker images build & push
- Branch protection rules
- Security scanning

**Этап 20: Документация и финализация**
- User documentation (installation, guides, tutorials)
- API documentation (Sphinx)
- Tutorial notebooks
- Developer documentation
- FAQ и troubleshooting
- Final testing (E2E, performance, security)
- Release preparation (CHANGELOG, GitHub release, Docker Hub)
- Roadmap для будущих версий

---

## Общая оценка времени

**Минимальная оценка:** 16 недель (4 месяца)
**Реалистичная оценка:** 22 недели (5.5 месяцев)

## Ключевые технологии

**Backend:**
- Python 3.10+
- PyTorch, LightGBM, XGBoost, CatBoost
- Pandas, NumPy, scikit-learn
- FastAPI, Redis
- MLflow

**Frontend (GUI):**
- React, TypeScript
- Material-UI / Ant Design
- Plotly.js

**Infrastructure:**
- Docker, Docker Compose
- GitHub Actions
- Prometheus, Grafana
- Nginx, Gunicorn

**APIs:**
- Tinkoff Investments API

## Приоритеты для MVP

### Must-have (базовый функционал):
- Этапы 00-07: Инфраструктура, данные, признаки, базовые модели
- Этапы 11-14: Бэктестинг, пайплайны, CLI

### Should-have (полноценная версия):
- Этапы 08-10: Продвинутые модели, оптимизация, оценка
- Этапы 15-17: GUI, Tinkoff API, мониторинг

### Nice-to-have (масштабирование):
- Этапы 18-20: Распределённые вычисления, CI/CD, финализация

## Рекомендации по использованию плана

1. **Последовательность выполнения**: Следуйте порядку этапов, они выстроены с учётом зависимостей
2. **Проверка готовности**: После каждого этапа проверьте все критерии готовности
3. **Тестирование**: Не пропускайте написание тестов, это сэкономит время в будущем
4. **Документация**: Документируйте по ходу разработки, а не в конце
5. **Коммиты**: Регулярные осмысленные коммиты после завершения логических блоков
6. **Промпты**: Используйте готовые промпты из каждого этапа для работы с AI-ассистентом
7. **Итерации**: Не стремитесь к идеалу сразу, делайте итеративно
8. **Feedback**: Тестируйте функциональность по мере разработки

## Контакты и поддержка

- Основное ТЗ: `technical_spec.md`
- Документация системы: `docs/system/`
- План реализации: `план/`

---

**Создано:** 27 октября 2025
**Версия плана:** 1.0
**Статус:** Готов к реализации ✅
