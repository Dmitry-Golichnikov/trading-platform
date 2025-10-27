# Стандарты документации

## 1. Типы документации

### 1.1 Иерархия документов

```
docs/
├── README.md                    # Главная страница
├── CONTRIBUTING.md              # Руководство для контрибьюторов
├── CHANGELOG.md                 # История изменений
├── technical_spec.md            # Техническое задание
├── architecture/                # Архитектура системы
│   ├── overview.md
│   ├── data_flow.md
│   └── components.md
├── api/                         # API документация
│   ├── rest_api.md
│   └── python_api.md
├── guides/                      # Руководства пользователя
│   ├── quickstart.md
│   ├── data_preparation.md
│   ├── model_training.md
│   └── backtesting.md
├── tutorials/                   # Пошаговые туториалы
│   ├── first_model.md
│   ├── custom_indicators.md
│   └── optimization.md
├── reference/                   # Справочная документация
│   ├── indicators.md
│   ├── models.md
│   └── metrics.md
└── system/                      # Системная документация
    ├── deployment.md
    ├── testing.md
    └── monitoring.md
```

## 2. README.md

### 2.1 Структура

```markdown
# Trading Platform

Модульная платформа для разработки и тестирования торговых моделей.

## ✨ Возможности

- 📊 Загрузка данных из Tinkoff Investments API
- 🔧 Расчёт технических индикаторов
- 🤖 Обучение ML/DL моделей
- 📈 Бэктестинг стратегий
- 📱 Web GUI для управления

## 🚀 Быстрый старт

### Установка

\`\`\`bash
# Клонировать репозиторий
git clone https://github.com/your-username/trading-platform.git
cd trading-platform

# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установить зависимости
pip install -r requirements.txt
\`\`\`

### Конфигурация

\`\`\`bash
# Создать .env файл
cp .env.example .env

# Добавить API токен
echo "TINKOFF_API_TOKEN=your_token_here" >> .env
\`\`\`

### Первый запуск

\`\`\`bash
# Загрузить данные
python -m src.data.loader --ticker SBER --from 2023-01-01

# Обучить модель
python -m src.modeling.train --config configs/models/lightgbm.yaml

# Запустить бэктест
python -m src.backtesting.run --model models/lightgbm_latest.pkl
\`\`\`

## 📖 Документация

- [Руководство пользователя](docs/guides/quickstart.md)
- [API Reference](docs/api/python_api.md)
- [Архитектура](docs/architecture/overview.md)

## 🛠️ Разработка

### Структура проекта

\`\`\`
src/
├── data/          # Загрузка и обработка данных
├── features/      # Feature engineering
├── labeling/      # Создание таргетов
├── modeling/      # Обучение моделей
├── backtesting/   # Тестирование стратегий
└── interfaces/    # GUI/CLI
\`\`\`

### Запуск тестов

\`\`\`bash
pytest tests/
\`\`\`

### Code Style

\`\`\`bash
# Форматирование
black src/
isort src/

# Линтинг
flake8 src/
mypy src/
\`\`\`

## 🤝 Контрибуция

Читайте [CONTRIBUTING.md](CONTRIBUTING.md) для деталей.

## 📄 Лицензия

MIT License - см. [LICENSE](LICENSE)

## 📧 Контакты

- Email: your.email@example.com
- Issues: [GitHub Issues](https://github.com/your-username/trading-platform/issues)
```

## 3. API Documentation

### 3.1 Python API (Docstrings)

Автоматическая генерация из docstrings с помощью Sphinx:

```bash
# Установка Sphinx
pip install sphinx sphinx-rtd-theme

# Инициализация
sphinx-quickstart docs/

# Генерация документации
cd docs/
make html

# Документация будет в docs/_build/html/
```

**conf.py:**
```python
# docs/conf.py
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Google/NumPy docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

# Theme
html_theme = 'sphinx_rtd_theme'

# Napoleon settings (Google style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
```

### 3.2 REST API (OpenAPI/Swagger)

```python
# src/interfaces/api/main.py
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI(
    title="Trading Platform API",
    description="API for trading platform management",
    version="0.1.0"
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Trading Platform API",
        version="0.1.0",
        description="Comprehensive API for trading platform",
        routes=app.routes,
    )

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Документация доступна на /docs (Swagger UI)
# и /redoc (ReDoc)
```

## 4. Руководства пользователя

### 4.1 Quickstart Guide

```markdown
# Quickstart Guide

Это руководство поможет вам начать работу с платформой за 15 минут.

## Предварительные требования

- Python 3.10+
- 8GB RAM (рекомендуется 16GB)
- 50GB свободного места на диске

## Шаг 1: Установка

\`\`\`bash
# Клонировать репозиторий
git clone https://github.com/your-username/trading-platform.git
cd trading-platform

# Установить зависимости
pip install -r requirements.txt
\`\`\`

## Шаг 2: Конфигурация

Создайте \`.env\` файл:

\`\`\`bash
TINKOFF_API_TOKEN=your_token_here
DATABASE_URL=sqlite:///trading.db
LOG_LEVEL=INFO
\`\`\`

## Шаг 3: Загрузка данных

\`\`\`bash
python -m src.data.loader \\
  --ticker SBER \\
  --from 2023-01-01 \\
  --to 2023-12-31 \\
  --timeframe 1h
\`\`\`

Данные будут сохранены в \`artifacts/data/SBER/1h/\`.

## Шаг 4: Обучение модели

\`\`\`bash
python -m src.modeling.train \\
  --config configs/models/lightgbm_basic.yaml
\`\`\`

Модель будет сохранена в \`artifacts/models/\`.

## Шаг 5: Бэктест

\`\`\`bash
python -m src.backtesting.run \\
  --model artifacts/models/lightgbm_20250126.pkl \\
  --data artifacts/data/SBER/1h/test.parquet \\
  --output artifacts/backtests/
\`\`\`

Результаты будут в \`artifacts/backtests/\`.

## Следующие шаги

- [Настройка индикаторов](features.md)
- [Создание кастомных моделей](custom_models.md)
- [Оптимизация гиперпараметров](optimization.md)
```

### 4.2 Tutorial Structure

```markdown
# Tutorial: Создание первой торговой модели

## Что вы изучите

В этом туториале вы:
- Загрузите исторические данные
- Создадите технические индикаторы
- Обучите модель LightGBM
- Протестируете стратегию

Время выполнения: ~30 минут

## Предварительные требования

- Завершен [Quickstart Guide](quickstart.md)
- Доступ к Tinkoff Investments API

## Шаг 1: Подготовка данных

...подробные инструкции с примерами кода...

## Шаг 2: Feature Engineering

...

## Шаг 3: Обучение модели

...

## Что дальше?

Поздравляем! Вы создали свою первую торговую модель.

Дальнейшее чтение:
- [Продвинутые индикаторы](advanced_indicators.md)
- [Ensemble методы](ensemble.md)
```

## 5. Changelog

### 5.1 Структура

```markdown
# Changelog

Все значимые изменения в этом проекте документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/),
и проект следует [Semantic Versioning](https://semver.org/lang/ru/).

## [Unreleased]

### Added
- Новые функции, которые будут в следующем релизе

### Changed
- Изменения существующей функциональности

### Deprecated
- Функции, которые скоро будут удалены

### Removed
- Удалённые функции

### Fixed
- Исправленные баги

### Security
- Исправления безопасности

## [0.2.0] - 2025-01-26

### Added
- LSTM модель для time series prediction
- WebSocket API для real-time updates
- Поддержка GPU обучения
- Dashboard для мониторинга экспериментов

### Changed
- Улучшена производительность расчёта индикаторов (2x ускорение)
- Обновлен UI дашборда

### Fixed
- Исправлена утечка памяти в backtesting engine
- Баг с NaN значениями в RSI при нулевых изменениях цены

## [0.1.0] - 2025-01-01

### Added
- Первый релиз
- Базовая функциональность загрузки данных
- Технические индикаторы (SMA, RSI, MACD)
- LightGBM, CatBoost, XGBoost модели
- Простой backtesting engine

[Unreleased]: https://github.com/user/repo/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/user/repo/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/user/repo/releases/tag/v0.1.0
```

## 6. Architecture Documentation

### 6.1 Архитектурные диаграммы

```markdown
# Architecture Overview

## High-Level Architecture

\`\`\`
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   GUI/CLI   │────▶│  Orchestrator│────▶│  Pipelines  │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   Services   │
                    └──────────────┘
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
         ┌──────────┐ ┌──────────┐ ┌──────────┐
         │   Data   │ │ Features │ │ Modeling │
         └──────────┘ └──────────┘ └──────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   Storage    │
                    └──────────────┘
\`\`\`

## Data Flow

1. **Ingestion**: Данные загружаются из Tinkoff API или локальных файлов
2. **Processing**: Фильтрация, нормализация, валидация
3. **Feature Engineering**: Расчёт технических индикаторов
4. **Labeling**: Создание таргетов
5. **Training**: Обучение моделей
6. **Backtesting**: Симуляция стратегий
7. **Analysis**: Визуализация и анализ результатов

## Component Details

### Data Module

Отвечает за:
- Загрузку данных из источников
- Валидацию и очистку
- Кэширование
- Версионирование датасетов

...подробное описание...
```

## 7. Генерация документации

### 7.1 Automated Documentation

```bash
# Генерация API документации
sphinx-apidoc -o docs/api src/

# Генерация документации
cd docs/
make html

# Открыть в браузере
open _build/html/index.html
```

### 7.2 CI/CD для документации

```yaml
# .github/workflows/docs.yml
name: Documentation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-docs:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install sphinx sphinx-rtd-theme
          pip install -r requirements.txt

      - name: Build documentation
        run: |
          cd docs/
          make html

      - name: Deploy to GitHub Pages
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/_build/html
```

## 8. Best Practices

### 8.1 Checklist
- [ ] README актуален и информативен
- [ ] Все public API документированы
- [ ] Примеры кода работают
- [ ] Диаграммы актуальны
- [ ] CHANGELOG обновляется
- [ ] Документация в CI/CD
- [ ] Версионирование документации
- [ ] Ссылки работают

### 8.2 Рекомендации

**Хорошая документация:**
- ✅ Понятна новичкам
- ✅ Содержит примеры
- ✅ Актуальна
- ✅ Хорошо структурирована
- ✅ Легко найти нужную информацию
- ✅ Содержит troubleshooting

**Антипаттерны:**
- ❌ Устаревшая документация
- ❌ Только autogenerated без примеров
- ❌ Отсутствие quickstart
- ❌ Нет troubleshooting
- ❌ Плохая навигация
- ❌ Отсутствие визуализаций

### 8.3 Maintenance

```bash
# Регулярная проверка
# - Все ли ссылки работают?
# - Примеры кода выполняются?
# - Документация соответствует коду?

# Автоматическая проверка ссылок
linkchecker docs/_build/html/index.html

# Проверка примеров кода
pytest --doctest-modules docs/
```
