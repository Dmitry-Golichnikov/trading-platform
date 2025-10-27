# Trading Platform

Модульная платформа для разработки и тестирования торговых моделей.

## Описание

Воспроизводимая система для подготовки данных, обучения моделей машинного обучения и бэктестинга торговых стратегий на локальной машине. Обеспечивает прозрачность экспериментов, контроль версий данных и конфигураций, удобный UX.

## Возможности

- 📊 **Управление данными**: Загрузка и нормализация исторических данных из локальных файлов и Tinkoff Investments API
- 🔧 **Генерация признаков**: Декларативные пайплайны с библиотекой из 30+ технических индикаторов
- 🎯 **Разметка таргетов**: Horizon, triple barrier и кастомные методы разметки
- 🤖 **Моделирование**: Поддержка классических ML и нейросетевых моделей (LightGBM, XGBoost, CatBoost, LSTM, Transformer)
- 📈 **Бэктестинг**: Полнофункциональный движок с учетом комиссий, проскальзывания и риск-параметров
- 🔍 **Оптимизация**: Поиск гиперпараметров, AutoML, оптимизация порогов по E[PnL]
- 📊 **Трекинг экспериментов**: Интеграция с MLflow для логирования метрик и артефактов
- 💻 **CLI и GUI**: Удобные интерфейсы для управления всеми этапами

## Требования к системе

- **Python**: 3.10 или выше
- **ОС**: Windows, Linux, macOS
- **RAM**: минимум 8 GB, рекомендуется 16 GB+
- **GPU** (опционально): NVIDIA GPU с CUDA для ускорения обучения нейросетей
- **Место на диске**: минимум 10 GB для артефактов и данных

## Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-username/trading-platform.git
cd trading-platform
```

### 2. Создание виртуального окружения

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. Установка зависимостей

```bash
# Основные зависимости
pip install -r requirements.txt

# Зависимости для разработки
pip install -r requirements-dev.txt
```

**Примечание**: TA-Lib требует предварительной установки C библиотеки:
- **Windows**: Скачайте wheel файл с [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
- **Linux**: `sudo apt-get install ta-lib`
- **macOS**: `brew install ta-lib`

### 4. Настройка окружения

```bash
# Скопируйте шаблон конфигурации
cp .env.example .env

# Отредактируйте .env и добавьте ваши настройки
# Например, TINKOFF_API_TOKEN для работы с Tinkoff API
```

### 5. Проверка установки

```bash
# Запустите тесты
pytest tests/

# Проверьте форматирование
black --check src/ tests/
flake8 src/ tests/
mypy src/
```

## Структура проекта

```
trading-platform/
├── src/                    # Исходный код
│   ├── data/              # Загрузка и хранение данных
│   ├── features/          # Генерация признаков
│   ├── labeling/          # Разметка таргетов
│   ├── modeling/          # Модели и обучение
│   ├── evaluation/        # Оценка моделей
│   ├── backtesting/       # Бэктестинг
│   ├── pipelines/         # Пайплайны
│   ├── orchestration/     # Оркестрация
│   ├── interfaces/        # CLI и GUI
│   └── common/            # Общие утилиты
├── configs/               # Конфигурационные файлы
├── artifacts/             # Артефакты (не в git)
├── tests/                 # Тесты
├── docs/                  # Документация
├── scripts/               # Вспомогательные скрипты
└── infra/                 # Инфраструктура (Docker, CI/CD)
```

## Использование

### CLI

```bash
# Загрузка данных
python -m src.interfaces.cli data load --source local --file data.csv

# Генерация признаков
python -m src.interfaces.cli features generate --config configs/features/default.yaml

# Обучение модели
python -m src.interfaces.cli models train --config configs/models/lightgbm.yaml

# Бэктестинг
python -m src.interfaces.cli backtest run --config configs/backtests/strategy1.yaml
```

### Docker

```bash
# Сборка образа
docker build -t trading-platform .

# Запуск на CPU
docker-compose -f docker-compose.cpu.yml up

# Запуск на GPU
docker-compose -f docker-compose.gpu.yml up
```

## Разработка

### Настройка pre-commit hooks

```bash
pre-commit install
```

### Запуск тестов

```bash
# Все тесты
pytest

# С покрытием
pytest --cov=src --cov-report=html

# Только быстрые тесты
pytest -m "not slow"
```

### Линтеры и форматирование

```bash
# Автоматическое форматирование
black src/ tests/
isort src/ tests/

# Проверка стиля
flake8 src/ tests/

# Проверка типов
mypy src/
```

## Документация

- [Техническое задание](technical_spec.md)
- [План реализации](plan/README.md)
- [Быстрый старт](plan/QUICK_START.md)
- [Системная документация](docs/system/)

## Roadmap

Подробный план разработки в 20 этапов:

- ✅ **Этап 00**: Инициализация проекта
- ⏳ **Этап 01**: Модуль данных
- ⏳ **Этап 02**: Обработка и валидация данных
- ... (см. [план реализации](plan/README.md))

## Лицензия

[Укажите вашу лицензию]

## Контакты

[Ваши контактные данные]

---

**Версия**: 0.1.0
**Статус**: В разработке 🚧
