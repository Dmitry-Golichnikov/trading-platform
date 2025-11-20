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

### 2. Выбор метода установки

#### Вариант A — pip (только CPU)

- Подходит, если не требуется поддержка GPU.
- Создайте виртуальное окружение и активируйте его:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

- Установите зависимости:

```bash
pip install -r requirements/base.txt
pip install -r requirements/cpu.txt
```

> TA-Lib требует системную библиотеку: Windows — wheel с [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib), Linux — `sudo apt-get install ta-lib`, macOS — `brew install ta-lib`.

#### Вариант B — Conda (CPU и GPU)

- Убедитесь, что выполнено `conda init` для вашей оболочки (например, `conda init powershell`).
- Создайте окружение под нужную конфигурацию:

```bash
# CPU
conda env create -f requirements/conda-cpu.yml
conda activate trading-platform-cpu

# GPU
conda env create -f requirements/conda-gpu.yml
conda activate trading-platform-gpu
```

#### Вариант C — Docker (CPU и GPU)

- Для CPU: `docker compose -f docker-compose.cpu.yml up -d trading-platform-cpu`
- Для GPU: `docker compose -f docker-compose.gpu.yml up -d trading-platform-gpu` (нужен NVIDIA Container Toolkit)
- Для разработки доступны сервисы `trading-platform-dev` и `trading-platform-gpu-dev`.

### 3. Настройка окружения

```bash
# Скопируйте шаблон конфигурации
cp .env.example .env

# Отредактируйте .env и добавьте ваши настройки
# Например, TINKOFF_API_TOKEN для работы с Tinkoff API
```

### 4. Проверка установки

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

### GUI

Desktop приложение на PyQt6 с полным функционалом платформы:

```bash
# Установить GUI зависимости
pip install -e ".[gui]"

# Запустить GUI
trading-gui

# Или напрямую
python src/interfaces/gui/scripts/run_gui.py
```

**Возможности GUI:**
- 📊 Управление данными с визуализацией графиков
- 🔧 Конфигуратор признаков с 30+ индикаторами
- 🎯 Разметка данных (Triple Barrier, Horizon)
- 🤖 Real-time мониторинг обучения моделей
- 📈 Бэктестинг с equity curves и метриками
- 🔬 Комплексные эксперименты (batch processing)

📚 **Документация GUI**: [src/interfaces/gui/desktop/README.md](src/interfaces/gui/desktop/README.md)

### Docker

Проект предоставляет готовые Docker конфигурации для работы на CPU и GPU.

#### CPU режим (ноутбук)

```bash
# Создайте .env файл
cp .env.example .env

# Сборка и запуск контейнера
docker compose -f docker-compose.cpu.yml up -d trading-platform-cpu
```

```bash
# Остановить контейнер
docker compose -f docker-compose.cpu.yml down
```

#### GPU режим (стационарный ПК)

**Требования**: NVIDIA GPU, CUDA 11.8+, NVIDIA Container Toolkit / Docker Desktop GPU support

```bash
# Сборка и запуск контейнера (используется официальный pytorch/pytorch)
docker compose -f docker-compose.gpu.yml up -d trading-platform-gpu
```

```bash
# Остановить контейнер
docker compose -f docker-compose.gpu.yml down
```


#### Доступ к сервисам в Development режиме

- **Контейнер**: подключение через `docker compose ... exec <service> bash`
- **MLflow / Jupyter**: устанавливаются вручную при необходимости (`pip install -r requirements-dev.txt`, запуск из shell)

> ℹ️ **Windows**: используйте Docker Desktop с WSL2, убедитесь что проект находится в расшариваемом каталоге (`C:\Users\<user>\`), и запускайте команды через PowerShell/WSL2. Для GPU включите поддержку GPU в Docker Desktop и обновите драйвер NVIDIA (522+).

📚 **Подробное руководство**: [docs/Docker_Guide.md](docs/Docker_Guide.md)

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

### Общее
- [Техническое задание](technical_spec.md)
- [План реализации](plan/README.md)
- [Быстрый старт](plan/QUICK_START.md)

### Инфраструктура
- [Docker руководство](docs/Docker_Guide.md) - Полное руководство по использованию Docker
- [Системная документация](docs/system/)

### Компоненты
- [Данные](docs/artifacts/Datasets_and_Feature_Store.md)
- [Модели](docs/models/)
- [Индикаторы](docs/indicators/)
- [Метрики](docs/metrics/)

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
