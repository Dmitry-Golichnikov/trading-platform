# Управление зависимостями

## 1. Инструменты управления

### 1.1 Poetry (рекомендуется)

```toml
# pyproject.toml
[tool.poetry]
name = "trading-platform"
version = "0.1.0"
description = "Modular platform for trading model development"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.10"
pandas = "^2.0.0"
numpy = "^1.24.0"
torch = {version = "^2.0.0", optional = true}
scikit-learn = "^1.3.0"
lightgbm = "^4.0.0"
catboost = "^1.2.0"
xgboost = "^2.0.0"
mlflow = "^2.8.0"
pydantic = "^2.4.0"
structlog = "^23.1.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4.0"
pytest-cov = "^4.1.0"
black = "^23.10.0"
isort = "^5.12.0"
flake8 = "^6.1.0"
mypy = "^1.6.0"
pre-commit = "^3.5.0"

[tool.poetry.extras]
gpu = ["torch"]
```

```bash
# Установка зависимостей
poetry install

# Установка с GPU поддержкой
poetry install --extras gpu

# Добавить новую зависимость
poetry add pandas

# Добавить dev зависимость
poetry add --dev pytest

# Обновить зависимости
poetry update

# Экспорт в requirements.txt
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

### 1.2 pip + pip-tools

```bash
# requirements.in (основные зависимости)
pandas>=2.0.0
numpy>=1.24.0
torch>=2.0.0
scikit-learn>=1.3.0

# Компилировать в requirements.txt с точными версиями
pip-compile requirements.in

# requirements.txt будет содержать все зависимости с точными версиями
# pandas==2.1.0
# numpy==1.24.3
# ...

# Установка
pip install -r requirements.txt

# Обновление зависимостей
pip-compile --upgrade requirements.in
```

### 1.3 Conda

```yaml
# environment.yaml
name: trading-platform
channels:
  - conda-forge
  - defaults

dependencies:
  - python=3.10
  - pandas>=2.0.0
  - numpy>=1.24.0
  - pytorch>=2.0.0
  - scikit-learn>=1.3.0
  - pip
  - pip:
      - mlflow>=2.8.0
      - pydantic>=2.4.0
```

```bash
# Создать окружение
conda env create -f environment.yaml

# Активировать
conda activate trading-platform

# Обновить
conda env update -f environment.yaml

# Экспорт
conda env export > environment-lock.yaml
```

## 2. Версионирование зависимостей

### 2.1 Semantic Versioning

```python
# requirements.txt - примеры версионирования

# Точная версия (не рекомендуется для библиотек)
pandas==2.0.0

# Минимальная версия
pandas>=2.0.0

# Compatible release (рекомендуется)
pandas~=2.0.0  # Эквивалентно >=2.0.0, <2.1.0

# Диапазон версий
pandas>=2.0.0,<3.0.0

# Исключить конкретную версию
pandas>=2.0.0,!=2.0.1

# Caret (Poetry syntax)
# pandas^2.0.0 -> >=2.0.0, <3.0.0
```

### 2.2 Lock файлы

```bash
# Poetry автоматически создает poetry.lock
# Коммитить в Git для воспроизводимости!

# pip-tools
pip-compile --generate-hashes requirements.in -o requirements.txt

# Conda
conda list --explicit > conda-lock.txt
```

## 3. Организация requirements

### 3.1 Множественные requirements файлы

```
requirements/
├── base.txt           # Базовые зависимости для всех окружений
├── production.txt     # Production (минимальные)
├── development.txt    # Development (с dev tools)
├── testing.txt        # Тестирование
└── docs.txt          # Для генерации документации
```

**base.txt:**
```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pydantic>=2.4.0
structlog>=23.1.0
```

**production.txt:**
```txt
-r base.txt

# Production-specific
gunicorn>=21.2.0
psycopg2-binary>=2.9.0
boto3>=1.28.0
```

**development.txt:**
```txt
-r base.txt
-r testing.txt

# Dev tools
black>=23.10.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.6.0
ipython>=8.15.0
jupyter>=1.0.0
```

**testing.txt:**
```txt
-r base.txt

pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
hypothesis>=6.88.0
```

### 3.2 Опциональные зависимости

```toml
# pyproject.toml
[tool.poetry.extras]
gpu = ["torch", "torchvision"]
visualization = ["plotly", "matplotlib", "seaborn"]
deep-learning = ["tensorflow", "keras"]
all = ["torch", "torchvision", "plotly", "matplotlib", "tensorflow"]
```

```bash
# Установка с опциональными зависимостями
poetry install --extras "gpu visualization"

# Установка всех опциональных
poetry install --extras all
```

## 4. Security Scanning

### 4.1 Vulnerability Scanning

```bash
# Safety - проверка известных уязвимостей
pip install safety
safety check

# или с файлом requirements
safety check -r requirements.txt

# pip-audit (более современный)
pip install pip-audit
pip-audit

# Trivy - сканирование контейнеров и зависимостей
trivy fs --severity HIGH,CRITICAL .
```

### 4.2 Automated Security Checks

```yaml
# .github/workflows/security.yml
name: Security Scan

on:
  schedule:
    - cron: '0 0 * * 0'  # Еженедельно
  push:
    branches: [main]

jobs:
  security:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install pip-audit safety

      - name: Run pip-audit
        run: pip-audit -r requirements.txt

      - name: Run Safety
        run: safety check -r requirements.txt --json

      - name: Upload SARIF results
        uses: github/codeql-action/upload-sarif@v2
        if: always()
```

### 4.3 Dependabot

```yaml
# .github/dependabot.yml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    reviewers:
      - "your-username"
    labels:
      - "dependencies"
      - "python"

    # Автоматически обновлять только patch версии
    allow:
      - dependency-type: "direct"
        update-type: "semver:patch"

  # Docker dependencies
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

## 5. Обновление зависимостей

### 5.1 Стратегия обновления

```python
# scripts/check_outdated.py
"""Check for outdated dependencies"""
import subprocess
import json

def check_outdated_packages():
    """Проверить устаревшие пакеты"""
    result = subprocess.run(
        ['pip', 'list', '--outdated', '--format=json'],
        capture_output=True,
        text=True
    )

    outdated = json.loads(result.stdout)

    # Группировать по важности
    critical = []
    major = []
    minor = []

    for pkg in outdated:
        current = tuple(map(int, pkg['version'].split('.')))
        latest = tuple(map(int, pkg['latest_version'].split('.')))

        if current[0] < latest[0]:
            major.append(pkg)
        elif current[1] < latest[1]:
            minor.append(pkg)
        else:
            critical.append(pkg)

    print(f"Critical updates (patch): {len(critical)}")
    print(f"Minor updates: {len(minor)}")
    print(f"Major updates: {len(major)}")

    return {'critical': critical, 'minor': minor, 'major': major}
```

### 5.2 Процесс обновления

1. **Проверка**: `poetry show --outdated` или `pip list --outdated`
2. **Анализ**: Прочитать CHANGELOG обновляемых пакетов
3. **Тестирование**: Обновить в dev окружении
4. **CI/CD**: Убедиться что все тесты проходят
5. **Review**: Code review изменений
6. **Deploy**: Постепенный rollout

```bash
# Обновление с осторожностью

# 1. Создать отдельную ветку
git checkout -b deps/update-pandas

# 2. Обновить конкретный пакет
poetry update pandas

# 3. Запустить тесты
pytest

# 4. Если OK - коммит
git commit -m "chore: update pandas to 2.1.0"

# 5. Push и PR
git push origin deps/update-pandas
```

## 6. Разрешение конфликтов

### 6.1 Dependency Resolution

```python
# pip install pip-tools pipdeptree

# Визуализация дерева зависимостей
pipdeptree

# Найти конфликты
pipdeptree --warn fail

# Пример вывода при конфликте:
# Warning!!! Possibly conflicting dependencies found:
# * package-a==1.0.0
#  - requires package-b>=2.0.0
# * package-c==1.5.0
#  - requires package-b<2.0.0
```

### 6.2 Решение конфликтов

```bash
# Option 1: Обновить конфликтующие пакеты
poetry update package-a package-c

# Option 2: Зафиксировать совместимые версии
# pyproject.toml
[tool.poetry.dependencies]
package-b = "^2.0.0"  # Явно указать версию

# Option 3: Использовать extras/optional dependencies
[tool.poetry.extras]
option-a = ["package-a"]
option-c = ["package-c"]
```

## 7. Virtual Environments

### 7.1 Venv

```bash
# Создать
python -m venv venv

# Активировать (Linux/Mac)
source venv/bin/activate

# Активировать (Windows)
venv\Scripts\activate

# Деактивировать
deactivate
```

### 7.2 Poetry Virtual Envs

```bash
# Poetry автоматически создает venv

# Показать путь к venv
poetry env info --path

# Удалить venv
poetry env remove python

# Использовать конкретную версию Python
poetry env use python3.10

# Выполнить команду в venv
poetry run python script.py
```

### 7.3 Conda Environments

```bash
# Создать
conda create -n trading-platform python=3.10

# Активировать
conda activate trading-platform

# Деактивировать
conda deactivate

# Список окружений
conda env list

# Удалить
conda env remove -n trading-platform
```

## 8. Docker Dependencies

### 8.1 Multi-stage Build

```dockerfile
# Dockerfile
FROM python:3.10-slim as builder

# Установить зависимости для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копировать только файлы зависимостей
COPY requirements.txt .

# Установить зависимости в виртуальное окружение
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.10-slim

# Копировать только venv из builder
COPY --from=builder /opt/venv /opt/venv

# Копировать приложение
COPY src/ /app/src/

ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app

CMD ["python", "-m", "src.main"]
```

### 8.2 Caching Dependencies

```dockerfile
# Оптимизированный Dockerfile с кэшированием

# Stage 1: Dependencies
FROM python:3.10-slim as dependencies

# Копировать только requirements FIRST
# (этот layer будет закэширован если requirements не изменились)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application
FROM python:3.10-slim

COPY --from=dependencies /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY src/ /app/src/

WORKDIR /app
CMD ["python", "-m", "src.main"]
```

## 9. Best Practices

### 9.1 Checklist
- [ ] Использовать lock файлы (poetry.lock, requirements.txt)
- [ ] Коммитить lock файлы в Git
- [ ] Pinned versions в production
- [ ] Регулярное сканирование уязвимостей
- [ ] Автоматические обновления (Dependabot)
- [ ] Разделение dev/prod зависимостей
- [ ] Документировать причины version constraints
- [ ] Тестировать обновления перед deploy

### 9.2 Рекомендации

```python
# ✅ Хорошо - указать минимальную версию
pandas>=2.0.0

# ✅ Хорошо - compatible release
pandas~=2.0.0

# ⚠️ Осторожно - точная версия (только для production/lock files)
pandas==2.0.0

# ❌ Плохо - без версии (непредсказуемо)
pandas

# ❌ Плохо - слишком старая версия
pandas>=1.0.0  # Если текущая - 2.x
```

### 9.3 Антипаттерны
- ❌ Не указывать версии вообще
- ❌ Использовать разные версии в dev и prod
- ❌ Не коммитить lock файлы
- ❌ Игнорировать security warnings
- ❌ Обновлять все зависимости сразу без тестирования
- ❌ Копировать venv в Docker образ
- ❌ Устанавливать зависимости как root в контейнере

## 10. Troubleshooting

### 10.1 Общие проблемы

```bash
# Проблема: Conflict resolution
# Решение: Очистить кэш и переустановить
poetry cache clear pypi --all
poetry install

# Проблема: Медленная установка
# Решение: Использовать pip с кэшем
pip install --cache-dir=/tmp/pip-cache -r requirements.txt

# Проблема: ImportError после установки
# Решение: Проверить что venv активирован
which python  # Должен быть путь к venv

# Проблема: Разные результаты на разных машинах
# Решение: Использовать lock файлы и точные версии
```

### 10.2 Debug Dependencies

```bash
# Показать установленные пакеты
pip list

# Показать информацию о пакете
pip show pandas

# Проверить зависимости пакета
pip show pandas | grep Requires

# Найти откуда пакет установлен
python -c "import pandas; print(pandas.__file__)"

# Проверить версию импортированного модуля
python -c "import pandas; print(pandas.__version__)"
```
