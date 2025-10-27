# Создание виртуального окружения
python -m venv .venv
.venv\Scripts\activate

# Установка зависимостей
pip install -e .
pip install -r requirements-dev.txt

# Запуск тестов
pytest tests/

# Проверка линтеров
black --check src/ tests/
flake8 src/ tests/
mypy src/
