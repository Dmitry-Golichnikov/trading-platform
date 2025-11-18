# Руководство по использованию Docker

## Содержание
- [Введение](#введение)
- [Требования](#требования)
- [Структура Docker конфигураций](#структура-docker-конфигураций)
- [Быстрый старт](#быстрый-старт)
- [Режим CPU](#режим-cpu)
- [Режим GPU](#режим-gpu)
- [Development окружение](#development-окружение)
- [Работа с артефактами](#работа-с-артефактами)
- [Переменные окружения](#переменные-окружения)
- [Troubleshooting](#troubleshooting)

---

## Введение

Проект предоставляет готовые Docker конфигурации для запуска торговой платформы в различных режимах:
- **CPU режим** - для работы на ноутбуках и системах без GPU
- **GPU режим** - для обучения нейросетевых моделей на видеокартах NVIDIA
- **Development режим** - для разработки с Jupyter Lab и MLflow UI

---

## Требования

### Общие требования
- Docker Engine 20.10+
- Docker Compose 2.0+
- Минимум 8 GB RAM
- 10 GB свободного места на диске

### Для GPU режима дополнительно
- NVIDIA GPU с поддержкой CUDA 11.8+
- NVIDIA Driver 520+
- NVIDIA Container Toolkit

### Особенности Windows
- Рекомендуется установить **Docker Desktop** с поддержкой WSL2 (Windows Subsystem for Linux)
- Включите **WSL2** и создайте дистрибутив Ubuntu для использования Linux-контейнеров
- Убедитесь, что файловая шеровая директория (обычно `C:\Users\<user>\`) добавлена в **Resources → File Sharing** в настройках Docker Desktop
- Для GPU-режима требуется Windows 11 22H2+, драйвер NVIDIA 522+ и установленный "GPU Support" в Docker Desktop (CUDA через WSL2)
- Команды из примеров можно выполнять в PowerShell / Windows Terminal; для Linux-команд используйте префикс `wsl --` или открывайте оболочку внутри WSL
- Используйте `docker compose` (через пробел) вместо устаревшей команды `docker-compose`, либо включите соответствующий алиас в Docker Desktop

#### Установка NVIDIA Container Toolkit (Linux)

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Проверка GPU доступности

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## Структура Docker конфигураций

```
.
├── Dockerfile                  # Multi-stage Dockerfile
│   ├── base                   # Базовые системные зависимости
│   ├── dependencies           # Python зависимости
│   ├── development            # Dev окружение с тестами
│   ├── production             # Production образ (CPU)
│   ├── gpu-base              # GPU базовый образ
│   └── gpu-production        # Production образ (GPU)
├── docker-compose.cpu.yml     # Конфигурация для CPU
├── docker-compose.gpu.yml     # Конфигурация для GPU
└── .dockerignore              # Исключения при сборке
```

---

## Быстрый старт

### 1. Создайте .env файл

```bash
cp .env.example .env
```

Отредактируйте `.env` и добавьте свои настройки:

```env
# Tinkoff API
TINKOFF_API_TOKEN=your_token_here
TINKOFF_API_SANDBOX=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Performance
N_JOBS=-1
GPU_ENABLED=false
GPU_DEVICE_ID=0
```

### 2. Выберите режим работы

**Для CPU (ноутбук):**
```bash
docker-compose -f docker-compose.cpu.yml up -d
```

**Для GPU (стационарный ПК):**
```bash
docker-compose -f docker-compose.gpu.yml up -d
```

### 3. Проверьте статус

```bash
# CPU
docker-compose -f docker-compose.cpu.yml ps

# GPU
docker-compose -f docker-compose.gpu.yml ps
```

---

## Режим CPU

#### Сборка образа

```bash
docker compose -f docker-compose.cpu.yml build trading-platform-cpu
```

#### Запуск контейнера

```bash
docker compose -f docker-compose.cpu.yml up -d trading-platform-cpu
```

#### Работа внутри контейнера

```bash
# Открыть shell
docker compose -f docker-compose.cpu.yml exec trading-platform-cpu bash

# Запустить тесты (если установлены dev-зависимости)
docker compose -f docker-compose.cpu.yml exec trading-platform-cpu pytest tests/
```

#### Остановка

```bash
docker compose -f docker-compose.cpu.yml down
```

---

## Режим GPU

В GPU-режиме проект использует официальный образ **pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime**, в котором уже предустановлены PyTorch, CUDA 12.4 и cuDNN 9.

### Сборка образа

```bash
docker compose -f docker-compose.gpu.yml build trading-platform-gpu
```

### Запуск контейнера

```bash
docker compose -f docker-compose.gpu.yml up -d trading-platform-gpu
```

### Проверка GPU

```bash
docker compose -f docker-compose.gpu.yml exec trading-platform-gpu python -c "import torch; print(torch.cuda.is_available())"
```

### Остановка

```bash
docker compose -f docker-compose.gpu.yml down
```

### Обновление зависимостей

- CPU образ собирает зависимости из `requirements/base.txt`. Для PyTorch и связанных пакетов используйте `requirements/cpu.txt`.
- GPU образ ставит зависимости из `requirements/gpu.txt`; базовый образ уже содержит CUDA-runtime.
- Обновляя версии, корректируйте соответствующий файл и пересобирайте контейнеры.

Доступ к сервисам:
- Jupyter Lab: http://localhost:8888
- MLflow UI: http://localhost:5000
- TensorBoard: http://localhost:6006 (опционально)

---

## Development окружение

### Установка дополнительных пакетов

```bash
# Войти в контейнер
docker-compose -f docker-compose.cpu.yml exec trading-platform-dev bash

# Установить пакет
pip install package-name

# Или добавить в requirements-dev.txt и пересобрать
docker-compose -f docker-compose.cpu.yml build trading-platform-dev
```

### Работа с Jupyter Lab

```bash
# Запустить Jupyter
docker-compose -f docker-compose.cpu.yml up -d trading-platform-dev

# Открыть в браузере
open http://localhost:8888
```

Jupyter настроен без токена/пароля для удобства разработки. **Не используйте эту конфигурацию в production!**

### Запуск тестов

```bash
# Все тесты
docker-compose -f docker-compose.cpu.yml exec trading-platform-dev pytest

# Конкретный модуль
docker-compose -f docker-compose.cpu.yml exec trading-platform-dev pytest tests/unit/

# С покрытием
docker-compose -f docker-compose.cpu.yml exec trading-platform-dev pytest --cov=src

# С параметризацией
docker-compose -f docker-compose.cpu.yml exec trading-platform-dev pytest -v -k "test_name"
```

### Линтеры и форматирование

```bash
# Black
docker-compose -f docker-compose.cpu.yml exec trading-platform-dev black src/ tests/

# isort
docker-compose -f docker-compose.cpu.yml exec trading-platform-dev isort src/ tests/

# flake8
docker-compose -f docker-compose.cpu.yml exec trading-platform-dev flake8 src/ tests/

# mypy
docker-compose -f docker-compose.cpu.yml exec trading-platform-dev mypy src/
```

---

## Работа с артефактами

Все артефакты сохраняются в директории `./artifacts` на хосте и монтируются в контейнер.

### Структура artifacts

```
artifacts/
├── data/         # Загруженные данные
├── models/       # Обученные модели
├── features/     # Сгенерированные признаки
├── backtests/    # Результаты бэктестов
├── logs/         # Файлы логов
└── manifests/    # Метаданные
```

### Резервное копирование

```bash
# Создать backup
tar -czf artifacts_backup_$(date +%Y%m%d).tar.gz artifacts/

# Восстановить из backup
tar -xzf artifacts_backup_20241027.tar.gz
```

### Очистка артефактов

```bash
# Удалить все артефакты (ОСТОРОЖНО!)
rm -rf artifacts/*

# Удалить только логи
rm -rf artifacts/logs/*

# Удалить старые модели
find artifacts/models/ -mtime +30 -delete
```

---

## Переменные окружения

### Основные переменные

| Переменная | Описание | Значение по умолчанию |
|-----------|----------|----------------------|
| `LOG_LEVEL` | Уровень логирования | `INFO` |
| `LOG_FORMAT` | Формат логов (json/text) | `json` |
| `GPU_ENABLED` | Использовать GPU | `false` |
| `GPU_DEVICE_ID` | ID GPU устройства | `0` |
| `N_JOBS` | Количество процессов | `-1` (все ядра) |
| `MLFLOW_TRACKING_URI` | URI для MLflow | `file:///app/artifacts/mlruns` |

### Tinkoff API

| Переменная | Описание | Обязательная |
|-----------|----------|--------------|
| `TINKOFF_API_TOKEN` | API токен | Да |
| `TINKOFF_API_SANDBOX` | Использовать sandbox | Нет |

### Пути к директориям

| Переменная | Описание | Значение по умолчанию |
|-----------|----------|----------------------|
| `DATA_DIR` | Директория данных | `/app/artifacts/data` |
| `MODELS_DIR` | Директория моделей | `/app/artifacts/models` |
| `LOGS_DIR` | Директория логов | `/app/artifacts/logs` |

---

## Troubleshooting

### Контейнер не запускается

```bash
# Проверить логи
docker-compose -f docker-compose.cpu.yml logs

# Проверить статус
docker-compose -f docker-compose.cpu.yml ps

# Пересобрать образ
docker-compose -f docker-compose.cpu.yml build --no-cache
```

### GPU не обнаружен

```bash
# Проверить драйвер на хосте
nvidia-smi

# Проверить NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Проверить переменную CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
```

### Ошибки при установке TA-Lib

TA-Lib установлен в базовом образе. Если возникают проблемы:

```bash
# Пересобрать базовый stage
docker-compose -f docker-compose.cpu.yml build --no-cache
```

### Недостаточно памяти

Увеличьте лимиты в docker-compose файле:

```yaml
deploy:
  resources:
    limits:
      memory: 16G  # Увеличить до 16GB
```

### Jupyter не доступен

```bash
# Проверить что контейнер запущен
docker-compose -f docker-compose.cpu.yml ps trading-platform-dev

# Проверить логи Jupyter
docker-compose -f docker-compose.cpu.yml logs trading-platform-dev

# Проверить что порт не занят
netstat -an | grep 8888
```

### MLflow не работает

```bash
# Проверить директорию mlruns
ls -la artifacts/mlruns/

# Создать директорию если нужно
mkdir -p artifacts/mlruns

# Перезапустить контейнер
docker-compose -f docker-compose.cpu.yml restart trading-platform-dev
```

### Конфликт портов

Если порты 8888 или 5000 заняты, измените их в docker-compose файле:

```yaml
ports:
  - "9999:8888"  # Jupyter на порту 9999
  - "5001:5000"  # MLflow на порту 5001
```

### Образ слишком большой

```bash
# Очистить неиспользуемые образы
docker image prune -a

# Посмотреть размер слоев
docker history trading-platform:cpu

# Использовать production образ вместо development
docker-compose -f docker-compose.cpu.yml build --target production
```

---

## Полезные команды

### Мониторинг ресурсов

```bash
# Использование ресурсов всеми контейнерами
docker stats

# Использование диска
docker system df
```

### Очистка

```bash
# Удалить остановленные контейнеры
docker container prune

# Удалить неиспользуемые образы
docker image prune -a

# Удалить неиспользуемые volumes
docker volume prune

# Полная очистка
docker system prune -a --volumes
```

### Экспорт/импорт образов

```bash
# Сохранить образ в файл
docker save trading-platform:cpu -o trading-platform-cpu.tar

# Загрузить образ из файла
docker load -i trading-platform-cpu.tar

# Передать на другую машину
scp trading-platform-cpu.tar user@remote:/path/
```

---

## Best Practices

1. **Используйте .env файл** для управления переменными окружения
2. **Регулярно обновляйте образы** с помощью `docker-compose build`
3. **Делайте backup артефактов** перед экспериментами
4. **Используйте volume mounts** для кода при разработке
5. **Ограничивайте ресурсы** через deploy.resources
6. **Мониторьте логи** с помощью `docker-compose logs -f`
7. **Используйте healthcheck** для проверки работоспособности
8. **Не храните секреты** в docker-compose файлах

---

## Дополнительные ресурсы

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Jupyter Docker Stacks](https://jupyter-docker-stacks.readthedocs.io/)
