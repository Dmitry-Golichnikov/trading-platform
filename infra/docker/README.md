# Docker конфигурации

Эта директория содержит дополнительные Docker конфигурации и скрипты для проекта.

## Основные файлы Docker

Основные Docker конфигурации находятся в корне проекта:

- `../../Dockerfile` - Multi-stage Dockerfile
- `../../docker-compose.cpu.yml` - Конфигурация для CPU
- `../../docker-compose.gpu.yml` - Конфигурация для GPU
- `../../.dockerignore` - Исключения при сборке

## Документация

Полное руководство по использованию Docker: [docs/Docker_Guide.md](../../docs/Docker_Guide.md)

## Быстрый старт

### CPU режим

```bash
# Из корня проекта
docker compose -f docker-compose.cpu.yml up -d trading-platform-cpu
```

### GPU режим (официальный образ PyTorch)

```bash
# Сборка и запуск
docker compose -f docker-compose.gpu.yml up -d trading-platform-gpu
```

### Управление

```bash
# Подключиться к контейнеру
# CPU
docker compose -f docker-compose.cpu.yml exec trading-platform-cpu bash

# GPU
docker compose -f docker-compose.gpu.yml exec trading-platform-gpu bash

# Остановка
docker compose -f docker-compose.cpu.yml down
docker compose -f docker-compose.gpu.yml down
```

Если нужны инструменты разработки (pytest, Jupyter, MLflow) внутри контейнера — установите dev-зависимости вручную:

```bash
docker compose -f docker-compose.cpu.yml exec trading-platform-cpu pip install -r requirements-dev.txt
```

Доступ к сервисам:
- Jupyter Lab: http://localhost:8888
- MLflow UI: http://localhost:5000

## Структура образов

```
┌─────────────────┐
│  base           │  Python 3.10 + системные зависимости + TA-Lib
└────────┬────────┘
         │
         ├─────────────────────────────────┐
         │                                 │
┌────────▼────────┐              ┌────────▼────────┐
│  dependencies   │              │   gpu-base      │
│  + requirements │              │   CUDA 11.8     │
└────────┬────────┘              └────────┬────────┘
         │                                 │
    ┌────┴────┐                           │
    │         │                            │
┌───▼────┐ ┌─▼──────────┐      ┌──────────▼─────────┐
│ prod   │ │ development│      │  gpu-production    │
│ (CPU)  │ │  + dev deps│      │  + PyTorch (CUDA)  │
└────────┘ └────────────┘      └────────────────────┘
```

## Дополнительные скрипты

В будущем здесь могут быть добавлены:
- Скрипты для автоматической сборки образов
- Health check скрипты
- Скрипты миграции данных
- Утилиты для мониторинга

## Требования

- Docker Engine 20.10+
- Docker Compose 2.0+
- Для GPU: NVIDIA Container Toolkit (для WSL2 включается через Docker Desktop)

### Особенности Windows
- Установите **Docker Desktop** и активируйте **WSL2 backend**
- Создайте Linux-дистрибутив (Ubuntu) в WSL2 и запускайте сборки из него для лучшей совместимости
- Убедитесь, что путь проекта находится внутри каталога, расшариваемого Docker Desktop (`C:\Users\...`)
- Для GPU используйте Windows 11 22H2+, драйвер NVIDIA 522+ и включите GPU Support в Docker Desktop
- Команды `docker compose` выполняйте в PowerShell/Windows Terminal; Linux-утилиты можно запускать через `wsl -- <команда>`

## Полезные команды

```bash
# Сборка
docker-compose -f docker-compose.cpu.yml build

# Запуск
docker-compose -f docker-compose.cpu.yml up -d

# Логи
docker-compose -f docker-compose.cpu.yml logs -f

# Остановка
docker-compose -f docker-compose.cpu.yml down
```

## Troubleshooting

См. раздел Troubleshooting в [docs/Docker_Guide.md](../../docs/Docker_Guide.md)
