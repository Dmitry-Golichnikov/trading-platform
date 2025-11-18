# Multi-stage Dockerfile для trading-platform
# Поддержка CPU и GPU режимов

ARG PYTHON_VERSION=3.13
ARG CUDA_VERSION=12.8.0
ARG CUDNN_VERSION=9

# ============================================================================
# Stage 1: base dependencies
# ============================================================================
FROM python:${PYTHON_VERSION}-slim AS base

# Установка переменных окружения
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Установка TA-Lib (требуется для технического анализа)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Создание рабочей директории
WORKDIR /app

# ============================================================================
# Stage 2: Dependencies installation
# ============================================================================
FROM base AS dependencies

ARG CPU_REQUIREMENTS=cpu

# Копирование файлов зависимостей
COPY requirements/${CPU_REQUIREMENTS}.txt ./requirements.txt

# Установка Python зависимостей
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ============================================================================
# Stage 3: Development (с dev зависимостями)
# ============================================================================
FROM dependencies AS development

# Копирование всего проекта
COPY . .

# Установка проекта в editable режиме
RUN pip install -e .

# Создание директорий для артефактов
RUN mkdir -p artifacts/{data,models,features,backtests,logs,manifests}

# Порт для Jupyter (опционально)
EXPOSE 8888

# Порт для MLflow UI (опционально)
EXPOSE 5000

CMD ["bASh"]

# ============================================================================
# Stage 4: Production (только необходимое для работы)
# ============================================================================
FROM dependencies AS production

# Копирование только исходного кода
COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml ./

# Установка проекта
RUN pip install .

# Создание директорий для артефактов
RUN mkdir -p artifacts/{data,models,features,backtests,logs,manifests}

# Создание non-root пользователя
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Healthcheck (опционально, можно настроить позже)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD python -c "import src; print('OK')" || exit 1

CMD ["python", "-c", "import src; print('Trading platform ready')"]

# ============================================================================
# Stage 5: GPU support (на основе официального PyTorch образа)
# ============================================================================
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime AS gpu-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Установка TA-Lib (если нужно)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# ============================================================================
# Stage 6: GPU Production
# ============================================================================
FROM gpu-base AS gpu-production

# Копирование файлов зависимостей
COPY requirements/gpu.txt ./requirements.txt

# Установка Python зависимостей (с GPU поддержкой для PyTorch)
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt


# Копирование исходного кода
COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml ./

# Установка проекта
RUN pip install .

# Создание директорий для артефактов
RUN mkdir -p artifacts/{data,models,features,backtests,logs,manifests}

# Создание non-root пользователя
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

CMD ["python", "-c", "import src; import torch; print(f'GPU available: {torch.cuda.is_available()}'); print('Trading platform ready')"]
