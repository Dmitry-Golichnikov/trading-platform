# GUI Quick Start

## Быстрый старт GUI

### Запуск в режиме разработки

#### 1. Backend

```bash
# Из корня проекта
cd src/interfaces/gui/backend
python -m src.interfaces.gui.backend.main

# Или из корня проекта
python -m src.interfaces.gui.backend.main
```

Backend будет доступен по адресу: http://localhost:8000

API документация (Swagger): http://localhost:8000/docs

#### 2. Frontend

```bash
# Из корня проекта
cd src/interfaces/gui/frontend
npm install
npm run dev
```

Frontend будет доступен по адресу: http://localhost:5173

### Использование Docker Compose

```bash
# Из корня проекта
docker-compose -f docker-compose.gui.yml up

# С пересборкой
docker-compose -f docker-compose.gui.yml up --build

# В фоновом режиме
docker-compose -f docker-compose.gui.yml up -d
```

После запуска:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Запуск с помощью скриптов

#### Linux/Mac

```bash
./src/interfaces/gui/start_gui.sh
```

#### Windows

```bash
.\src\interfaces\gui\start_gui.bat
```

## Основные функции

### 1. Управление данными (Datasets)
- Просмотр списка датасетов
- Детальная информация о датасете
- Проверка качества данных
- Удаление датасетов

### 2. Генерация признаков (Features)
- **Создание наборов признаков** из существующих датасетов
- Выбор конфигурации признаков:
  - Дефолтные индикаторы (SMA, EMA, RSI, MACD, Bollinger Bands)
  - Трансформации (лаги, скользящие окна)
  - Кастомная конфигурация (YAML)
- Просмотр списка созданных наборов признаков
- Удаление наборов признаков

### 3. Эксперименты (Experiments)
- Создание новых экспериментов
- Мониторинг статуса экспериментов
- Просмотр метрик
- Сравнение экспериментов
- Отмена и удаление экспериментов

### 4. Модели (Models)
- Просмотр списка обученных моделей
- Детальная информация о модели
- Метрики и feature importance
- Деплой моделей
- Удаление моделей

### 5. Бэктесты (Backtests)
- Запуск бэктестов на моделях
- Просмотр результатов
- Equity curves
- Trade lists
- Метрики стратегий

### 6. Система (System)
- Health check
- Системная информация
- Мониторинг ресурсов (CPU, RAM, GPU)
- Активные задачи

## Основной workflow

### Пример: От данных до результатов

1. **Загрузите данные** (Datasets)
   - Данные должны быть в формате Parquet в `artifacts/data/{ticker}/{timeframe}/`

2. **Сгенерируйте признаки** (Features) ✨ **НОВОЕ**
   - Перейдите на страницу "Features"
   - Нажмите "Generate Features"
   - Выберите датасет
   - Укажите название набора признаков
   - Выберите конфигурацию (дефолтную или кастомную)
   - Нажмите "Generate"
   - Дождитесь завершения генерации

3. **Создайте эксперимент** (Experiments)
   - Задайте конфигурацию модели
   - Укажите набор признаков (feature_set_id)
   - Запустите обучение

4. **Оцените модель** (Models)
   - Посмотрите метрики
   - Проанализируйте feature importance
   - Деплойте лучшую модель

5. **Запустите бэктест** (Backtests)
   - Выберите модель
   - Настройте параметры стратегии
   - Проанализируйте результаты

## API Endpoints для генерации признаков

### Список наборов признаков
```
GET /api/features
GET /api/features?dataset_id={dataset_id}
```

### Получить информацию о наборе признаков
```
GET /api/features/{feature_set_id}
```

### Получить данные признаков
```
GET /api/features/{feature_set_id}/data?limit=1000
GET /api/features/{feature_set_id}/data?columns=SMA_20,RSI_14&limit=500
```

### Генерация признаков
```
POST /api/features/generate

Body:
{
  "name": "My Features",
  "dataset_id": "SBER_1h",
  "description": "Technical indicators for SBER 1h",
  "config": {
    "indicators": [
      {"name": "SMA", "params": {"period": 20}},
      {"name": "EMA", "params": {"period": 20}},
      {"name": "RSI", "params": {"period": 14}}
    ],
    "transformers": [
      {"type": "lags", "params": {"lags": [1, 2, 3, 5]}}
    ]
  }
}
```

### Удаление набора признаков
```
DELETE /api/features/{feature_set_id}
```

## Конфигурация признаков

### Дефолтная конфигурация
```yaml
indicators:
  - name: SMA
    params:
      period: 20
  - name: EMA
    params:
      period: 20
  - name: RSI
    params:
      period: 14
  - name: MACD
    params:
      fast: 12
      slow: 26
      signal: 9
  - name: BollingerBands
    params:
      period: 20
      std_dev: 2

transformers:
  - type: lags
    params:
      lags: [1, 2, 3, 5, 10]
  - type: rolling
    params:
      windows: [5, 10, 20]
      functions: [mean, std]
```

### Кастомная конфигурация
Вы можете создать файл YAML в `configs/features/` и указать путь в `config_path`:
```json
{
  "name": "Advanced Features",
  "dataset_id": "SBER_1h",
  "config_path": "features/advanced_config.yml"
}
```

## WebSocket для real-time обновлений

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/global');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Update:', message);
};
```

## Troubleshooting

### Backend не запускается
- Убедитесь, что все зависимости установлены: `pip install -r requirements.txt`
- Проверьте, что порт 8000 не занят
- Запускайте из корня проекта: `python -m src.interfaces.gui.backend.main`

### Frontend не запускается
- Убедитесь, что Node.js установлен: `node --version`
- Установите зависимости: `npm install`
- Проверьте, что порт 5173 не занят

### Генерация признаков не работает
- Убедитесь, что датасет существует в `artifacts/data/`
- Проверьте формат датасета (должен быть Parquet)
- Проверьте логи backend для деталей ошибки

### Данные не отображаются
- Проверьте, что backend работает: http://localhost:8000/health
- Откройте консоль браузера для ошибок
- Проверьте API requests в Network tab

## Дополнительная информация

- Полная документация: `docs/GUI_GUIDE.md`
- API документация: http://localhost:8000/docs (когда backend запущен)
- Конфигурация: `src/interfaces/gui/README.md`
