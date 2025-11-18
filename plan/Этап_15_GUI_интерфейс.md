# Этап 15: GUI интерфейс

## Цель
Web-based GUI для управления платформой, визуализации и мониторинга.

## Зависимости
Этапы 00-14

## Технологии

**Backend:**
- FastAPI для REST API
- WebSocket для real-time updates

**Frontend:**
- React + TypeScript
- Material-UI / Ant Design
- Plotly.js для графиков
- Redux для state management

## Структура

```
src/interfaces/gui/
├── backend/
│   ├── api/
│   │   ├── routers/           # API endpoints
│   │   ├── models.py          # Pydantic models
│   │   └── dependencies.py
│   ├── services/              # Business logic
│   └── main.py                # FastAPI app
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── store/
│   │   └── api/
│   └── package.json
└── docker-compose.gui.yml
```

## Основные экраны

### 1. Dashboard
- Обзор последних экспериментов
- Лучшие модели
- Графики производительности
- Активные задачи

### 2. Data Management
- Список датасетов
- Загрузка данных
- Quality reports
- Визуализация данных (цена + индикаторы)

### 3. Feature Engineering
- Конфигурация признаков (drag-and-drop)
- Предпросмотр признаков
- Feature importance визуализация

### 4. Model Training
- Создание эксперимента
- Конфигурация модели
- Мониторинг обучения (real-time)
- Логи

### 5. Model Evaluation
- Метрики моделей
- Calibration plots
- Feature importance
- SHAP visualizations

### 6. Backtesting
- Конфигурация бэктеста
- Equity curves
- Trade list
- Метрики стратегии

### 7. Experiments
- Список экспериментов
- Сравнение (таблица, графики)
- MLflow интеграция

## API Endpoints

```python
# FastAPI routes

@app.get("/api/datasets")
async def list_datasets(): ...

@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile): ...

@app.get("/api/experiments")
async def list_experiments(filter: str = None): ...

@app.post("/api/experiments")
async def create_experiment(config: ExperimentConfig): ...

@app.get("/api/experiments/{id}/status")
async def experiment_status(id: str): ...

@app.get("/api/models")
async def list_models(): ...

@app.get("/api/models/{id}/metrics")
async def model_metrics(id: str): ...

@app.post("/api/backtest/run")
async def run_backtest(config: BacktestConfig): ...

# WebSocket для real-time updates
@app.websocket("/ws/training/{experiment_id}")
async def training_updates(websocket: WebSocket, experiment_id: str): ...
```

## Ключевые фичи

- **Real-time updates** (через WebSocket)
- **Interactive charts** (Plotly)
- **Виртуализация** для больших списков
- **Preset management** (сохранение/загрузка конфигов)
- **Dark/Light theme**
- **Responsive design**

## Визуализации

1. **Price charts with indicators**
   - Переключение тикеров/таймфреймов
   - Наложение индикаторов
   - Zoom/Pan

2. **Equity curves**
   - Сравнение стратегий
   - Drawdown overlay

3. **Feature importance**
   - Bar charts
   - SHAP plots

4. **Training curves**
   - Loss/metrics по эпохам
   - Real-time updates

## Критерии готовности

- [ ] Backend API (FastAPI) работает
- [ ] Frontend React app работает
- [ ] Все основные экраны реализованы
- [ ] WebSocket для real-time updates
- [ ] Interactive charts
- [ ] Authentication (базовая, опционально)
- [ ] Docker-compose для GUI
- [ ] Responsive design

## Промпты

```
Фаза 1 - Backend API:
Реализуй FastAPI backend в src/interfaces/gui/backend/:
1. API роутеры для всех операций
2. WebSocket endpoints для real-time
3. Интеграция с src/orchestration/
4. CORS настройки
5. OpenAPI docs

Фаза 2 - Frontend:
Создай React приложение в src/interfaces/gui/frontend/:
1. Setup (create-react-app или Vite)
2. Основные компоненты и страницы
3. API client (axios/fetch)
4. State management (Redux/Zustand)
5. Routing (react-router)

Фаза 3 - Визуализации:
1. Price chart component (с Plotly)
2. Equity curve component
3. Training curves component
4. Metrics tables

Фаза 4 - Integration:
1. WebSocket подключение
2. Real-time updates в UI
3. Docker-compose для всего стека
```

## Важные замечания

**Производительность:**
- Виртуализация для больших списков (react-window)
- Debouncing для search inputs
- Lazy loading компонентов

**UX:**
- Loading states
- Error handling
- Toast notifications
- Keyboard shortcuts

**Deployment:**
- Nginx для frontend
- Gunicorn/Uvicorn для backend
- Docker containers

## Следующий этап
[Этап 16: Интеграция Tinkoff API](Этап_16_Интеграция_Tinkoff_API.md)
