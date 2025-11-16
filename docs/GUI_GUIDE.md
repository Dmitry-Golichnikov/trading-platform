# Trading Platform GUI Guide

## Overview

The Trading Platform GUI provides a web-based interface for managing all aspects of the trading platform, including datasets, experiments, models, backtests, and system monitoring.

## Features

### 1. Dashboard
- System health overview (CPU, Memory, Disk, GPU)
- Recent experiments and models
- Quick access to all features

### 2. Data Management
- Browse all available datasets
- View dataset details (ticker, timeframe, quality score)
- Quality reports and statistics
- Upload new datasets

### 3. Feature Engineering ✨ UPDATED
- **Генерация наборов признаков** через мастер-конфигуратор
- Выбор десятков индикаторов: трендовых, моментум, волатильности, объёмных, higher TF
- Добавление календарных признаков, one-hot тикеров, лагов, rolling-статистик, разностей
- Настройка chunk size, incremental режима, batch-запуска “для всех датасетов”
- Панель задач:
  - запуск/пауза/возобновление/отмена/перезапуск
  - прогресс-бар и статус в режиме реального времени
  - фильтры по статусу и автоматически обновляемый список
- Каталог готовых наборов с фильтром по датасету и удалением устаревших наборов

### 4. Experiments
- List all experiments with status
- Create new experiments
- Monitor running experiments in real-time
- View experiment metrics and results
- Compare multiple experiments

### 5. Models
- Browse trained models
- View model details and metrics
- Deploy models for backtesting
- Delete unused models

### 6. Backtests
- List all backtest results
- Run new backtests
- View backtest metrics and equity curves
- Analyze trade performance

### 7. System Monitoring
- Real-time system health
- CPU, Memory, Disk usage
- GPU availability and utilization
- Active tasks counter

## Quick Start

### Option 1: Development Mode (Recommended for development)

**Start Backend:**
```bash
cd src/interfaces/gui/backend
python main.py
```

Backend will be available at: http://localhost:8000
API Documentation: http://localhost:8000/api/docs

**Start Frontend:**
```bash
cd src/interfaces/gui/frontend
npm install
npm run dev
```

Frontend will be available at: http://localhost:5173

### Option 2: Using Start Script

**Linux/Mac:**
```bash
cd src/interfaces/gui
chmod +x start_gui.sh
./start_gui.sh dev
```

**Windows:**
```cmd
cd src\interfaces\gui
start_gui.bat dev
```

### Option 3: Docker Compose

**Development:**
```bash
docker-compose -f docker-compose.gui.yml up
```

**Production:**
```bash
docker-compose -f docker-compose.gui.yml --profile production up
```

## Architecture

### Backend (FastAPI)
- **Location:** `src/interfaces/gui/backend/`
- **Framework:** FastAPI
- **Features:**
  - REST API endpoints
  - WebSocket for real-time updates
  - Pydantic models for validation
  - Service layer for business logic

### Frontend (React + TypeScript)
- **Location:** `src/interfaces/gui/frontend/`
- **Framework:** React 18 with TypeScript
- **Build Tool:** Vite
- **UI Library:** Material-UI (MUI)
- **State Management:** Zustand
- **Data Fetching:** React Query
- **Charts:** Plotly.js

## Configuration

### Backend Configuration

Create `.env` file in `backend/` directory:
```env
API_HOST=0.0.0.0
API_PORT=8000
MLFLOW_TRACKING_URI=http://localhost:5000
ARTIFACTS_DIR=../../../artifacts
CONFIGS_DIR=../../../configs
```

### Frontend Configuration

Create `.env` file in `frontend/` directory:
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## API Documentation

The backend provides interactive API documentation:

- **Swagger UI:** http://localhost:8000/api/docs
- **ReDoc:** http://localhost:8000/api/redoc
- **OpenAPI Spec:** http://localhost:8000/api/openapi.json

## Real-time Updates

The GUI uses WebSocket connections for real-time updates:

### Training Updates
Connect to `/ws/training/{experiment_id}` to receive live updates during model training:
- Training progress
- Current epoch
- Metrics (loss, accuracy, etc.)
- Phase information

### Task Updates
Connect to `/ws/task/{task_id}` to receive updates on long-running tasks:
- Task progress
- Status changes
- Completion notifications

### Global Updates
Connect to `/ws/global` for system-wide notifications and updates.

## Charts and Visualizations

### Equity Curves
- Line chart showing portfolio value over time
- Supports multiple strategies comparison
- Interactive zoom and pan

### Price Charts
- Candlestick charts with volume
- Multiple timeframe support
- Technical indicators overlay

### Training Curves
- Loss curves (train and validation)
- Metric curves (accuracy, F1, etc.)
- Real-time updates during training

### Metrics Charts
- Bar charts for model metrics
- Horizontal bar charts for feature importance
- Customizable layouts

## Usage Examples

### Creating an Experiment

1. Navigate to **Experiments** page
2. Click **Create Experiment**
3. Fill in experiment details:
   - Name
   - Description (optional)
   - Configuration (JSON/YAML)
   - Tags (optional)
4. Click **Create**
5. Monitor progress in real-time

### Running a Backtest

1. Navigate to **Backtests** page
2. Click **Run Backtest**
3. Select:
   - Model to test
   - Dataset (optional, uses default if not specified)
   - Backtest configuration
4. Click **Run**
5. View results when complete

### Viewing Model Metrics

1. Navigate to **Models** page
2. Click on a model to view details
3. View:
   - Training/validation metrics
   - Feature importance
   - Confusion matrix
   - Calibration plots

## Keyboard Shortcuts

- `Ctrl/Cmd + K`: Quick search
- `Ctrl/Cmd + B`: Toggle sidebar
- `Ctrl/Cmd + D`: Toggle dark mode
- `Esc`: Close dialogs

## Theme

The GUI supports both light and dark themes. Toggle between themes using the button in the top-right corner of the navigation bar.

## Troubleshooting

### Backend Issues

**Backend not starting:**
- Check Python version (3.10+)
- Install dependencies: `pip install -r backend/requirements.txt`
- Check if port 8000 is available
- View logs in backend console

**API errors:**
- Check if artifacts directory exists
- Verify MLflow is running (for experiment tracking)
- Check backend logs for detailed error messages

### Frontend Issues

**Frontend not loading:**
- Check Node version (18+)
- Clear cache: `rm -rf node_modules && npm install`
- Verify backend is running and accessible
- Check browser console for errors

**WebSocket connection failed:**
- Verify backend is running
- Check WebSocket URL in `.env`
- Ensure CORS is properly configured

**Charts not displaying:**
- Check if Plotly.js is installed
- Verify data format matches expected structure
- Check browser console for errors

### Performance Issues

**Slow page load:**
- Enable production build: `npm run build`
- Use Docker with production profile
- Check network latency to backend

**High memory usage:**
- Reduce data fetch limits
- Enable pagination for large lists
- Clear browser cache

## Development

### Adding New Features

#### Backend
1. Create service in `backend/services/`
2. Create router in `backend/api/routers/`
3. Add Pydantic models in `backend/api/models.py`
4. Register router in `backend/main.py`

#### Frontend
1. Add API methods in `frontend/src/api/client.ts`
2. Create components in `frontend/src/components/`
3. Create pages in `frontend/src/pages/`
4. Add routes in `frontend/src/App.tsx`

### Testing

**Backend:**
```bash
pytest tests/integration/test_gui_backend.py
```

**Frontend:**
```bash
cd frontend
npm run test
```

### Code Style

**Backend:**
- Follow PEP 8
- Use type hints
- Document with docstrings

**Frontend:**
- Use TypeScript strict mode
- Follow React best practices
- Use functional components with hooks

## Production Deployment

### Build Frontend

```bash
cd frontend
npm run build
```

This creates an optimized build in `dist/`.

### Run with Docker

```bash
docker-compose -f docker-compose.gui.yml --profile production up -d
```

This starts:
- Backend API on port 8000
- Nginx serving frontend on port 80

### Nginx Configuration

See `infra/nginx/nginx.conf` for Nginx configuration.

## Security Considerations

- CORS is configured for allowed origins only
- Input validation with Pydantic
- No authentication in single-user mode (add if needed for multi-user)
- WebSocket connections are not authenticated by default

## Performance Optimization

### Backend
- Uses async/await for non-blocking I/O
- Caching for frequently accessed data
- Connection pooling

### Frontend
- Code splitting with lazy loading
- Virtual scrolling for large lists
- Debouncing for search inputs
- React Query caching

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Mobile Support

The GUI is responsive and works on mobile devices, though the experience is optimized for desktop.

## Contributing

See main project CONTRIBUTING.md for guidelines.

## Support

For issues or questions:
1. Check this guide and README.md
2. Check API documentation at /api/docs
3. Review logs (backend console, browser console)
4. Create an issue on GitHub

## License

See main project LICENSE file.
