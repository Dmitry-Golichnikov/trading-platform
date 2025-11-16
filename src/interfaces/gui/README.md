# Trading Platform GUI

Web-based graphical user interface for the trading platform.

## Overview

The GUI provides:
- **Dashboard**: Overview of system status, recent experiments, and models
- **Data Management**: Browse and manage datasets
- **Feature Generation**: Generate feature sets from datasets ✨ NEW
- **Experiments**: Create, monitor, and compare experiments
- **Models**: View and manage trained models
- **Backtests**: Run and analyze backtests
- **System**: Monitor system health and resources

## Architecture

### Backend (FastAPI)

Located in `backend/`:

- **API Routers**: REST API endpoints for all operations
  - `datasets.py`: Dataset operations
  - `features.py`: Feature generation ✨ NEW
  - `experiments.py`: Experiment management
  - `models.py`: Model operations
  - `backtests.py`: Backtest operations
  - `system.py`: System health and info
  - `websocket.py`: Real-time updates via WebSocket

- **Services**: Business logic layer
  - `dataset_service.py`
  - `feature_service.py` ✨ NEW
  - `experiment_service.py`
  - `model_service.py`
  - `backtest_service.py`

- **Models**: Pydantic models for request/response validation

### Frontend (React + TypeScript)

Located in `frontend/`:

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **UI Library**: Material-UI (MUI)
- **State Management**: Zustand
- **Data Fetching**: React Query
- **Charts**: Plotly.js
- **Routing**: React Router

## Quick Start

### Development Mode

#### 1. Start Backend

```bash
# From project root
cd src/interfaces/gui/backend
python main.py
```

The backend will be available at http://localhost:8000
API docs at http://localhost:8000/api/docs

#### 2. Start Frontend

```bash
# From project root
cd src/interfaces/gui/frontend
npm install
npm run dev
```

The frontend will be available at http://localhost:5173

### Using Docker Compose

```bash
# From project root
docker-compose -f docker-compose.gui.yml up
```

This starts both backend and frontend in development mode.

## Production Deployment

### 1. Build Frontend

```bash
cd src/interfaces/gui/frontend
npm run build
```

This creates an optimized production build in `dist/`.

### 2. Run with Docker Compose (Production)

```bash
docker-compose -f docker-compose.gui.yml --profile production up
```

This starts:
- Backend API on port 8000
- Nginx serving the frontend on port 80

## Configuration

### Backend Configuration

Environment variables (set in `.env` or docker-compose):

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Paths
ARTIFACTS_DIR=artifacts
CONFIGS_DIR=configs
```

### Frontend Configuration

Create `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## API Documentation

The backend provides interactive API documentation:

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **OpenAPI JSON**: http://localhost:8000/api/openapi.json

## Features

### Real-time Updates

The GUI uses WebSocket connections for real-time updates:

- **Training Progress**: Live updates during model training
- **Task Status**: Real-time task progress
- **System Metrics**: Continuous system health monitoring

Connect to WebSocket endpoints:
- `/ws/training/{experiment_id}` - Training updates
- `/ws/task/{task_id}` - Task updates
- `/ws/global` - Global updates

### Dark Mode

Toggle between light and dark themes using the button in the top-right corner.

### Responsive Design

The GUI is fully responsive and works on desktop, tablet, and mobile devices.

## Development

### Project Structure

```
frontend/
├── src/
│   ├── api/              # API client and WebSocket
│   ├── components/       # Reusable React components
│   ├── pages/            # Page components
│   ├── store/            # Zustand state management
│   ├── types/            # TypeScript type definitions
│   ├── hooks/            # Custom React hooks
│   ├── utils/            # Utility functions
│   ├── App.tsx           # Main App component
│   └── main.tsx          # Entry point
├── package.json
├── tsconfig.json
└── vite.config.ts

backend/
├── api/
│   ├── routers/          # API route handlers
│   ├── models.py         # Pydantic models
│   └── dependencies.py   # FastAPI dependencies
├── services/             # Business logic
└── main.py               # FastAPI app
```

### Adding New Features

#### Backend

1. Create a new router in `backend/api/routers/`
2. Create a service in `backend/services/`
3. Add Pydantic models in `backend/api/models.py`
4. Register the router in `backend/main.py`

#### Frontend

1. Create API client methods in `frontend/src/api/client.ts`
2. Add TypeScript types in `frontend/src/types/`
3. Create components in `frontend/src/components/`
4. Create pages in `frontend/src/pages/`
5. Add routes in `frontend/src/App.tsx`

### Code Style

#### Backend

- Follow PEP 8
- Use type hints
- Document with docstrings

#### Frontend

- Use TypeScript strict mode
- Follow React best practices
- Use functional components with hooks

## Testing

### Backend Tests

```bash
pytest tests/integration/test_gui_backend.py
```

### Frontend Tests

```bash
cd frontend
npm run test
```

## Troubleshooting

### Backend not starting

- Check Python version (3.10+)
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check logs in `logs/` directory

### Frontend not loading

- Clear node_modules: `rm -rf node_modules && npm install`
- Check Node version (18+)
- Verify backend is running

### WebSocket connection issues

- Ensure backend is running and accessible
- Check CORS settings in backend
- Verify WebSocket URL in frontend configuration

### API errors

- Check API logs in backend console
- Verify artifacts directory exists and has correct permissions
- Check MLflow is running (if using experiment tracking)

## Performance

### Backend

- Uses async/await for non-blocking I/O
- Connection pooling for database access
- Caching for frequently accessed data

### Frontend

- Code splitting with React lazy loading
- Virtual scrolling for large lists
- Debouncing for search inputs
- React Query for efficient data fetching and caching

## Security

- CORS configured for allowed origins only
- Input validation with Pydantic
- No authentication in single-user mode (add if needed)

## Contributing

See main project CONTRIBUTING.md for guidelines.

## License

See main project LICENSE file.
