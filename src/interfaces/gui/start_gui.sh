#!/bin/bash
# Start GUI (Backend + Frontend)

echo "Starting Trading Platform GUI..."

# Check if running in development or production mode
MODE=${1:-dev}

if [ "$MODE" == "dev" ]; then
    echo "Starting in DEVELOPMENT mode..."

    # Start backend in background
    echo "Starting backend..."
    cd backend
    python main.py &
    BACKEND_PID=$!
    cd ..

    # Wait for backend to start
    sleep 3

    # Start frontend
    echo "Starting frontend..."
    cd frontend
    npm install
    npm run dev

    # Cleanup on exit
    trap "kill $BACKEND_PID" EXIT

elif [ "$MODE" == "prod" ]; then
    echo "Starting in PRODUCTION mode..."
    docker-compose -f ../../docker-compose.gui.yml --profile production up
else
    echo "Invalid mode: $MODE"
    echo "Usage: ./start_gui.sh [dev|prod]"
    exit 1
fi
