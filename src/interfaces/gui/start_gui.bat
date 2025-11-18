@echo off
REM Start GUI (Backend + Frontend) on Windows

echo Starting Trading Platform GUI...

set MODE=%1
if "%MODE%"=="" set MODE=dev

if "%MODE%"=="dev" (
    echo Starting in DEVELOPMENT mode...

    REM Start backend in background
    echo Starting backend...
    start "Backend" cmd /k "cd backend && python main.py"

    REM Wait for backend to start
    timeout /t 3 /nobreak

    REM Start frontend
    echo Starting frontend...
    cd frontend
    call npm install
    call npm run dev

) else if "%MODE%"=="prod" (
    echo Starting in PRODUCTION mode...
    docker-compose -f ..\..\docker-compose.gui.yml --profile production up
) else (
    echo Invalid mode: %MODE%
    echo Usage: start_gui.bat [dev^|prod]
    exit /b 1
)
