"""
FastAPI Backend Application

Main entry point for the GUI backend.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.interfaces.gui.backend.api.routers import backtests, datasets, experiments, features, models, system, websocket

# Create FastAPI app
app = FastAPI(
    title="Trading Platform GUI API",
    description="REST API and WebSocket endpoints for trading platform GUI",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8080",  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(datasets.router)
app.include_router(features.router)
app.include_router(experiments.router)
app.include_router(models.router)
app.include_router(backtests.router)
app.include_router(system.router)
app.include_router(websocket.router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {"name": "Trading Platform GUI API", "version": "0.1.0", "docs": "/api/docs", "health": "/api/system/health"}


# Health check
@app.get("/health")
async def health_check():
    """Simple health check"""
    return {"status": "ok"}


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(status_code=404, content={"detail": "Resource not found"})


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    print("Starting Trading Platform GUI Backend...")
    print("API Docs: http://localhost:8000/api/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown tasks"""
    print("Shutting down Trading Platform GUI Backend...")


def main():
    """Run the application"""
    uvicorn.run("src.interfaces.gui.backend.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")


if __name__ == "__main__":
    main()
