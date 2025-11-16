"""
System Router

API endpoints for system information and health checks.
"""

from datetime import datetime

import psutil
import torch
from fastapi import APIRouter

from src.interfaces.gui.backend.api.models import SystemHealth

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health", response_model=SystemHealth)
async def get_system_health():
    """Get system health status"""
    # CPU usage
    cpu_usage = psutil.cpu_percent(interval=1)

    # Memory usage
    memory = psutil.virtual_memory()
    memory_usage = memory.percent

    # Disk usage
    disk = psutil.disk_usage("/")
    disk_usage = disk.percent

    # GPU info
    gpu_available = torch.cuda.is_available()
    gpu_usage = None
    gpu_memory_usage = None

    if gpu_available:
        try:
            # Get GPU stats (first GPU only)
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage = float(utilization.gpu)

            # GPU memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_usage = (mem_info.used / mem_info.total) * 100

            pynvml.nvmlShutdown()
        except Exception:
            # If pynvml not available, just report GPU is available
            pass

    # Count active tasks (placeholder - would need to check task queue)
    active_tasks = 0

    # Determine overall status
    status = "healthy"
    if cpu_usage > 90 or memory_usage > 90 or disk_usage > 90:
        status = "warning"
    if cpu_usage > 95 or memory_usage > 95 or disk_usage > 95:
        status = "critical"

    return SystemHealth(
        status=status,
        cpu_usage=cpu_usage,
        memory_usage=memory_usage,
        disk_usage=disk_usage,
        gpu_available=gpu_available,
        gpu_usage=gpu_usage,
        gpu_memory_usage=gpu_memory_usage,
        active_tasks=active_tasks,
        timestamp=datetime.now(),
    )


@router.get("/info")
async def get_system_info():
    """Get detailed system information"""
    return {
        "cpu": {
            "count": psutil.cpu_count(),
            "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        },
        "memory": {
            "total_gb": psutil.virtual_memory().total / (1024**3),
            "available_gb": psutil.virtual_memory().available / (1024**3),
        },
        "disk": {
            "total_gb": psutil.disk_usage("/").total / (1024**3),
            "free_gb": psutil.disk_usage("/").free / (1024**3),
        },
        "gpu": {
            "available": torch.cuda.is_available(),
            "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": (
                [{"id": i, "name": torch.cuda.get_device_name(i)} for i in range(torch.cuda.device_count())]
                if torch.cuda.is_available()
                else []
            ),
        },
    }
