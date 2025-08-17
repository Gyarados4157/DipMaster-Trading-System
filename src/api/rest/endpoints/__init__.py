"""
API端点模块
==========

导出所有API路由。
"""

from .pnl import router as pnl_router
from .positions import router as positions_router
from .fills import router as fills_router
from .risk import router as risk_router
from .alerts import router as alerts_router
from .performance import router as performance_router
from .health import router as health_router

__all__ = [
    "pnl_router",
    "positions_router", 
    "fills_router",
    "risk_router",
    "alerts_router",
    "performance_router",
    "health_router"
]