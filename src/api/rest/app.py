"""
FastAPI应用程序
==============

创建和配置FastAPI应用实例。
"""

import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from ..config import APIConfig
from ..database import ClickHouseClient
from ..kafka import KafkaConsumerManager
from .endpoints import (
    pnl_router, positions_router, fills_router,
    risk_router, alerts_router, performance_router, health_router
)
from .middleware import (
    TimingMiddleware, RateLimitMiddleware, AuthMiddleware,
    ErrorHandlingMiddleware
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("启动DipMaster数据API服务")
    
    # 初始化数据库连接
    await app.state.db_client.connect()
    logger.info("数据库连接已建立")
    
    # 启动Kafka消费者
    await app.state.kafka_consumer.start()
    logger.info("Kafka消费者已启动")
    
    yield
    
    # 关闭时
    logger.info("关闭DipMaster数据API服务")
    
    # 停止Kafka消费者
    await app.state.kafka_consumer.stop()
    logger.info("Kafka消费者已停止")
    
    # 关闭数据库连接
    await app.state.db_client.disconnect()
    logger.info("数据库连接已关闭")


def create_app(config: APIConfig, db_client: ClickHouseClient, kafka_consumer: KafkaConsumerManager) -> FastAPI:
    """创建FastAPI应用"""
    
    app = FastAPI(
        title="DipMaster Data API",
        description="DipMaster交易系统数据服务API",
        version="1.0.0",
        docs_url="/docs" if config.debug else None,
        redoc_url="/redoc" if config.debug else None,
        lifespan=lifespan
    )
    
    # 设置应用状态
    app.state.config = config
    app.state.db_client = db_client
    app.state.kafka_consumer = kafka_consumer
    app.state.start_time = datetime.utcnow()
    
    # 添加中间件
    _add_middleware(app, config)
    
    # 注册路由
    _register_routers(app)
    
    # 添加异常处理器
    _add_exception_handlers(app)
    
    return app


def _add_middleware(app: FastAPI, config: APIConfig):
    """添加中间件"""
    
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Gzip压缩
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 自定义中间件
    app.add_middleware(TimingMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    
    if config.enable_rate_limiting:
        app.add_middleware(
            RateLimitMiddleware,
            calls=config.rate_limit_calls,
            period=config.rate_limit_period
        )
    
    if config.enable_auth:
        app.add_middleware(AuthMiddleware, api_key=config.api_key)


def _register_routers(app: FastAPI):
    """注册路由"""
    
    # API路由
    app.include_router(pnl_router, prefix="/api", tags=["PnL"])
    app.include_router(positions_router, prefix="/api", tags=["Positions"])
    app.include_router(fills_router, prefix="/api", tags=["Fills"])
    app.include_router(risk_router, prefix="/api", tags=["Risk"])
    app.include_router(alerts_router, prefix="/api", tags=["Alerts"])
    app.include_router(performance_router, prefix="/api", tags=["Performance"])
    
    # 健康检查路由
    app.include_router(health_router, prefix="", tags=["Health"])
    
    # 根路径
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "DipMaster Data API",
            "version": "1.0.0",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat()
        }


def _add_exception_handlers(app: FastAPI):
    """添加异常处理器"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error_code": f"HTTP_{exc.status_code}",
                "error_message": exc.detail,
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": getattr(request.state, "request_id", None)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"未处理的异常: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "INTERNAL_SERVER_ERROR",
                "error_message": "服务器内部错误",
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": getattr(request.state, "request_id", None)
            }
        )


def run_server(
    app: FastAPI,
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    log_level: str = "info"
):
    """运行服务器"""
    
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        access_log=True,
        server_header=False,
        date_header=False
    )
    
    server = uvicorn.Server(config)
    
    logger.info(f"启动API服务器: http://{host}:{port}")
    
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("服务器已停止")
    except Exception as e:
        logger.error(f"服务器运行错误: {e}")
        raise