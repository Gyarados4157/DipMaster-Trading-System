"""
DipMaster Enhanced V4 - Dashboard API Service
高性能实时交易仪表板API服务

功能模块：
1. Kafka事件消费者
2. ClickHouse时序数据库
3. REST API端点
4. WebSocket实时流
5. Redis缓存和聚合
6. JWT认证和权限控制
7. 监控和健康检查
"""

import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import structlog
from typing import Dict, Any

from .config import DashboardConfig
from .kafka_consumer import KafkaConsumerService
from .database import DatabaseManager
from .cache import CacheManager
from .auth import AuthManager
from .api import create_api_routes
from .websocket import WebSocketManager
from .monitoring import MonitoringService

# 配置结构化日志
logger = structlog.get_logger(__name__)

# 速率限制器
limiter = Limiter(key_func=get_remote_address)

class DashboardAPIService:
    """DipMaster Dashboard API Service 主服务类"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.kafka_consumer = None
        self.db_manager = None
        self.cache_manager = None
        self.auth_manager = None
        self.websocket_manager = None
        self.monitoring = None
        
    async def initialize(self):
        """初始化所有服务组件"""
        try:
            logger.info("初始化Dashboard API服务...")
            
            # 初始化数据库管理器
            self.db_manager = DatabaseManager(self.config.database)
            await self.db_manager.initialize()
            logger.info("数据库管理器初始化完成")
            
            # 初始化缓存管理器
            self.cache_manager = CacheManager(self.config.redis)
            await self.cache_manager.initialize()
            logger.info("缓存管理器初始化完成")
            
            # 初始化认证管理器
            self.auth_manager = AuthManager(self.config.auth)
            logger.info("认证管理器初始化完成")
            
            # 初始化WebSocket管理器
            self.websocket_manager = WebSocketManager(
                self.cache_manager, 
                self.auth_manager
            )
            logger.info("WebSocket管理器初始化完成")
            
            # 初始化监控服务
            self.monitoring = MonitoringService(
                self.db_manager,
                self.cache_manager
            )
            await self.monitoring.initialize()
            logger.info("监控服务初始化完成")
            
            # 初始化Kafka消费者服务
            self.kafka_consumer = KafkaConsumerService(
                self.config.kafka,
                self.db_manager,
                self.cache_manager,
                self.websocket_manager,
                self.monitoring
            )
            await self.kafka_consumer.initialize()
            logger.info("Kafka消费者服务初始化完成")
            
            # 启动后台任务
            await self._start_background_tasks()
            
            logger.info("Dashboard API服务初始化完成")
            
        except Exception as e:
            logger.error(f"服务初始化失败: {e}")
            raise
    
    async def shutdown(self):
        """关闭所有服务组件"""
        try:
            logger.info("关闭Dashboard API服务...")
            
            if self.kafka_consumer:
                await self.kafka_consumer.shutdown()
            
            if self.monitoring:
                await self.monitoring.shutdown()
                
            if self.websocket_manager:
                await self.websocket_manager.shutdown()
            
            if self.cache_manager:
                await self.cache_manager.shutdown()
                
            if self.db_manager:
                await self.db_manager.shutdown()
            
            logger.info("Dashboard API服务关闭完成")
            
        except Exception as e:
            logger.error(f"服务关闭失败: {e}")
            raise
    
    async def _start_background_tasks(self):
        """启动后台任务"""
        # 启动Kafka消费者任务
        asyncio.create_task(self.kafka_consumer.start_consuming())
        
        # 启动监控任务
        asyncio.create_task(self.monitoring.start_monitoring())
        
        # 启动缓存清理任务
        asyncio.create_task(self.cache_manager.start_cleanup_task())

# 全局服务实例
dashboard_service: DashboardAPIService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI生命周期管理"""
    global dashboard_service
    
    # 启动
    config = DashboardConfig.load_from_file("config/dashboard_config.json")
    dashboard_service = DashboardAPIService(config)
    await dashboard_service.initialize()
    
    yield
    
    # 关闭
    if dashboard_service:
        await dashboard_service.shutdown()

def create_app(config_path: str = None) -> FastAPI:
    """创建FastAPI应用实例"""
    
    app = FastAPI(
        title="DipMaster Enhanced V4 - Dashboard API",
        description="高性能实时交易仪表板API服务",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # 添加中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应该限制域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 添加速率限制
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # 注册API路由
    app.include_router(create_api_routes(), prefix="/api/v1")
    
    # 注册WebSocket路由
    @app.websocket("/ws/{path:path}")
    async def websocket_endpoint(websocket, path: str):
        if dashboard_service and dashboard_service.websocket_manager:
            await dashboard_service.websocket_manager.handle_connection(websocket, path)
        else:
            await websocket.close(code=1013)
    
    # 健康检查端点
    @app.get("/health")
    @limiter.limit("10/minute")
    async def health_check(request: Request):
        """系统健康检查"""
        if not dashboard_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        health_status = await dashboard_service.monitoring.get_health_status()
        
        if health_status["status"] == "healthy":
            return health_status
        else:
            raise HTTPException(status_code=503, detail=health_status)
    
    # 根路径
    @app.get("/")
    async def root():
        return {
            "service": "DipMaster Enhanced V4 - Dashboard API",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs"
        }
    
    return app

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DipMaster Dashboard API Service")
    parser.add_argument("--config", default="config/dashboard_config.json", help="配置文件路径")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--reload", action="store_true", help="开启热重载")
    
    args = parser.parse_args()
    
    # 创建应用
    app = create_app(args.config)
    
    # 启动服务
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        access_log=True,
        use_colors=True
    )

if __name__ == "__main__":
    main()