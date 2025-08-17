"""
DipMaster数据API服务主程序
========================

启动完整的数据API服务，包括Kafka消费、数据库存储和REST/WebSocket接口。
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.config import ServiceConfig
from src.api.database import ClickHouseClient
from src.api.kafka import KafkaConsumerManager
from src.api.websocket import WebSocketManager, WebSocketHandlerManager
from src.api.cache import CacheManager, cache_warm_up_function
from src.api.rest.app import create_app, run_server

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dipmaster_api.log')
    ]
)

logger = logging.getLogger(__name__)


class DipMasterAPIService:
    """DipMaster API服务主类"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        
        # 核心组件
        self.db_client = None
        self.kafka_consumer = None
        self.ws_manager = None
        self.ws_handlers = None
        self.cache_manager = None
        self.app = None
        
        # 运行状态
        self.running = False
        self.shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """初始化所有组件"""
        logger.info("初始化DipMaster API服务...")
        
        try:
            # 验证配置
            config_issues = self.config.validate_config()
            if config_issues:
                for issue in config_issues:
                    logger.warning(f"配置问题: {issue}")
            
            # 初始化缓存管理器
            logger.info("初始化缓存管理器...")
            cache_config = {
                'memory_max_size': 2000,
                'default_ttl': 300,
                'enable_redis': False  # 可以通过配置启用
            }
            self.cache_manager = CacheManager(cache_config)
            await self.cache_manager.start()
            
            # 预热缓存
            await self.cache_manager.warm_up_cache(cache_warm_up_function)
            
            # 初始化数据库客户端
            logger.info("初始化ClickHouse数据库连接...")
            self.db_client = ClickHouseClient(self.config.database)
            await self.db_client.connect()
            
            # 初始化Kafka消费者
            logger.info("初始化Kafka消费者...")
            self.kafka_consumer = KafkaConsumerManager(
                self.config.kafka,
                self.db_client
            )
            
            # 初始化WebSocket管理器
            logger.info("初始化WebSocket管理器...")
            self.ws_manager = WebSocketManager(
                max_connections=self.config.websocket.max_connections,
                heartbeat_interval=self.config.websocket.heartbeat_interval
            )
            await self.ws_manager.start()
            
            # 初始化WebSocket处理器
            self.ws_handlers = WebSocketHandlerManager(
                self.ws_manager,
                self.db_client
            )
            await self.ws_handlers.start()
            
            # 创建FastAPI应用
            logger.info("创建FastAPI应用...")
            self.app = create_app(
                self.config.api,
                self.db_client,
                self.kafka_consumer
            )
            
            # 注入依赖
            self._inject_dependencies()
            
            logger.info("所有组件初始化完成")
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            await self.cleanup()
            raise
    
    def _inject_dependencies(self):
        """注入依赖到FastAPI应用"""
        # 将组件注入到应用状态
        self.app.state.ws_manager = self.ws_manager
        self.app.state.ws_handlers = self.ws_handlers
        self.app.state.cache_manager = self.cache_manager
        
        # 更新WebSocket端点的依赖
        from src.api.rest.endpoints import websocket
        
        async def get_websocket_manager():
            return self.ws_manager
        
        # 替换依赖函数
        websocket.get_websocket_manager = get_websocket_manager
        
        # 注册WebSocket路由
        self.app.include_router(
            websocket.router,
            prefix="",
            tags=["WebSocket"]
        )
    
    async def start(self):
        """启动服务"""
        if self.running:
            logger.warning("服务已在运行")
            return
        
        logger.info("启动DipMaster API服务...")
        
        try:
            # 启动Kafka消费者
            await self.kafka_consumer.start()
            
            # 连接Kafka事件到WebSocket处理器
            self._connect_kafka_to_websocket()
            
            self.running = True
            logger.info("DipMaster API服务启动成功")
            
            # 启动HTTP服务器
            await self._run_server()
            
        except Exception as e:
            logger.error(f"启动服务失败: {e}")
            await self.stop()
            raise
    
    def _connect_kafka_to_websocket(self):
        """连接Kafka事件到WebSocket处理器"""
        # 这里需要实现Kafka事件到WebSocket的桥接
        # 由于Kafka消费者和WebSocket处理器是独立的，
        # 我们需要一个事件桥接机制
        
        # TODO: 实现事件桥接
        # 可以通过观察者模式或消息队列来实现
        pass
    
    async def _run_server(self):
        """运行HTTP服务器"""
        import uvicorn
        
        config = uvicorn.Config(
            app=self.app,
            host=self.config.api.host,
            port=self.config.api.port,
            log_level=self.config.api.log_level.lower(),
            access_log=True,
            server_header=False,
            date_header=False
        )
        
        server = uvicorn.Server(config)
        
        logger.info(f"HTTP服务器启动: http://{self.config.api.host}:{self.config.api.port}")
        
        # 安装信号处理器
        self._install_signal_handlers()
        
        try:
            await server.serve()
        except asyncio.CancelledError:
            logger.info("HTTP服务器收到停止信号")
        except Exception as e:
            logger.error(f"HTTP服务器异常: {e}")
    
    def _install_signal_handlers(self):
        """安装信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"收到信号 {signum}，开始优雅停机...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def stop(self):
        """停止服务"""
        if not self.running:
            return
        
        logger.info("停止DipMaster API服务...")
        self.running = False
        
        await self.cleanup()
        
        logger.info("DipMaster API服务已停止")
        self.shutdown_event.set()
    
    async def cleanup(self):
        """清理资源"""
        logger.info("清理服务资源...")
        
        # 停止WebSocket处理器
        if self.ws_handlers:
            await self.ws_handlers.stop()
        
        # 停止WebSocket管理器
        if self.ws_manager:
            await self.ws_manager.stop()
        
        # 停止Kafka消费者
        if self.kafka_consumer:
            await self.kafka_consumer.stop()
        
        # 关闭数据库连接
        if self.db_client:
            await self.db_client.disconnect()
        
        # 停止缓存管理器
        if self.cache_manager:
            await self.cache_manager.stop()
        
        logger.info("资源清理完成")
    
    async def health_check(self) -> dict:
        """健康检查"""
        checks = {
            "service": "healthy" if self.running else "unhealthy",
            "database": "unknown",
            "kafka": "unknown",
            "websocket": "unknown",
            "cache": "unknown"
        }
        
        try:
            # 检查数据库
            if self.db_client:
                db_health = await self.db_client.health_check()
                checks["database"] = db_health.get("status", "unknown")
            
            # 检查Kafka
            if self.kafka_consumer:
                kafka_health = await self.kafka_consumer.health_check()
                checks["kafka"] = kafka_health.get("status", "unknown")
            
            # 检查WebSocket
            if self.ws_manager:
                checks["websocket"] = "healthy" if self.ws_manager.running else "unhealthy"
            
            # 检查缓存
            if self.cache_manager:
                checks["cache"] = "healthy" if self.cache_manager.running else "unhealthy"
            
        except Exception as e:
            logger.error(f"健康检查异常: {e}")
        
        return checks


async def main():
    """主函数"""
    try:
        # 加载配置
        logger.info("加载配置...")
        config = ServiceConfig.from_env()
        
        logger.info(f"服务配置: {config.service_name} v{config.version} ({config.environment})")
        
        # 创建服务实例
        service = DipMasterAPIService(config)
        
        # 初始化服务
        await service.initialize()
        
        # 启动服务
        await service.start()
        
        # 等待停止信号
        await service.shutdown_event.wait()
        
    except KeyboardInterrupt:
        logger.info("收到键盘中断信号")
    except Exception as e:
        logger.error(f"服务运行异常: {e}", exc_info=True)
        return 1
    
    logger.info("服务已退出")
    return 0


if __name__ == "__main__":
    # 运行主程序
    exit_code = asyncio.run(main())
    sys.exit(exit_code)