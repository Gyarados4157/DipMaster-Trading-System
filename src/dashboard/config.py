"""
Dashboard API Service 配置管理
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

@dataclass
class KafkaConfig:
    """Kafka配置"""
    bootstrap_servers: List[str]
    consumer_group: str
    topics: Dict[str, str]  # topic_name -> schema_version
    batch_size: int = 1000
    max_poll_interval_ms: int = 300000
    enable_auto_commit: bool = False
    auto_offset_reset: str = "latest"
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None

@dataclass
class DatabaseConfig:
    """ClickHouse数据库配置"""
    host: str
    port: int = 9000
    database: str = "dipmaster"
    username: str = "default"
    password: str = ""
    secure: bool = False
    verify: bool = True
    pool_size: int = 10
    max_overflow: int = 20
    
    # 表配置
    tables: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.tables is None:
            self.tables = {
                "exec_reports": {
                    "partition_by": "toYYYYMM(timestamp)",
                    "order_by": ["timestamp", "symbol"],
                    "ttl": "timestamp + INTERVAL 1 YEAR",
                    "settings": {
                        "index_granularity": 8192
                    }
                },
                "risk_metrics": {
                    "partition_by": "toYYYYMM(timestamp)",
                    "order_by": ["timestamp", "account_id"],
                    "ttl": "timestamp + INTERVAL 6 MONTH"
                },
                "pnl_curve": {
                    "partition_by": "toYYYYMM(timestamp)",
                    "order_by": ["timestamp", "symbol"],
                    "ttl": "timestamp + INTERVAL 2 YEAR"
                },
                "alerts": {
                    "partition_by": "toYYYYMM(timestamp)",
                    "order_by": ["timestamp", "severity"],
                    "ttl": "timestamp + INTERVAL 3 MONTH"
                }
            }

@dataclass
class RedisConfig:
    """Redis缓存配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 100
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    
    # 缓存策略
    cache_ttl: Dict[str, int] = None
    
    def __post_init__(self):
        if self.cache_ttl is None:
            self.cache_ttl = {
                "pnl_data": 60,        # PnL数据缓存60秒
                "positions": 30,       # 持仓数据缓存30秒
                "risk_metrics": 60,    # 风险指标缓存60秒
                "market_data": 10,     # 市场数据缓存10秒
                "alerts": 5,           # 告警数据缓存5秒
                "health_status": 30    # 健康状态缓存30秒
            }

@dataclass
class AuthConfig:
    """认证配置"""
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    api_key_header: str = "X-API-Key"
    
    # 权限配置
    roles: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.roles is None:
            self.roles = {
                "admin": ["read", "write", "admin"],
                "trader": ["read", "write"],
                "viewer": ["read"]
            }

@dataclass
class WebSocketConfig:
    """WebSocket配置"""
    max_connections: int = 1000
    heartbeat_interval: int = 30
    message_queue_size: int = 1000
    compression: bool = True
    
    # 频道配置
    channels: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = {
                "alerts": {
                    "buffer_size": 100,
                    "rate_limit": "10/second"
                },
                "positions": {
                    "buffer_size": 50,
                    "rate_limit": "5/second"
                },
                "pnl": {
                    "buffer_size": 200,
                    "rate_limit": "20/second"
                },
                "risk": {
                    "buffer_size": 50,
                    "rate_limit": "5/second"
                }
            }

@dataclass
class MonitoringConfig:
    """监控配置"""
    enabled: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    
    # 性能阈值
    thresholds: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = {
                "api_response_time": {
                    "warning": 0.2,    # 200ms
                    "critical": 0.5    # 500ms
                },
                "database_query_time": {
                    "warning": 0.1,    # 100ms
                    "critical": 0.3    # 300ms
                },
                "kafka_lag": {
                    "warning": 1000,   # 1000条消息
                    "critical": 5000   # 5000条消息
                },
                "memory_usage": {
                    "warning": 0.8,    # 80%
                    "critical": 0.9    # 90%
                },
                "cpu_usage": {
                    "warning": 0.7,    # 70%
                    "critical": 0.9    # 90%
                }
            }

@dataclass
class DashboardConfig:
    """Dashboard API Service完整配置"""
    kafka: KafkaConfig
    database: DatabaseConfig
    redis: RedisConfig
    auth: AuthConfig
    websocket: WebSocketConfig
    monitoring: MonitoringConfig
    
    # 全局配置
    debug: bool = False
    log_level: str = "INFO"
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'DashboardConfig':
        """从配置文件加载配置"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DashboardConfig':
        """从字典创建配置"""
        kafka_config = KafkaConfig(**data.get('kafka', {}))
        database_config = DatabaseConfig(**data.get('database', {}))
        redis_config = RedisConfig(**data.get('redis', {}))
        auth_config = AuthConfig(**data.get('auth', {}))
        websocket_config = WebSocketConfig(**data.get('websocket', {}))
        monitoring_config = MonitoringConfig(**data.get('monitoring', {}))
        
        return cls(
            kafka=kafka_config,
            database=database_config,
            redis=redis_config,
            auth=auth_config,
            websocket=websocket_config,
            monitoring=monitoring_config,
            debug=data.get('debug', False),
            log_level=data.get('log_level', 'INFO'),
            max_request_size=data.get('max_request_size', 10 * 1024 * 1024)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'kafka': self.kafka.__dict__,
            'database': self.database.__dict__,
            'redis': self.redis.__dict__,
            'auth': self.auth.__dict__,
            'websocket': self.websocket.__dict__,
            'monitoring': self.monitoring.__dict__,
            'debug': self.debug,
            'log_level': self.log_level,
            'max_request_size': self.max_request_size
        }
    
    def save_to_file(self, config_path: str):
        """保存配置到文件"""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

def create_default_config() -> DashboardConfig:
    """创建默认配置"""
    return DashboardConfig(
        kafka=KafkaConfig(
            bootstrap_servers=["localhost:9092"],
            consumer_group="dipmaster_dashboard",
            topics={
                "exec.reports.v1": "v1",
                "risk.metrics.v1": "v1",
                "alerts.v1": "v1",
                "strategy.performance.v1": "v1"
            }
        ),
        database=DatabaseConfig(
            host="localhost",
            port=9000,
            database="dipmaster",
            username="default",
            password=""
        ),
        redis=RedisConfig(
            host="localhost",
            port=6379,
            db=0
        ),
        auth=AuthConfig(
            jwt_secret_key="your-secret-key-change-in-production"
        ),
        websocket=WebSocketConfig(),
        monitoring=MonitoringConfig()
    )