"""
API服务配置
===========

定义API服务的配置参数。
"""

import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import ssl


class DatabaseConfig(BaseModel):
    """数据库配置"""
    host: str = Field(default="localhost", description="数据库主机")
    port: int = Field(default=9000, description="数据库端口")
    database: str = Field(default="dipmaster", description="数据库名称")
    username: str = Field(default="default", description="用户名")
    password: str = Field(default="", description="密码")
    secure: bool = Field(default=False, description="是否使用SSL")
    verify_ssl: bool = Field(default=True, description="是否验证SSL证书")
    ca_cert: Optional[str] = Field(None, description="CA证书路径")
    pool_size: int = Field(default=10, description="连接池大小")
    pool_timeout: int = Field(default=30, description="连接池超时(秒)")
    query_timeout: int = Field(default=60, description="查询超时(秒)")
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """从环境变量创建配置"""
        return cls(
            host=os.getenv("CLICKHOUSE_HOST", "localhost"),
            port=int(os.getenv("CLICKHOUSE_PORT", "9000")),
            database=os.getenv("CLICKHOUSE_DATABASE", "dipmaster"),
            username=os.getenv("CLICKHOUSE_USERNAME", "default"),
            password=os.getenv("CLICKHOUSE_PASSWORD", ""),
            secure=os.getenv("CLICKHOUSE_SECURE", "false").lower() == "true",
            verify_ssl=os.getenv("CLICKHOUSE_VERIFY_SSL", "true").lower() == "true",
            ca_cert=os.getenv("CLICKHOUSE_CA_CERT"),
            pool_size=int(os.getenv("CLICKHOUSE_POOL_SIZE", "10")),
            pool_timeout=int(os.getenv("CLICKHOUSE_POOL_TIMEOUT", "30")),
            query_timeout=int(os.getenv("CLICKHOUSE_QUERY_TIMEOUT", "60"))
        )


class KafkaConfig(BaseModel):
    """Kafka配置"""
    bootstrap_servers: List[str] = Field(
        default=["localhost:9092"], 
        description="Kafka服务器地址"
    )
    group_id: str = Field(default="dipmaster-api", description="消费者组ID")
    auto_offset_reset: str = Field(default="latest", description="偏移重置策略")
    security_protocol: str = Field(default="PLAINTEXT", description="安全协议")
    sasl_mechanism: Optional[str] = Field(None, description="SASL机制")
    sasl_username: Optional[str] = Field(None, description="SASL用户名")
    sasl_password: Optional[str] = Field(None, description="SASL密码")
    ssl_context: Optional[ssl.SSLContext] = Field(None, description="SSL上下文")
    
    @validator("auto_offset_reset")
    def validate_offset_reset(cls, v):
        if v not in ["earliest", "latest", "none"]:
            raise ValueError("auto_offset_reset必须是earliest、latest或none")
        return v
    
    @classmethod
    def from_env(cls) -> "KafkaConfig":
        """从环境变量创建配置"""
        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        servers = [s.strip() for s in bootstrap_servers.split(",")]
        
        return cls(
            bootstrap_servers=servers,
            group_id=os.getenv("KAFKA_GROUP_ID", "dipmaster-api"),
            auto_offset_reset=os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest"),
            security_protocol=os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
            sasl_mechanism=os.getenv("KAFKA_SASL_MECHANISM"),
            sasl_username=os.getenv("KAFKA_SASL_USERNAME"),
            sasl_password=os.getenv("KAFKA_SASL_PASSWORD")
        )


class APIConfig(BaseModel):
    """API配置"""
    host: str = Field(default="0.0.0.0", description="监听主机")
    port: int = Field(default=8000, description="监听端口")
    debug: bool = Field(default=False, description="调试模式")
    
    # 认证配置
    enable_auth: bool = Field(default=False, description="启用认证")
    api_key: Optional[str] = Field(None, description="API密钥")
    
    # CORS配置
    cors_origins: List[str] = Field(
        default=["*"], 
        description="允许的CORS源"
    )
    
    # 限流配置
    enable_rate_limiting: bool = Field(default=True, description="启用限流")
    rate_limit_calls: int = Field(default=100, description="限流次数")
    rate_limit_period: int = Field(default=60, description="限流周期(秒)")
    
    # 缓存配置
    enable_caching: bool = Field(default=True, description="启用缓存")
    cache_ttl: int = Field(default=60, description="缓存TTL(秒)")
    
    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    
    @validator("log_level")
    def validate_log_level(cls, v):
        if v.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("无效的日志级别")
        return v.upper()
    
    @classmethod
    def from_env(cls) -> "APIConfig":
        """从环境变量创建配置"""
        cors_origins = os.getenv("API_CORS_ORIGINS", "*")
        origins = [s.strip() for s in cors_origins.split(",")]
        
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            debug=os.getenv("API_DEBUG", "false").lower() == "true",
            enable_auth=os.getenv("API_ENABLE_AUTH", "false").lower() == "true",
            api_key=os.getenv("API_KEY"),
            cors_origins=origins,
            enable_rate_limiting=os.getenv("API_ENABLE_RATE_LIMITING", "true").lower() == "true",
            rate_limit_calls=int(os.getenv("API_RATE_LIMIT_CALLS", "100")),
            rate_limit_period=int(os.getenv("API_RATE_LIMIT_PERIOD", "60")),
            enable_caching=os.getenv("API_ENABLE_CACHING", "true").lower() == "true",
            cache_ttl=int(os.getenv("API_CACHE_TTL", "60")),
            log_level=os.getenv("API_LOG_LEVEL", "INFO")
        )


class WebSocketConfig(BaseModel):
    """WebSocket配置"""
    enable_websocket: bool = Field(default=True, description="启用WebSocket")
    max_connections: int = Field(default=1000, description="最大连接数")
    heartbeat_interval: int = Field(default=30, description="心跳间隔(秒)")
    message_queue_size: int = Field(default=1000, description="消息队列大小")
    
    @classmethod
    def from_env(cls) -> "WebSocketConfig":
        """从环境变量创建配置"""
        return cls(
            enable_websocket=os.getenv("WS_ENABLE", "true").lower() == "true",
            max_connections=int(os.getenv("WS_MAX_CONNECTIONS", "1000")),
            heartbeat_interval=int(os.getenv("WS_HEARTBEAT_INTERVAL", "30")),
            message_queue_size=int(os.getenv("WS_MESSAGE_QUEUE_SIZE", "1000"))
        )


class ServiceConfig(BaseModel):
    """服务整体配置"""
    api: APIConfig
    database: DatabaseConfig  
    kafka: KafkaConfig
    websocket: WebSocketConfig
    
    # 服务信息
    service_name: str = Field(default="dipmaster-data-api", description="服务名称")
    version: str = Field(default="1.0.0", description="服务版本")
    environment: str = Field(default="development", description="运行环境")
    
    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """从环境变量创建配置"""
        return cls(
            api=APIConfig.from_env(),
            database=DatabaseConfig.from_env(),
            kafka=KafkaConfig.from_env(),
            websocket=WebSocketConfig.from_env(),
            service_name=os.getenv("SERVICE_NAME", "dipmaster-data-api"),
            version=os.getenv("SERVICE_VERSION", "1.0.0"),
            environment=os.getenv("ENVIRONMENT", "development")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.dict()
    
    def validate_config(self) -> List[str]:
        """验证配置"""
        issues = []
        
        # 验证数据库配置
        if not self.database.host:
            issues.append("数据库主机不能为空")
        
        if self.database.port < 1 or self.database.port > 65535:
            issues.append("数据库端口必须在1-65535之间")
        
        # 验证Kafka配置
        if not self.kafka.bootstrap_servers:
            issues.append("Kafka服务器地址不能为空")
        
        # 验证API配置
        if self.api.enable_auth and not self.api.api_key:
            issues.append("启用认证时必须设置API密钥")
        
        if self.api.port < 1 or self.api.port > 65535:
            issues.append("API端口必须在1-65535之间")
        
        # 验证限流配置
        if self.api.rate_limit_calls < 1:
            issues.append("限流次数必须大于0")
        
        if self.api.rate_limit_period < 1:
            issues.append("限流周期必须大于0")
        
        return issues