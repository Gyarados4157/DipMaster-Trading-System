#!/usr/bin/env python3
"""
DipMaster Trading System - Advanced Monitoring Architecture
监控架构 - 实时监控和日志收集系统架构设计

Author: DipMaster Trading System Monitoring Agent
Date: 2025-08-18
Version: 1.0.0
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoringEventType(Enum):
    """监控事件类型枚举"""
    # 交易事件
    SIGNAL_GENERATED = "signal_generated"
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    POSITION_UPDATE = "position_update"
    EXECUTION_REPORT = "execution_report"
    
    # 风险事件
    RISK_LIMIT_BREACH = "risk_limit_breach"
    POSITION_SIZE_ALERT = "position_size_alert"
    DRAWDOWN_ALERT = "drawdown_alert"
    VAR_EXCEEDED = "var_exceeded"
    CORRELATION_ALERT = "correlation_alert"
    
    # 系统事件
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONNECTION_LOST = "connection_lost"
    CONNECTION_RESTORED = "connection_restored"
    API_ERROR = "api_error"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    
    # 策略事件
    STRATEGY_DRIFT = "strategy_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CONSISTENCY_VIOLATION = "consistency_violation"
    TIMING_VIOLATION = "timing_violation"
    
    # 健康检查事件
    HEALTH_CHECK = "health_check"
    COMPONENT_FAILURE = "component_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class AlertSeverity(Enum):
    """告警严重性级别"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class MonitoringMetric:
    """监控指标数据结构"""
    name: str
    value: float
    unit: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'labels': self.labels
        }


@dataclass
class AlertEvent:
    """告警事件数据结构"""
    alert_id: str
    severity: AlertSeverity
    category: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    affected_systems: List[str] = field(default_factory=list)
    recommended_action: str = ""
    auto_remediation: bool = False
    tags: Dict[str, str] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式用于Kafka发布"""
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.value,
            'category': self.category,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'affected_systems': self.affected_systems,
            'recommended_action': self.recommended_action,
            'auto_remediation': self.auto_remediation,
            'tags': self.tags,
            'details': self.details
        }


@dataclass
class TradingEvent:
    """交易事件数据结构"""
    event_id: str
    event_type: MonitoringEventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str = ""
    strategy: str = "dipmaster"
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'strategy': self.strategy,
            'data': self.data
        }


@dataclass
class PerformanceMetrics:
    """性能指标数据结构"""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    trades_count: int = 0
    avg_holding_time: float = 0.0
    success_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


class MonitoringComponent(ABC):
    """监控组件抽象基类"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.stats = {'events_processed': 0, 'errors_count': 0}
    
    @abstractmethod
    async def start(self) -> None:
        """启动组件"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """停止组件"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0
        return {
            'name': self.name,
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            **self.stats
        }


class DataCollector(MonitoringComponent):
    """数据收集器 - 收集系统和交易数据"""
    
    def __init__(self, name: str = "data_collector", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.collection_interval = config.get('collection_interval', 1.0) if config else 1.0
        self.buffer_size = config.get('buffer_size', 1000) if config else 1000
        self.data_buffer: List[Dict[str, Any]] = []
        self._collection_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """启动数据收集"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = datetime.now(timezone.utc)
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info(f"📊 {self.name} started")
    
    async def stop(self) -> None:
        """停止数据收集"""
        self.is_running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        logger.info(f"🛑 {self.name} stopped")
    
    async def _collection_loop(self) -> None:
        """数据收集主循环"""
        while self.is_running:
            try:
                # 收集系统指标
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Data collection error: {e}")
                self.stats['errors_count'] += 1
                await asyncio.sleep(1)
    
    async def _collect_system_metrics(self) -> None:
        """收集系统性能指标"""
        try:
            import psutil
            
            timestamp = datetime.now(timezone.utc)
            
            # 系统资源指标
            metrics = [
                MonitoringMetric(
                    name="system.cpu.usage_percent",
                    value=psutil.cpu_percent(),
                    unit="%",
                    timestamp=timestamp
                ),
                MonitoringMetric(
                    name="system.memory.usage_percent",
                    value=psutil.virtual_memory().percent,
                    unit="%",
                    timestamp=timestamp
                ),
                MonitoringMetric(
                    name="system.disk.usage_percent",
                    value=psutil.disk_usage('/').percent,
                    unit="%",
                    timestamp=timestamp
                )
            ]
            
            # 存储到缓冲区
            for metric in metrics:
                self.data_buffer.append(metric.to_dict())
                self.stats['events_processed'] += 1
            
            # 清理缓冲区
            if len(self.data_buffer) > self.buffer_size:
                self.data_buffer = self.data_buffer[-self.buffer_size:]
        
        except ImportError:
            # psutil不可用时跳过系统指标收集
            pass
        except Exception as e:
            logger.error(f"❌ System metrics collection failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'status': 'healthy' if self.is_running else 'unhealthy',
            'buffer_size': len(self.data_buffer),
            'collection_interval': self.collection_interval,
            **self.get_stats()
        }
    
    def get_recent_metrics(self, count: int = 100) -> List[Dict[str, Any]]:
        """获取最近的指标数据"""
        return self.data_buffer[-count:] if self.data_buffer else []


class EventProcessor(MonitoringComponent):
    """事件处理器 - 处理和路由监控事件"""
    
    def __init__(self, name: str = "event_processor", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_handlers: Dict[MonitoringEventType, List[callable]] = {}
        self._processing_task: Optional[asyncio.Task] = None
        self.max_queue_size = config.get('max_queue_size', 10000) if config else 10000
    
    async def start(self) -> None:
        """启动事件处理"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = datetime.now(timezone.utc)
        self._processing_task = asyncio.create_task(self._processing_loop())
        logger.info(f"⚡ {self.name} started")
    
    async def stop(self) -> None:
        """停止事件处理"""
        self.is_running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info(f"🛑 {self.name} stopped")
    
    def register_handler(self, event_type: MonitoringEventType, handler: callable) -> None:
        """注册事件处理器"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def publish_event(self, event: Union[TradingEvent, AlertEvent]) -> None:
        """发布事件到处理队列"""
        try:
            if self.event_queue.qsize() >= self.max_queue_size:
                logger.warning(f"⚠️ Event queue full, dropping oldest events")
                # 清理一半队列
                temp_events = []
                while not self.event_queue.empty() and len(temp_events) < self.max_queue_size // 2:
                    try:
                        temp_events.append(self.event_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                
                # 重新放入较新的事件
                for temp_event in temp_events:
                    try:
                        self.event_queue.put_nowait(temp_event)
                    except asyncio.QueueFull:
                        break
            
            await self.event_queue.put(event)
            
        except Exception as e:
            logger.error(f"❌ Failed to publish event: {e}")
            self.stats['errors_count'] += 1
    
    async def _processing_loop(self) -> None:
        """事件处理主循环"""
        while self.is_running:
            try:
                # 等待事件，设置超时避免阻塞
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                    await self._process_event(event)
                    self.stats['events_processed'] += 1
                except asyncio.TimeoutError:
                    continue
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Event processing error: {e}")
                self.stats['errors_count'] += 1
                await asyncio.sleep(0.1)
    
    async def _process_event(self, event: Union[TradingEvent, AlertEvent]) -> None:
        """处理单个事件"""
        try:
            if isinstance(event, TradingEvent):
                handlers = self.event_handlers.get(event.event_type, [])
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"❌ Event handler error: {e}")
            
            elif isinstance(event, AlertEvent):
                # 处理告警事件
                await self._handle_alert_event(event)
        
        except Exception as e:
            logger.error(f"❌ Failed to process event: {e}")
            self.stats['errors_count'] += 1
    
    async def _handle_alert_event(self, alert: AlertEvent) -> None:
        """处理告警事件"""
        logger.warning(f"🚨 Alert: [{alert.severity.value}] {alert.message}")
        
        # 根据严重性级别执行不同的处理逻辑
        if alert.severity == AlertSeverity.EMERGENCY:
            # 紧急告警处理
            logger.critical(f"🚨🚨 EMERGENCY ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.CRITICAL:
            # 严重告警处理
            logger.critical(f"🔥 CRITICAL ALERT: {alert.message}")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'status': 'healthy' if self.is_running else 'unhealthy',
            'queue_size': self.event_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'registered_handlers': sum(len(handlers) for handlers in self.event_handlers.values()),
            **self.get_stats()
        }


class AlertManager(MonitoringComponent):
    """告警管理器 - 管理告警生成和通知"""
    
    def __init__(self, name: str = "alert_manager", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.alert_rules: Dict[str, Dict[str, Any]] = config.get('alert_rules', {}) if config else {}
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_history: List[AlertEvent] = []
        self.max_history_size = config.get('max_history_size', 1000) if config else 1000
        self.cooldown_periods: Dict[str, datetime] = {}
    
    async def start(self) -> None:
        """启动告警管理器"""
        self.is_running = True
        self.start_time = datetime.now(timezone.utc)
        logger.info(f"🚨 {self.name} started with {len(self.alert_rules)} rules")
    
    async def stop(self) -> None:
        """停止告警管理器"""
        self.is_running = False
        logger.info(f"🛑 {self.name} stopped")
    
    async def create_alert(
        self,
        alert_id: str,
        severity: AlertSeverity,
        category: str,
        message: str,
        source: str = "",
        affected_systems: List[str] = None,
        recommended_action: str = "",
        auto_remediation: bool = False,
        tags: Dict[str, str] = None,
        details: Dict[str, Any] = None
    ) -> AlertEvent:
        """创建告警事件"""
        
        # 检查冷却期
        if alert_id in self.cooldown_periods:
            if datetime.now(timezone.utc) < self.cooldown_periods[alert_id]:
                return None  # 在冷却期内，不创建告警
        
        alert = AlertEvent(
            alert_id=alert_id,
            severity=severity,
            category=category,
            message=message,
            source=source,
            affected_systems=affected_systems or [],
            recommended_action=recommended_action,
            auto_remediation=auto_remediation,
            tags=tags or {},
            details=details or {}
        )
        
        # 添加到活跃告警
        self.active_alerts[alert_id] = alert
        
        # 添加到历史记录
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]
        
        # 设置冷却期
        cooldown_seconds = self.alert_rules.get(category, {}).get('cooldown_seconds', 300)
        self.cooldown_periods[alert_id] = datetime.now(timezone.utc) + timedelta(seconds=cooldown_seconds)
        
        self.stats['events_processed'] += 1
        logger.info(f"🚨 Created alert: {alert_id} - {message}")
        
        return alert
    
    async def resolve_alert(self, alert_id: str, resolution_message: str = "") -> bool:
        """解决告警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            del self.active_alerts[alert_id]
            
            logger.info(f"✅ Resolved alert: {alert_id} - {resolution_message}")
            return True
        return False
    
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[AlertEvent]:
        """获取活跃告警"""
        alerts = list(self.active_alerts.values())
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        return alerts
    
    def get_alert_history(self, count: int = 100) -> List[AlertEvent]:
        """获取告警历史"""
        return self.alert_history[-count:] if self.alert_history else []
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'status': 'healthy' if self.is_running else 'unhealthy',
            'active_alerts_count': len(self.active_alerts),
            'alert_history_size': len(self.alert_history),
            'alert_rules_count': len(self.alert_rules),
            **self.get_stats()
        }


class HealthMonitor(MonitoringComponent):
    """健康监控器 - 监控系统整体健康状态"""
    
    def __init__(self, name: str = "health_monitor", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.check_interval = config.get('check_interval', 60) if config else 60
        self.components: List[MonitoringComponent] = []
        self._monitoring_task: Optional[asyncio.Task] = None
        self.health_scores: Dict[str, float] = {}
        self.overall_health_score = 100.0
    
    async def start(self) -> None:
        """启动健康监控"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = datetime.now(timezone.utc)
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"💊 {self.name} started")
    
    async def stop(self) -> None:
        """停止健康监控"""
        self.is_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info(f"🛑 {self.name} stopped")
    
    def register_component(self, component: MonitoringComponent) -> None:
        """注册监控组件"""
        self.components.append(component)
        logger.info(f"📋 Registered component: {component.name}")
    
    async def _monitoring_loop(self) -> None:
        """健康监控主循环"""
        while self.is_running:
            try:
                await self._check_system_health()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Health monitoring error: {e}")
                self.stats['errors_count'] += 1
                await asyncio.sleep(1)
    
    async def _check_system_health(self) -> None:
        """检查系统健康状态"""
        try:
            component_scores = []
            
            for component in self.components:
                try:
                    health_data = await component.health_check()
                    score = 100.0 if health_data.get('status') == 'healthy' else 0.0
                    
                    # 根据具体指标调整分数
                    if 'errors_count' in health_data and health_data['errors_count'] > 0:
                        error_rate = health_data['errors_count'] / max(health_data.get('events_processed', 1), 1)
                        score = max(0, score - (error_rate * 50))
                    
                    self.health_scores[component.name] = score
                    component_scores.append(score)
                    
                except Exception as e:
                    logger.error(f"❌ Health check failed for {component.name}: {e}")
                    self.health_scores[component.name] = 0.0
                    component_scores.append(0.0)
            
            # 计算整体健康分数
            self.overall_health_score = sum(component_scores) / len(component_scores) if component_scores else 0.0
            
            self.stats['events_processed'] += 1
            
            # 记录健康检查结果
            logger.debug(f"📊 Health scores: {self.health_scores}, Overall: {self.overall_health_score:.1f}")
            
        except Exception as e:
            logger.error(f"❌ System health check failed: {e}")
            self.overall_health_score = 0.0
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'status': 'healthy' if self.overall_health_score > 70 else 'unhealthy',
            'overall_score': self.overall_health_score,
            'component_scores': self.health_scores,
            'registered_components': len(self.components),
            **self.get_stats()
        }
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """获取系统健康状态摘要"""
        return {
            'overall_score': self.overall_health_score,
            'component_scores': self.health_scores,
            'status': 'healthy' if self.overall_health_score > 70 else 'degraded' if self.overall_health_score > 30 else 'unhealthy',
            'total_components': len(self.components),
            'healthy_components': sum(1 for score in self.health_scores.values() if score > 70),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


@dataclass
class MonitoringSystemConfig:
    """监控系统配置"""
    # Kafka配置
    kafka_servers: List[str] = field(default_factory=lambda: ['localhost:9092'])
    kafka_client_id: str = "dipmaster-monitoring"
    
    # 数据收集配置
    data_collection_interval: float = 1.0
    data_buffer_size: int = 1000
    
    # 事件处理配置
    event_queue_size: int = 10000
    
    # 告警配置
    alert_history_size: int = 1000
    default_cooldown_seconds: int = 300
    
    # 健康监控配置
    health_check_interval: int = 60
    health_threshold: float = 70.0
    
    # 日志配置
    log_level: str = "INFO"
    log_directory: str = "logs/monitoring"
    
    # 仪表板配置
    dashboard_update_interval: int = 1
    dashboard_port: int = 8080


class MonitoringSystem:
    """完整的监控系统架构"""
    
    def __init__(self, config: MonitoringSystemConfig = None):
        self.config = config or MonitoringSystemConfig()
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # 初始化核心组件
        self.data_collector = DataCollector(
            config={'collection_interval': self.config.data_collection_interval,
                   'buffer_size': self.config.data_buffer_size}
        )
        
        self.event_processor = EventProcessor(
            config={'max_queue_size': self.config.event_queue_size}
        )
        
        self.alert_manager = AlertManager(
            config={'max_history_size': self.config.alert_history_size}
        )
        
        self.health_monitor = HealthMonitor(
            config={'check_interval': self.config.health_check_interval}
        )
        
        # 注册组件到健康监控
        self.health_monitor.register_component(self.data_collector)
        self.health_monitor.register_component(self.event_processor)
        self.health_monitor.register_component(self.alert_manager)
        
        # 系统统计
        self.system_stats = {
            'start_time': None,
            'uptime_seconds': 0,
            'total_events_processed': 0,
            'total_alerts_generated': 0,
            'total_errors': 0
        }
    
    async def start(self) -> None:
        """启动监控系统"""
        if self.is_running:
            logger.warning("⚠️ Monitoring system already running")
            return
        
        try:
            logger.info("🚀 Starting DipMaster Monitoring System...")
            
            self.is_running = True
            self.start_time = datetime.now(timezone.utc)
            self.system_stats['start_time'] = self.start_time.isoformat()
            
            # 启动所有核心组件
            await self.data_collector.start()
            await self.event_processor.start()
            await self.alert_manager.start()
            await self.health_monitor.start()
            
            # 创建系统启动事件
            startup_event = TradingEvent(
                event_id=f"system_startup_{int(time.time())}",
                event_type=MonitoringEventType.SYSTEM_STARTUP,
                data={'components': ['data_collector', 'event_processor', 'alert_manager', 'health_monitor']}
            )
            
            await self.event_processor.publish_event(startup_event)
            
            logger.info("✅ DipMaster Monitoring System started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring system: {e}")
            self.is_running = False
            raise
    
    async def stop(self) -> None:
        """停止监控系统"""
        if not self.is_running:
            logger.warning("⚠️ Monitoring system not running")
            return
        
        try:
            logger.info("🛑 Stopping DipMaster Monitoring System...")
            
            # 创建系统关闭事件
            shutdown_event = TradingEvent(
                event_id=f"system_shutdown_{int(time.time())}",
                event_type=MonitoringEventType.SYSTEM_SHUTDOWN,
                data={'uptime_seconds': (datetime.now(timezone.utc) - self.start_time).total_seconds()}
            )
            
            await self.event_processor.publish_event(shutdown_event)
            
            # 停止所有组件
            await self.health_monitor.stop()
            await self.alert_manager.stop()
            await self.event_processor.stop()
            await self.data_collector.stop()
            
            self.is_running = False
            
            logger.info("✅ DipMaster Monitoring System stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping monitoring system: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        if self.start_time:
            self.system_stats['uptime_seconds'] = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        # 收集各组件统计
        component_stats = {
            'data_collector': self.data_collector.get_stats(),
            'event_processor': self.event_processor.get_stats(),
            'alert_manager': self.alert_manager.get_stats(),
            'health_monitor': self.health_monitor.get_stats()
        }
        
        # 计算总计数
        self.system_stats['total_events_processed'] = sum(
            stats.get('events_processed', 0) for stats in component_stats.values()
        )
        self.system_stats['total_errors'] = sum(
            stats.get('errors_count', 0) for stats in component_stats.values()
        )
        
        return {
            'system_stats': self.system_stats,
            'component_stats': component_stats,
            'health_summary': self.health_monitor.get_system_health_summary(),
            'is_running': self.is_running
        }
    
    async def record_trading_signal(self, signal_data: Dict[str, Any]) -> None:
        """记录交易信号"""
        event = TradingEvent(
            event_id=f"signal_{signal_data.get('signal_id', int(time.time()))}",
            event_type=MonitoringEventType.SIGNAL_GENERATED,
            symbol=signal_data.get('symbol', ''),
            data=signal_data
        )
        
        await self.event_processor.publish_event(event)
    
    async def record_trade_execution(self, execution_data: Dict[str, Any]) -> None:
        """记录交易执行"""
        event = TradingEvent(
            event_id=f"exec_{execution_data.get('execution_id', int(time.time()))}",
            event_type=MonitoringEventType.EXECUTION_REPORT,
            symbol=execution_data.get('symbol', ''),
            data=execution_data
        )
        
        await self.event_processor.publish_event(event)
    
    async def create_alert(self, severity: AlertSeverity, category: str, message: str, **kwargs) -> AlertEvent:
        """创建告警"""
        alert_id = f"{category}_{int(time.time())}"
        return await self.alert_manager.create_alert(
            alert_id=alert_id,
            severity=severity,
            category=category,
            message=message,
            **kwargs
        )


# 工厂函数
def create_monitoring_system(config: MonitoringSystemConfig = None) -> MonitoringSystem:
    """创建监控系统实例"""
    return MonitoringSystem(config)


# 演示函数
async def monitoring_architecture_demo():
    """监控架构演示"""
    print("🚀 DipMaster Monitoring Architecture Demo")
    
    # 创建配置
    config = MonitoringSystemConfig(
        data_collection_interval=2.0,
        health_check_interval=30
    )
    
    # 创建监控系统
    monitoring = create_monitoring_system(config)
    
    try:
        # 启动系统
        await monitoring.start()
        print("✅ Monitoring system started")
        
        # 模拟交易信号
        signal_data = {
            'signal_id': 'sig_demo_001',
            'symbol': 'BTCUSDT',
            'signal_type': 'BUY',
            'confidence': 0.87,
            'price': 43250.50,
            'rsi': 34.2,
            'ma20_distance': -0.008
        }
        
        await monitoring.record_trading_signal(signal_data)
        print("📊 Recorded trading signal")
        
        # 模拟执行报告
        execution_data = {
            'execution_id': 'exec_demo_001',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.15,
            'price': 43245.00,
            'slippage_bps': 1.3,
            'latency_ms': 42
        }
        
        await monitoring.record_trade_execution(execution_data)
        print("⚡ Recorded execution report")
        
        # 创建测试告警
        alert = await monitoring.create_alert(
            severity=AlertSeverity.WARNING,
            category="SYSTEM_TEST",
            message="Demo alert for architecture validation",
            source="demo_system",
            recommended_action="This is a test alert, no action needed"
        )
        
        if alert:
            print(f"🚨 Created alert: {alert.alert_id}")
        
        # 等待处理
        await asyncio.sleep(3)
        
        # 获取系统状态
        status = await monitoring.get_system_status()
        print(f"📊 System uptime: {status['system_stats']['uptime_seconds']:.1f}s")
        print(f"📊 Events processed: {status['system_stats']['total_events_processed']}")
        print(f"📊 Health score: {status['health_summary']['overall_score']:.1f}")
        
        print("✅ Demo completed successfully")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    finally:
        await monitoring.stop()
        print("🛑 Monitoring system stopped")


if __name__ == "__main__":
    asyncio.run(monitoring_architecture_demo())