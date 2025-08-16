"""
监控和健康检查系统
API性能监控、错误追踪告警、访问日志记录、服务健康检查
"""

import asyncio
import time
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import structlog

from .config import MonitoringConfig

logger = structlog.get_logger(__name__)

@dataclass
class MetricPoint:
    """指标数据点"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}

@dataclass
class PerformanceMetrics:
    """性能指标"""
    response_time_p50: float
    response_time_p90: float
    response_time_p99: float
    request_rate: float
    error_rate: float
    active_connections: int

@dataclass
class SystemMetrics:
    """系统指标"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int

@dataclass
class DatabaseMetrics:
    """数据库指标"""
    connection_count: int
    query_count: int
    avg_query_time: float
    slow_queries: int
    cache_hit_rate: float

@dataclass
class KafkaMetrics:
    """Kafka指标"""
    consumer_lag: Dict[str, int]
    message_rate: Dict[str, float]
    error_count: Dict[str, int]
    partition_count: Dict[str, int]

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
    
    def record_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """记录计数器指标"""
        key = self._make_key(name, labels)
        self.counters[key] += value
        
        metric_point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=self.counters[key],
            labels=labels
        )
        self.metrics[name].append(metric_point)
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """记录仪表指标"""
        key = self._make_key(name, labels)
        self.gauges[key] = value
        
        metric_point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels
        )
        self.metrics[name].append(metric_point)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """记录直方图指标"""
        key = self._make_key(name, labels)
        self.histograms[key].append(value)
        
        # 保持最近1000个值
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
        
        metric_point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels
        )
        self.metrics[name].append(metric_point)
    
    def get_percentile(self, name: str, percentile: float, labels: Dict[str, str] = None) -> float:
        """获取百分位数"""
        key = self._make_key(name, labels)
        values = self.histograms.get(key, [])
        
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        
        return sorted_values[index]
    
    def get_counter_value(self, name: str, labels: Dict[str, str] = None) -> int:
        """获取计数器值"""
        key = self._make_key(name, labels)
        return self.counters.get(key, 0)
    
    def get_gauge_value(self, name: str, labels: Dict[str, str] = None) -> float:
        """获取仪表值"""
        key = self._make_key(name, labels)
        return self.gauges.get(key, 0.0)
    
    def get_recent_metrics(self, name: str, minutes: int = 5) -> List[MetricPoint]:
        """获取最近N分钟的指标"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_metrics = []
        
        for metric in self.metrics.get(name, []):
            if metric.timestamp >= cutoff_time:
                recent_metrics.append(metric)
        
        return recent_metrics
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """生成指标键"""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.request_start_times: Dict[str, float] = {}
    
    def start_request(self, request_id: str, endpoint: str):
        """开始请求计时"""
        self.request_start_times[request_id] = time.time()
        self.metrics.record_counter("http_requests_total", labels={"endpoint": endpoint})
    
    def end_request(self, request_id: str, endpoint: str, status_code: int):
        """结束请求计时"""
        if request_id in self.request_start_times:
            duration = time.time() - self.request_start_times[request_id]
            del self.request_start_times[request_id]
            
            # 记录响应时间
            self.metrics.record_histogram(
                "http_request_duration_seconds",
                duration,
                labels={"endpoint": endpoint, "status_code": str(status_code)}
            )
            
            # 记录错误率
            if status_code >= 400:
                self.metrics.record_counter(
                    "http_requests_errors_total",
                    labels={"endpoint": endpoint, "status_code": str(status_code)}
                )
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取性能指标"""
        # 响应时间百分位数
        p50 = self.metrics.get_percentile("http_request_duration_seconds", 50)
        p90 = self.metrics.get_percentile("http_request_duration_seconds", 90) 
        p99 = self.metrics.get_percentile("http_request_duration_seconds", 99)
        
        # 请求率（最近5分钟）
        recent_requests = self.metrics.get_recent_metrics("http_requests_total", 5)
        request_rate = len(recent_requests) / 5 / 60  # 每秒请求数
        
        # 错误率
        total_requests = self.metrics.get_counter_value("http_requests_total")
        error_requests = self.metrics.get_counter_value("http_requests_errors_total")
        error_rate = error_requests / max(total_requests, 1)
        
        # 活跃连接数
        active_connections = self.metrics.get_gauge_value("websocket_connections_active")
        
        return PerformanceMetrics(
            response_time_p50=p50,
            response_time_p90=p90,
            response_time_p99=p99,
            request_rate=request_rate,
            error_rate=error_rate,
            active_connections=int(active_connections)
        )

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    async def collect_system_metrics(self):
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.record_gauge("system_cpu_usage_percent", cpu_percent)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            self.metrics.record_gauge("system_memory_usage_percent", memory.percent)
            self.metrics.record_gauge("system_memory_used_bytes", memory.used)
            self.metrics.record_gauge("system_memory_available_bytes", memory.available)
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.metrics.record_gauge("system_disk_usage_percent", disk_percent)
            self.metrics.record_gauge("system_disk_used_bytes", disk.used)
            self.metrics.record_gauge("system_disk_free_bytes", disk.free)
            
            # 网络IO
            network = psutil.net_io_counters()
            self.metrics.record_counter("system_network_bytes_sent_total", network.bytes_sent)
            self.metrics.record_counter("system_network_bytes_recv_total", network.bytes_recv)
            
            # 进程数
            process_count = len(psutil.pids())
            self.metrics.record_gauge("system_processes_count", process_count)
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
    
    def get_system_metrics(self) -> SystemMetrics:
        """获取系统指标"""
        return SystemMetrics(
            cpu_usage=self.metrics.get_gauge_value("system_cpu_usage_percent"),
            memory_usage=self.metrics.get_gauge_value("system_memory_usage_percent"),
            disk_usage=self.metrics.get_gauge_value("system_disk_usage_percent"),
            network_io={
                "bytes_sent": self.metrics.get_counter_value("system_network_bytes_sent_total"),
                "bytes_recv": self.metrics.get_counter_value("system_network_bytes_recv_total")
            },
            process_count=int(self.metrics.get_gauge_value("system_processes_count"))
        )

class AlertManager:
    """告警管理器"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.thresholds = config.thresholds
        self.alerts: List[Dict[str, Any]] = []
        self.alert_states: Dict[str, bool] = {}  # 防止重复告警
    
    async def check_thresholds(self, metrics: Dict[str, Any]):
        """检查指标阈值"""
        try:
            current_time = datetime.utcnow()
            
            # 检查API响应时间
            if "performance" in metrics:
                perf = metrics["performance"]
                await self._check_threshold(
                    "api_response_time",
                    perf.response_time_p99,
                    "API响应时间P99过高",
                    current_time
                )
            
            # 检查系统资源
            if "system" in metrics:
                sys_metrics = metrics["system"]
                
                await self._check_threshold(
                    "cpu_usage",
                    sys_metrics.cpu_usage / 100,
                    "CPU使用率过高",
                    current_time
                )
                
                await self._check_threshold(
                    "memory_usage",
                    sys_metrics.memory_usage / 100,
                    "内存使用率过高",
                    current_time
                )
            
            # 检查Kafka消费延迟
            if "kafka" in metrics:
                kafka_metrics = metrics["kafka"]
                for topic, lag in kafka_metrics.consumer_lag.items():
                    await self._check_threshold(
                        "kafka_lag",
                        lag,
                        f"Kafka主题{topic}消费延迟过高",
                        current_time
                    )
            
        except Exception as e:
            logger.error(f"检查告警阈值失败: {e}")
    
    async def _check_threshold(self, metric_name: str, current_value: float,
                             alert_message: str, timestamp: datetime):
        """检查单个指标阈值"""
        if metric_name not in self.thresholds:
            return
        
        threshold_config = self.thresholds[metric_name]
        warning_threshold = threshold_config.get("warning")
        critical_threshold = threshold_config.get("critical")
        
        alert_key = f"{metric_name}_{current_value}"
        severity = None
        
        if critical_threshold and current_value >= critical_threshold:
            severity = "CRITICAL"
        elif warning_threshold and current_value >= warning_threshold:
            severity = "WARNING"
        
        if severity:
            # 防止重复告警（同一指标1小时内只告警一次）
            if alert_key not in self.alert_states or \
               (timestamp - self.alert_states.get(alert_key + "_time", datetime.min)).seconds > 3600:
                
                alert = {
                    "alert_id": f"{metric_name}_{int(timestamp.timestamp())}",
                    "timestamp": timestamp,
                    "severity": severity,
                    "metric": metric_name,
                    "current_value": current_value,
                    "threshold": critical_threshold if severity == "CRITICAL" else warning_threshold,
                    "message": alert_message,
                    "resolved": False
                }
                
                self.alerts.append(alert)
                self.alert_states[alert_key] = True
                self.alert_states[alert_key + "_time"] = timestamp
                
                logger.warning(f"触发{severity}告警: {alert_message}, 当前值: {current_value}")
        else:
            # 清除告警状态
            if alert_key in self.alert_states:
                del self.alert_states[alert_key]
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        return [alert for alert in self.alerts if not alert["resolved"]]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        for alert in self.alerts:
            if alert["alert_id"] == alert_id:
                alert["resolved"] = True
                alert["resolved_at"] = datetime.utcnow()
                return True
        return False

class HealthChecker:
    """健康检查器"""
    
    def __init__(self, db_manager, cache_manager, websocket_manager):
        self.db_manager = db_manager
        self.cache_manager = cache_manager
        self.websocket_manager = websocket_manager
    
    async def check_all_components(self) -> Dict[str, Any]:
        """检查所有组件健康状态"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "components": {},
            "overall_status": "healthy"
        }
        
        # 检查数据库
        try:
            db_health = await self.db_manager.get_health_status()
            health_status["components"]["database"] = db_health
            
            if db_health.get("status") != "healthy":
                health_status["overall_status"] = "degraded"
                
        except Exception as e:
            health_status["components"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "unhealthy"
        
        # 检查缓存
        try:
            cache_health = await self.cache_manager.get_health_status()
            health_status["components"]["cache"] = cache_health
            
            if cache_health.get("status") != "healthy":
                health_status["overall_status"] = "degraded"
                
        except Exception as e:
            health_status["components"]["cache"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "unhealthy"
        
        # 检查WebSocket
        try:
            ws_stats = await self.websocket_manager.get_connection_stats()
            health_status["components"]["websocket"] = {
                "status": "healthy",
                "stats": ws_stats
            }
            
        except Exception as e:
            health_status["components"]["websocket"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # 检查系统资源
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            system_health = {
                "status": "healthy",
                "cpu_usage": cpu_percent,
                "memory_usage": memory_percent
            }
            
            if cpu_percent > 90 or memory_percent > 90:
                system_health["status"] = "degraded"
                health_status["overall_status"] = "degraded"
            
            health_status["components"]["system"] = system_health
            
        except Exception as e:
            health_status["components"]["system"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        return health_status

class MonitoringService:
    """监控服务主类"""
    
    def __init__(self, db_manager, cache_manager, websocket_manager=None):
        self.config = MonitoringConfig()
        self.db_manager = db_manager
        self.cache_manager = cache_manager
        self.websocket_manager = websocket_manager
        
        # 初始化监控组件
        self.metrics_collector = MetricsCollector()
        self.performance_monitor = PerformanceMonitor(self.metrics_collector)
        self.system_monitor = SystemMonitor(self.metrics_collector)
        self.alert_manager = AlertManager(self.config)
        self.health_checker = HealthChecker(db_manager, cache_manager, websocket_manager)
        
        self._running = False
        self._monitoring_task = None
    
    async def initialize(self):
        """初始化监控服务"""
        try:
            logger.info("监控服务初始化完成")
        except Exception as e:
            logger.error(f"监控服务初始化失败: {e}")
            raise
    
    async def start_monitoring(self):
        """启动监控任务"""
        self._running = True
        
        async def monitoring_loop():
            while self._running:
                try:
                    # 收集系统指标
                    await self.system_monitor.collect_system_metrics()
                    
                    # 收集所有指标
                    all_metrics = await self.collect_all_metrics()
                    
                    # 检查告警阈值
                    await self.alert_manager.check_thresholds(all_metrics)
                    
                    # 等待下一次收集
                    await asyncio.sleep(self.config.health_check_interval)
                    
                except Exception as e:
                    logger.error(f"监控循环异常: {e}")
                    await asyncio.sleep(self.config.health_check_interval)
        
        self._monitoring_task = asyncio.create_task(monitoring_loop())
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """收集所有指标"""
        metrics = {}
        
        try:
            # 性能指标
            metrics["performance"] = self.performance_monitor.get_performance_metrics()
            
            # 系统指标
            metrics["system"] = self.system_monitor.get_system_metrics()
            
            # 数据库指标
            if self.db_manager:
                db_health = await self.db_manager.get_health_status()
                metrics["database"] = db_health
            
            # 缓存指标
            if self.cache_manager:
                cache_stats = await self.cache_manager.get_cache_stats()
                metrics["cache"] = cache_stats
            
            # WebSocket指标
            if self.websocket_manager:
                ws_stats = await self.websocket_manager.get_connection_stats()
                metrics["websocket"] = ws_stats
            
            return metrics
            
        except Exception as e:
            logger.error(f"收集指标失败: {e}")
            return {}
    
    async def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return await self.health_checker.check_all_components()
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        try:
            all_metrics = await self.collect_all_metrics()
            active_alerts = self.alert_manager.get_active_alerts()
            
            summary = {
                "timestamp": datetime.utcnow(),
                "metrics": all_metrics,
                "alerts": {
                    "active_count": len(active_alerts),
                    "critical_count": len([a for a in active_alerts if a["severity"] == "CRITICAL"]),
                    "warning_count": len([a for a in active_alerts if a["severity"] == "WARNING"]),
                    "recent_alerts": active_alerts[-10:]  # 最近10个告警
                },
                "performance": {
                    "requests_per_minute": self.metrics_collector.get_counter_value("http_requests_total"),
                    "error_rate": self.metrics_collector.get_counter_value("http_requests_errors_total"),
                    "avg_response_time": self.metrics_collector.get_percentile("http_request_duration_seconds", 50)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"获取指标摘要失败: {e}")
            return {"error": str(e)}
    
    # 业务指标记录方法
    
    async def update_kafka_metrics(self, topic: str, message_count: int):
        """更新Kafka指标"""
        self.metrics_collector.record_counter(
            "kafka_messages_consumed_total",
            message_count,
            labels={"topic": topic}
        )
    
    async def update_exec_metrics(self, exec_count: int):
        """更新执行报告指标"""
        self.metrics_collector.record_counter("exec_reports_processed_total", exec_count)
    
    async def update_risk_metrics(self, risk_count: int):
        """更新风险指标"""
        self.metrics_collector.record_counter("risk_metrics_processed_total", risk_count)
    
    async def update_alert_metrics(self, alerts: List[Dict[str, Any]]):
        """更新告警指标"""
        for alert in alerts:
            severity = alert.get("severity", "INFO")
            self.metrics_collector.record_counter(
                "alerts_generated_total",
                labels={"severity": severity}
            )
    
    async def update_strategy_metrics(self, performance_events: List[Dict[str, Any]]):
        """更新策略指标"""
        for event in performance_events:
            strategy_id = event.get("strategy_id", "unknown")
            self.metrics_collector.record_counter(
                "strategy_performance_updates_total",
                labels={"strategy_id": strategy_id}
            )
    
    async def record_error(self, component: str, error_message: str):
        """记录错误"""
        self.metrics_collector.record_counter(
            "errors_total",
            labels={"component": component}
        )
        
        logger.error(f"组件错误 [{component}]: {error_message}")
    
    async def shutdown(self):
        """关闭监控服务"""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("监控服务关闭完成")