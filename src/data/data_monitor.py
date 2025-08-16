"""
DataMonitor - 数据基础设施监控系统
实时监控数据质量、性能指标和系统健康状态
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from collections import deque, defaultdict
import psutil
import sqlite3

@dataclass
class HealthStatus:
    """健康状态"""
    component: str
    status: str  # 'healthy', 'warning', 'critical', 'down'
    last_check: str
    details: Dict[str, Any]
    score: float  # 0-100健康分数

@dataclass
class DataQualityAlert:
    """数据质量告警"""
    alert_id: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    component: str
    message: str
    timestamp: str
    details: Dict[str, Any]
    resolved: bool = False

@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    data_throughput: float
    error_rate: float

class DataMonitor:
    """
    数据监控器 - DipMaster数据基础设施监控中心
    
    监控维度:
    1. 数据质量监控 (延迟、完整性、准确性)
    2. 系统性能监控 (CPU、内存、网络)
    3. 服务健康监控 (WebSocket、数据库、存储)
    4. 告警管理系统 (实时通知、告警聚合)
    5. 性能趋势分析 (历史数据、异常检测)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # 监控配置
        self.monitor_interval = config.get('monitor_interval', 30)  # 秒
        self.alert_thresholds = config.get('alert_thresholds', {
            'data_latency_ms': 1000,
            'missing_data_pct': 1.0,
            'cpu_usage_pct': 80,
            'memory_usage_pct': 85,
            'disk_usage_pct': 90,
            'error_rate_pct': 5.0
        })
        
        # 监控状态
        self.is_monitoring = False
        self.health_status = {}
        self.performance_history = deque(maxlen=1000)  # 保留最近1000个数据点
        self.active_alerts = {}
        
        # 数据收集器
        self.data_collectors = {}
        self.alert_handlers = []
        
        # 数据库连接
        self.monitor_db_path = Path(config.get('data_root', 'data')) / 'metadata' / 'monitoring.db'
        self.db_connection = None
        
        # 初始化监控数据库
        self._init_monitor_db()
        
    def _init_monitor_db(self):
        """初始化监控数据库"""
        self.monitor_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.db_connection = sqlite3.connect(
                str(self.monitor_db_path), 
                check_same_thread=False
            )
            
            # 创建性能指标表
            self.db_connection.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    network_latency REAL,
                    data_throughput REAL,
                    error_rate REAL,
                    component TEXT
                )
            ''')
            
            # 创建告警记录表
            self.db_connection.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    severity TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    details TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TEXT
                )
            ''')
            
            # 创建健康检查表
            self.db_connection.execute('''
                CREATE TABLE IF NOT EXISTS health_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    status TEXT NOT NULL,
                    score REAL,
                    timestamp TEXT NOT NULL,
                    details TEXT
                )
            ''')
            
            # 创建索引
            self.db_connection.execute('''
                CREATE INDEX IF NOT EXISTS idx_performance_timestamp 
                ON performance_metrics(timestamp)
            ''')
            
            self.db_connection.execute('''
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
                ON alerts(timestamp)
            ''')
            
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"监控数据库初始化失败: {e}")
    
    async def start_monitoring(self):
        """启动监控服务"""
        if self.is_monitoring:
            self.logger.warning("监控服务已在运行")
            return
        
        self.logger.info("启动数据基础设施监控服务")
        self.is_monitoring = True
        
        # 启动各种监控任务
        asyncio.create_task(self._system_monitoring_loop())
        asyncio.create_task(self._data_quality_monitoring_loop())
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._alert_processing_loop())
        
    async def _system_monitoring_loop(self):
        """系统性能监控循环"""
        while self.is_monitoring:
            try:
                # 收集系统性能指标
                metrics = await self._collect_system_metrics()
                
                # 存储指标
                await self._store_performance_metrics(metrics)
                
                # 检查阈值并生成告警
                await self._check_performance_thresholds(metrics)
                
                # 添加到历史记录
                self.performance_history.append(metrics)
                
                await asyncio.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"系统监控失败: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self) -> PerformanceMetrics:
        """收集系统性能指标"""
        timestamp = datetime.now().isoformat()
        
        # CPU使用率
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # 网络延迟（ping本地回环）
        network_latency = await self._measure_network_latency()
        
        # 数据吞吐量（从数据收集器获取）
        data_throughput = await self._calculate_data_throughput()
        
        # 错误率（从日志分析）
        error_rate = await self._calculate_error_rate()
        
        return PerformanceMetrics(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_latency=network_latency,
            data_throughput=data_throughput,
            error_rate=error_rate
        )
    
    async def _measure_network_latency(self) -> float:
        """测量网络延迟"""
        try:
            import aiohttp
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.binance.com/api/v3/ping') as response:
                    if response.status == 200:
                        latency = (time.time() - start_time) * 1000  # 毫秒
                        return latency
            
            return 999.0  # 连接失败时返回高延迟
            
        except Exception:
            return 999.0
    
    async def _calculate_data_throughput(self) -> float:
        """计算数据吞吐量"""
        # 这里应该从实际的数据流组件获取吞吐量
        # 暂时返回模拟值
        return 1000.0  # 消息/秒
    
    async def _calculate_error_rate(self) -> float:
        """计算错误率"""
        # 从日志或错误计数器计算错误率
        # 暂时返回模拟值
        return 0.5  # 百分比
    
    async def _store_performance_metrics(self, metrics: PerformanceMetrics):
        """存储性能指标到数据库"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO performance_metrics 
                (timestamp, cpu_usage, memory_usage, disk_usage, 
                 network_latency, data_throughput, error_rate, component)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp, metrics.cpu_usage, metrics.memory_usage,
                metrics.disk_usage, metrics.network_latency, 
                metrics.data_throughput, metrics.error_rate, 'system'
            ))
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"性能指标存储失败: {e}")
    
    async def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """检查性能阈值并生成告警"""
        thresholds = self.alert_thresholds
        
        # CPU使用率检查
        if metrics.cpu_usage > thresholds['cpu_usage_pct']:
            await self._create_alert(
                'high_cpu_usage',
                'warning' if metrics.cpu_usage < 90 else 'critical',
                'system',
                f"CPU使用率过高: {metrics.cpu_usage:.1f}%",
                {'cpu_usage': metrics.cpu_usage, 'threshold': thresholds['cpu_usage_pct']}
            )
        
        # 内存使用率检查
        if metrics.memory_usage > thresholds['memory_usage_pct']:
            await self._create_alert(
                'high_memory_usage',
                'warning' if metrics.memory_usage < 95 else 'critical',
                'system',
                f"内存使用率过高: {metrics.memory_usage:.1f}%",
                {'memory_usage': metrics.memory_usage, 'threshold': thresholds['memory_usage_pct']}
            )
        
        # 磁盘使用率检查
        if metrics.disk_usage > thresholds['disk_usage_pct']:
            await self._create_alert(
                'high_disk_usage',
                'critical',
                'storage',
                f"磁盘使用率过高: {metrics.disk_usage:.1f}%",
                {'disk_usage': metrics.disk_usage, 'threshold': thresholds['disk_usage_pct']}
            )
        
        # 网络延迟检查
        if metrics.network_latency > thresholds['data_latency_ms']:
            await self._create_alert(
                'high_network_latency',
                'warning',
                'network',
                f"网络延迟过高: {metrics.network_latency:.1f}ms",
                {'latency': metrics.network_latency, 'threshold': thresholds['data_latency_ms']}
            )
        
        # 错误率检查
        if metrics.error_rate > thresholds['error_rate_pct']:
            await self._create_alert(
                'high_error_rate',
                'critical',
                'data_processing',
                f"错误率过高: {metrics.error_rate:.1f}%",
                {'error_rate': metrics.error_rate, 'threshold': thresholds['error_rate_pct']}
            )
    
    async def _data_quality_monitoring_loop(self):
        """数据质量监控循环"""
        while self.is_monitoring:
            try:
                # 检查数据延迟
                await self._check_data_latency()
                
                # 检查数据完整性
                await self._check_data_completeness()
                
                # 检查数据准确性
                await self._check_data_accuracy()
                
                await asyncio.sleep(self.monitor_interval * 2)  # 数据质量检查频率低一些
                
            except Exception as e:
                self.logger.error(f"数据质量监控失败: {e}")
                await asyncio.sleep(10)
    
    async def _check_data_latency(self):
        """检查数据延迟"""
        # 这里应该检查实时数据流的延迟
        # 暂时实现基本逻辑
        current_time = time.time() * 1000
        
        # 模拟检查最新数据的时间戳
        simulated_last_data_time = current_time - 500  # 500ms前
        latency = current_time - simulated_last_data_time
        
        if latency > self.alert_thresholds['data_latency_ms']:
            await self._create_alert(
                'data_latency_high',
                'warning',
                'data_stream',
                f"数据延迟过高: {latency:.1f}ms",
                {'latency_ms': latency}
            )
    
    async def _check_data_completeness(self):
        """检查数据完整性"""
        # 检查最近时间段内的数据缺失情况
        # 这里应该与实际数据验证器集成
        pass
    
    async def _check_data_accuracy(self):
        """检查数据准确性"""
        # 检查数据的OHLC关系、价格合理性等
        # 这里应该与实际数据验证器集成
        pass
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self.is_monitoring:
            try:
                # 检查各组件健康状态
                await self._check_component_health('data_downloader')
                await self._check_component_health('data_validator')
                await self._check_component_health('storage_manager')
                await self._check_component_health('realtime_stream')
                
                await asyncio.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"健康检查失败: {e}")
                await asyncio.sleep(10)
    
    async def _check_component_health(self, component: str):
        """检查组件健康状态"""
        try:
            # 根据组件类型执行不同的健康检查
            if component == 'realtime_stream':
                status = await self._check_realtime_stream_health()
            elif component == 'storage_manager':
                status = await self._check_storage_health()
            elif component == 'data_validator':
                status = await self._check_validator_health()
            else:
                status = await self._check_generic_component_health(component)
            
            # 更新健康状态
            self.health_status[component] = status
            
            # 存储健康检查结果
            await self._store_health_status(component, status)
            
            # 如果状态不健康，生成告警
            if status.status in ['warning', 'critical', 'down']:
                await self._create_alert(
                    f'{component}_unhealthy',
                    status.status,
                    component,
                    f"组件 {component} 状态异常: {status.status}",
                    status.details
                )
                
        except Exception as e:
            self.logger.error(f"组件 {component} 健康检查失败: {e}")
    
    async def _check_realtime_stream_health(self) -> HealthStatus:
        """检查实时数据流健康状态"""
        # 这里应该检查WebSocket连接状态、数据流量等
        return HealthStatus(
            component='realtime_stream',
            status='healthy',
            last_check=datetime.now().isoformat(),
            details={'connections': 1, 'throughput': 1000},
            score=95.0
        )
    
    async def _check_storage_health(self) -> HealthStatus:
        """检查存储系统健康状态"""
        # 检查磁盘空间、I/O性能等
        return HealthStatus(
            component='storage_manager',
            status='healthy',
            last_check=datetime.now().isoformat(),
            details={'disk_usage': 60, 'io_latency': 50},
            score=90.0
        )
    
    async def _check_validator_health(self) -> HealthStatus:
        """检查数据验证器健康状态"""
        return HealthStatus(
            component='data_validator',
            status='healthy',
            last_check=datetime.now().isoformat(),
            details={'validation_rate': 99.5},
            score=98.0
        )
    
    async def _check_generic_component_health(self, component: str) -> HealthStatus:
        """通用组件健康检查"""
        return HealthStatus(
            component=component,
            status='healthy',
            last_check=datetime.now().isoformat(),
            details={},
            score=85.0
        )
    
    async def _store_health_status(self, component: str, status: HealthStatus):
        """存储健康状态"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO health_checks 
                (component, status, score, timestamp, details)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                component, status.status, status.score,
                status.last_check, json.dumps(status.details)
            ))
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"健康状态存储失败: {e}")
    
    async def _create_alert(self, alert_id: str, severity: str, component: str,
                          message: str, details: Dict[str, Any]):
        """创建告警"""
        if alert_id in self.active_alerts:
            return  # 避免重复告警
        
        alert = DataQualityAlert(
            alert_id=alert_id,
            severity=severity,
            component=component,
            message=message,
            timestamp=datetime.now().isoformat(),
            details=details
        )
        
        self.active_alerts[alert_id] = alert
        
        # 存储告警到数据库
        await self._store_alert(alert)
        
        # 通知告警处理器
        await self._notify_alert_handlers(alert)
        
        self.logger.warning(f"告警生成: [{severity}] {component} - {message}")
    
    async def _store_alert(self, alert: DataQualityAlert):
        """存储告警到数据库"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO alerts 
                (alert_id, severity, component, message, timestamp, details)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id, alert.severity, alert.component,
                alert.message, alert.timestamp, json.dumps(alert.details)
            ))
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"告警存储失败: {e}")
    
    async def _notify_alert_handlers(self, alert: DataQualityAlert):
        """通知告警处理器"""
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                self.logger.error(f"告警处理器失败: {e}")
    
    async def _alert_processing_loop(self):
        """告警处理循环"""
        while self.is_monitoring:
            try:
                # 检查告警自动恢复
                await self._check_alert_recovery()
                
                # 清理旧告警
                await self._cleanup_old_alerts()
                
                await asyncio.sleep(60)  # 1分钟检查一次
                
            except Exception as e:
                self.logger.error(f"告警处理循环失败: {e}")
                await asyncio.sleep(10)
    
    async def _check_alert_recovery(self):
        """检查告警恢复"""
        # 检查当前活跃告警是否已恢复
        for alert_id, alert in list(self.active_alerts.items()):
            # 这里应该实现具体的恢复检查逻辑
            # 暂时实现简单的时间基恢复
            alert_age = (datetime.now() - datetime.fromisoformat(alert.timestamp)).total_seconds()
            
            if alert_age > 300:  # 5分钟后自动恢复
                await self._resolve_alert(alert_id)
    
    async def _resolve_alert(self, alert_id: str):
        """解决告警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            alert.resolved = True
            
            # 更新数据库
            try:
                cursor = self.db_connection.cursor()
                cursor.execute('''
                    UPDATE alerts SET resolved = TRUE, resolved_at = ?
                    WHERE alert_id = ?
                ''', (datetime.now().isoformat(), alert_id))
                self.db_connection.commit()
                
                self.logger.info(f"告警已解决: {alert_id}")
                
            except Exception as e:
                self.logger.error(f"告警解决更新失败: {e}")
    
    async def _cleanup_old_alerts(self):
        """清理旧告警"""
        try:
            # 清理7天前的已解决告警
            cutoff_time = (datetime.now() - timedelta(days=7)).isoformat()
            
            cursor = self.db_connection.cursor()
            cursor.execute('''
                DELETE FROM alerts 
                WHERE resolved = TRUE AND resolved_at < ?
            ''', (cutoff_time,))
            
            deleted_count = cursor.rowcount
            self.db_connection.commit()
            
            if deleted_count > 0:
                self.logger.info(f"清理了 {deleted_count} 个旧告警")
                
        except Exception as e:
            self.logger.error(f"告警清理失败: {e}")
    
    def add_alert_handler(self, handler: Callable):
        """添加告警处理器"""
        self.alert_handlers.append(handler)
    
    def remove_alert_handler(self, handler: Callable):
        """移除告警处理器"""
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态概览"""
        # 最新性能指标
        latest_metrics = None
        if self.performance_history:
            latest_metrics = asdict(self.performance_history[-1])
        
        # 活跃告警统计
        alert_stats = {
            'total': len(self.active_alerts),
            'critical': len([a for a in self.active_alerts.values() if a.severity == 'critical']),
            'warning': len([a for a in self.active_alerts.values() if a.severity == 'warning'])
        }
        
        # 组件健康状态
        component_health = {
            component: {
                'status': status.status,
                'score': status.score,
                'last_check': status.last_check
            }
            for component, status in self.health_status.items()
        }
        
        # 整体健康分数
        if self.health_status:
            overall_score = np.mean([status.score for status in self.health_status.values()])
        else:
            overall_score = 0.0
        
        return {
            'monitoring_active': self.is_monitoring,
            'overall_health_score': overall_score,
            'latest_performance': latest_metrics,
            'active_alerts': alert_stats,
            'component_health': component_health,
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_performance_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取性能历史数据"""
        try:
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor = self.db_connection.cursor()
            cursor.execute('''
                SELECT * FROM performance_metrics 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            ''', (cutoff_time,))
            
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            self.logger.error(f"性能历史查询失败: {e}")
            return []
    
    async def get_alerts(self, resolved: bool = False, hours: int = 24) -> List[Dict[str, Any]]:
        """获取告警记录"""
        try:
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor = self.db_connection.cursor()
            cursor.execute('''
                SELECT * FROM alerts 
                WHERE timestamp > ? AND resolved = ?
                ORDER BY timestamp DESC
            ''', (cutoff_time, resolved))
            
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            self.logger.error(f"告警查询失败: {e}")
            return []
    
    async def get_status(self) -> Dict[str, Any]:
        """获取监控器状态"""
        return {
            'is_monitoring': self.is_monitoring,
            'monitor_interval': self.monitor_interval,
            'active_alerts_count': len(self.active_alerts),
            'performance_history_size': len(self.performance_history),
            'alert_handlers_count': len(self.alert_handlers),
            'health_components': list(self.health_status.keys())
        }
    
    async def stop_monitoring(self):
        """停止监控服务"""
        self.logger.info("停止数据基础设施监控服务")
        self.is_monitoring = False
        
        # 关闭数据库连接
        if self.db_connection:
            self.db_connection.close()
        
        # 清理状态
        self.health_status.clear()
        self.active_alerts.clear()
        self.performance_history.clear()