#!/usr/bin/env python3
"""
DipMaster Trading System - Data Pipeline Monitor
数据流管道性能监控和质量验证系统

Author: DipMaster Development Team
Date: 2025-08-16
Version: 4.0.0
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sqlite3

logger = logging.getLogger(__name__)

class DataQualityStatus(str, Enum):
    """数据质量状态"""
    EXCELLENT = "EXCELLENT"  # >99% quality
    GOOD = "GOOD"           # 95-99% quality  
    WARNING = "WARNING"     # 90-95% quality
    POOR = "POOR"          # <90% quality
    CRITICAL = "CRITICAL"   # <80% quality

class PipelineStage(str, Enum):
    """管道阶段"""
    INGESTION = "INGESTION"     # 数据摄取
    PROCESSING = "PROCESSING"   # 数据处理
    FEATURE_ENG = "FEATURE_ENG" # 特征工程
    VALIDATION = "VALIDATION"   # 数据验证
    STORAGE = "STORAGE"        # 数据存储

@dataclass
class DataQualityMetrics:
    """数据质量指标"""
    timestamp: datetime
    symbol: str
    timeframe: str
    
    # 完整性指标
    total_records: int
    missing_records: int
    completeness_ratio: float
    
    # 准确性指标
    price_anomalies: int
    volume_anomalies: int
    timestamp_gaps: int
    
    # 一致性指标
    ohlc_consistency: float
    volume_consistency: float
    
    # 及时性指标
    data_latency_ms: float
    processing_time_ms: float
    
    # 整体质量分数
    quality_score: float
    status: DataQualityStatus

@dataclass
class PipelinePerformanceMetrics:
    """管道性能指标"""
    timestamp: datetime
    stage: PipelineStage
    
    # 性能指标
    throughput_records_per_second: float
    latency_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    
    # 错误指标
    error_count: int
    success_rate: float
    
    # 资源使用
    disk_io_mb_per_s: float
    network_io_mb_per_s: float

class DataPipelineMonitor:
    """数据管道监控器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring_db = Path("data/monitoring/pipeline_monitor.db")
        self.monitoring_db.parent.mkdir(parents=True, exist_ok=True)
        
        self.quality_thresholds = config.get('quality_thresholds', {
            'completeness_min': 0.95,      # 95% minimum completeness
            'latency_max_ms': 1000,        # 1 second max latency
            'error_rate_max': 0.05,        # 5% max error rate
            'anomaly_threshold': 0.02       # 2% max anomalies
        })
        
        self.performance_targets = config.get('performance_targets', {
            'throughput_min_rps': 100,     # 100 records/second minimum
            'cpu_usage_max': 0.8,          # 80% max CPU usage
            'memory_usage_max_mb': 2048,   # 2GB max memory
            'success_rate_min': 0.95       # 95% min success rate
        })
        
        self.alert_channels = config.get('alert_channels', ['log'])
        self.monitoring_interval = config.get('monitoring_interval', 30)  # seconds
        
        self._init_database()
        self._start_background_monitoring()
        
    def _init_database(self):
        """初始化监控数据库"""
        with sqlite3.connect(self.monitoring_db) as conn:
            # 数据质量表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    total_records INTEGER,
                    missing_records INTEGER,
                    completeness_ratio REAL,
                    price_anomalies INTEGER,
                    volume_anomalies INTEGER,
                    timestamp_gaps INTEGER,
                    ohlc_consistency REAL,
                    volume_consistency REAL,
                    data_latency_ms REAL,
                    processing_time_ms REAL,
                    quality_score REAL,
                    status TEXT
                )
            """)
            
            # 管道性能表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    throughput_rps REAL,
                    latency_ms REAL,
                    cpu_usage_percent REAL,
                    memory_usage_mb REAL,
                    error_count INTEGER,
                    success_rate REAL,
                    disk_io_mb_per_s REAL,
                    network_io_mb_per_s REAL
                )
            """)
            
            # 告警表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            conn.commit()
    
    def _start_background_monitoring(self):
        """启动后台监控"""
        import threading
        self.monitoring_thread = threading.Thread(
            target=self._background_monitor_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Data pipeline background monitoring started")
    
    def _background_monitor_loop(self):
        """后台监控循环"""
        while True:
            try:
                # 执行定期监控检查
                self._periodic_health_check()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def monitor_data_quality(self, 
                           data: pd.DataFrame, 
                           symbol: str, 
                           timeframe: str,
                           stage: PipelineStage = PipelineStage.VALIDATION) -> DataQualityMetrics:
        """监控数据质量"""
        start_time = time.time()
        
        # 计算完整性指标
        total_records = len(data)
        missing_records = data.isnull().sum().sum()
        completeness_ratio = 1 - (missing_records / (total_records * len(data.columns))) if total_records > 0 else 0
        
        # 计算准确性指标
        price_anomalies = self._detect_price_anomalies(data)
        volume_anomalies = self._detect_volume_anomalies(data)
        timestamp_gaps = self._detect_timestamp_gaps(data)
        
        # 计算一致性指标
        ohlc_consistency = self._check_ohlc_consistency(data)
        volume_consistency = self._check_volume_consistency(data)
        
        # 计算处理时间
        processing_time_ms = (time.time() - start_time) * 1000
        
        # 数据延迟（假设最新数据的时间戳延迟）
        data_latency_ms = self._calculate_data_latency(data)
        
        # 计算整体质量分数
        quality_score = self._calculate_quality_score({
            'completeness_ratio': completeness_ratio,
            'price_anomaly_rate': price_anomalies / total_records if total_records > 0 else 0,
            'volume_anomaly_rate': volume_anomalies / total_records if total_records > 0 else 0,
            'timestamp_gap_rate': timestamp_gaps / total_records if total_records > 0 else 0,
            'ohlc_consistency': ohlc_consistency,
            'volume_consistency': volume_consistency
        })
        
        # 确定质量状态
        status = self._determine_quality_status(quality_score)
        
        # 创建质量指标
        metrics = DataQualityMetrics(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            total_records=total_records,
            missing_records=missing_records,
            completeness_ratio=completeness_ratio,
            price_anomalies=price_anomalies,
            volume_anomalies=volume_anomalies,
            timestamp_gaps=timestamp_gaps,
            ohlc_consistency=ohlc_consistency,
            volume_consistency=volume_consistency,
            data_latency_ms=data_latency_ms,
            processing_time_ms=processing_time_ms,
            quality_score=quality_score,
            status=status
        )
        
        # 保存到数据库
        self._save_quality_metrics(metrics)
        
        # 检查是否需要告警
        self._check_quality_alerts(metrics)
        
        logger.info(f"Data quality monitored: {symbol}_{timeframe} - Score: {quality_score:.3f}, Status: {status}")
        return metrics
    
    def monitor_pipeline_performance(self, 
                                   stage: PipelineStage,
                                   start_time: float,
                                   record_count: int,
                                   error_count: int = 0) -> PipelinePerformanceMetrics:
        """监控管道性能"""
        end_time = time.time()
        processing_duration = end_time - start_time
        
        # 计算性能指标
        throughput_rps = record_count / processing_duration if processing_duration > 0 else 0
        latency_ms = processing_duration * 1000
        success_rate = (record_count - error_count) / record_count if record_count > 0 else 1.0
        
        # 获取系统资源使用
        cpu_usage, memory_usage = self._get_system_resources()
        
        # 获取I/O指标
        disk_io, network_io = self._get_io_metrics()
        
        # 创建性能指标
        metrics = PipelinePerformanceMetrics(
            timestamp=datetime.now(),
            stage=stage,
            throughput_records_per_second=throughput_rps,
            latency_ms=latency_ms,
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_usage,
            error_count=error_count,
            success_rate=success_rate,
            disk_io_mb_per_s=disk_io,
            network_io_mb_per_s=network_io
        )
        
        # 保存到数据库
        self._save_performance_metrics(metrics)
        
        # 检查性能告警
        self._check_performance_alerts(metrics)
        
        logger.debug(f"Pipeline performance monitored: {stage} - Throughput: {throughput_rps:.1f} rps, Latency: {latency_ms:.1f}ms")
        return metrics
    
    def _detect_price_anomalies(self, data: pd.DataFrame) -> int:
        """检测价格异常"""
        if 'close' not in data.columns:
            return 0
        
        price_data = data['close'].dropna()
        if len(price_data) < 2:
            return 0
        
        # 计算价格变化率
        price_changes = price_data.pct_change().dropna()
        
        # 使用3σ规则检测异常
        mean_change = price_changes.mean()
        std_change = price_changes.std()
        threshold = 3 * std_change
        
        anomalies = abs(price_changes - mean_change) > threshold
        return anomalies.sum()
    
    def _detect_volume_anomalies(self, data: pd.DataFrame) -> int:
        """检测成交量异常"""
        if 'volume' not in data.columns:
            return 0
        
        volume_data = data['volume'].dropna()
        if len(volume_data) < 2:
            return 0
        
        # 使用IQR方法检测异常
        q1 = volume_data.quantile(0.25)
        q3 = volume_data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        
        anomalies = (volume_data < lower_bound) | (volume_data > upper_bound)
        return anomalies.sum()
    
    def _detect_timestamp_gaps(self, data: pd.DataFrame) -> int:
        """检测时间戳间隙"""
        if 'timestamp' not in data.columns and data.index.name != 'timestamp':
            return 0
        
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
        else:
            timestamps = data.index
        
        timestamps = timestamps.sort_values()
        
        # 计算时间间隔
        time_diffs = timestamps.diff().dropna()
        
        # 期望的时间间隔（基于数据频率）
        expected_interval = time_diffs.median()
        
        # 检测超过期望间隔2倍的间隙
        gaps = time_diffs > (expected_interval * 2)
        return gaps.sum()
    
    def _check_ohlc_consistency(self, data: pd.DataFrame) -> float:
        """检查OHLC一致性"""
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return 1.0
        
        ohlc_data = data[required_cols].dropna()
        if len(ohlc_data) == 0:
            return 1.0
        
        # 检查high >= max(open, close) and low <= min(open, close)
        valid_high = (ohlc_data['high'] >= ohlc_data[['open', 'close']].max(axis=1))
        valid_low = (ohlc_data['low'] <= ohlc_data[['open', 'close']].min(axis=1))
        
        consistency = (valid_high & valid_low).mean()
        return consistency
    
    def _check_volume_consistency(self, data: pd.DataFrame) -> float:
        """检查成交量一致性"""
        if 'volume' not in data.columns:
            return 1.0
        
        volume_data = data['volume'].dropna()
        if len(volume_data) == 0:
            return 1.0
        
        # 检查成交量非负
        valid_volume = volume_data >= 0
        consistency = valid_volume.mean()
        return consistency
    
    def _calculate_data_latency(self, data: pd.DataFrame) -> float:
        """计算数据延迟"""
        if len(data) == 0:
            return 0.0
        
        # 获取最新数据的时间戳
        if 'timestamp' in data.columns:
            latest_timestamp = pd.to_datetime(data['timestamp']).max()
        elif hasattr(data.index, 'max'):
            latest_timestamp = data.index.max()
        else:
            return 0.0
        
        # 计算与当前时间的差异
        current_time = datetime.now()
        if isinstance(latest_timestamp, str):
            latest_timestamp = pd.to_datetime(latest_timestamp)
        
        latency = (current_time - latest_timestamp).total_seconds() * 1000
        return max(0, latency)  # 确保非负
    
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """计算质量分数"""
        weights = {
            'completeness_ratio': 0.3,
            'price_anomaly_rate': 0.2,
            'volume_anomaly_rate': 0.15,
            'timestamp_gap_rate': 0.15,
            'ohlc_consistency': 0.1,
            'volume_consistency': 0.1
        }
        
        score = 0.0
        
        # 完整性（正向指标）
        score += weights['completeness_ratio'] * metrics['completeness_ratio']
        
        # 异常率（负向指标，需要反转）
        score += weights['price_anomaly_rate'] * (1 - min(1, metrics['price_anomaly_rate']))
        score += weights['volume_anomaly_rate'] * (1 - min(1, metrics['volume_anomaly_rate']))
        score += weights['timestamp_gap_rate'] * (1 - min(1, metrics['timestamp_gap_rate']))
        
        # 一致性（正向指标）
        score += weights['ohlc_consistency'] * metrics['ohlc_consistency']
        score += weights['volume_consistency'] * metrics['volume_consistency']
        
        return min(1.0, max(0.0, score))  # 确保在0-1范围内
    
    def _determine_quality_status(self, quality_score: float) -> DataQualityStatus:
        """确定质量状态"""
        if quality_score >= 0.99:
            return DataQualityStatus.EXCELLENT
        elif quality_score >= 0.95:
            return DataQualityStatus.GOOD
        elif quality_score >= 0.90:
            return DataQualityStatus.WARNING
        elif quality_score >= 0.80:
            return DataQualityStatus.POOR
        else:
            return DataQualityStatus.CRITICAL
    
    def _get_system_resources(self) -> Tuple[float, float]:
        """获取系统资源使用情况"""
        try:
            import psutil
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            memory_usage_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
            return cpu_usage, memory_usage_mb
        except ImportError:
            # 如果psutil不可用，返回默认值
            return 0.0, 0.0
    
    def _get_io_metrics(self) -> Tuple[float, float]:
        """获取I/O指标"""
        try:
            import psutil
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()
            
            # 简化的I/O速率计算
            disk_io_mb_s = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024) / self.monitoring_interval
            net_io_mb_s = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024) / self.monitoring_interval
            
            return disk_io_mb_s, net_io_mb_s
        except ImportError:
            return 0.0, 0.0
    
    def _save_quality_metrics(self, metrics: DataQualityMetrics):
        """保存质量指标到数据库"""
        with sqlite3.connect(self.monitoring_db) as conn:
            conn.execute("""
                INSERT INTO data_quality (
                    timestamp, symbol, timeframe, total_records, missing_records,
                    completeness_ratio, price_anomalies, volume_anomalies, timestamp_gaps,
                    ohlc_consistency, volume_consistency, data_latency_ms, processing_time_ms,
                    quality_score, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(),
                metrics.symbol,
                metrics.timeframe,
                metrics.total_records,
                metrics.missing_records,
                metrics.completeness_ratio,
                metrics.price_anomalies,
                metrics.volume_anomalies,
                metrics.timestamp_gaps,
                metrics.ohlc_consistency,
                metrics.volume_consistency,
                metrics.data_latency_ms,
                metrics.processing_time_ms,
                metrics.quality_score,
                metrics.status
            ))
    
    def _save_performance_metrics(self, metrics: PipelinePerformanceMetrics):
        """保存性能指标到数据库"""
        with sqlite3.connect(self.monitoring_db) as conn:
            conn.execute("""
                INSERT INTO pipeline_performance (
                    timestamp, stage, throughput_rps, latency_ms, cpu_usage_percent,
                    memory_usage_mb, error_count, success_rate, disk_io_mb_per_s, network_io_mb_per_s
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(),
                metrics.stage,
                metrics.throughput_records_per_second,
                metrics.latency_ms,
                metrics.cpu_usage_percent,
                metrics.memory_usage_mb,
                metrics.error_count,
                metrics.success_rate,
                metrics.disk_io_mb_per_s,
                metrics.network_io_mb_per_s
            ))
    
    def _check_quality_alerts(self, metrics: DataQualityMetrics):
        """检查质量告警"""
        alerts = []
        
        # 检查完整性
        if metrics.completeness_ratio < self.quality_thresholds['completeness_min']:
            alerts.append({
                'type': 'DATA_COMPLETENESS',
                'severity': 'HIGH' if metrics.completeness_ratio < 0.9 else 'MEDIUM',
                'message': f"Data completeness below threshold: {metrics.completeness_ratio:.2%}",
                'details': f"Symbol: {metrics.symbol}, Timeframe: {metrics.timeframe}"
            })
        
        # 检查延迟
        if metrics.data_latency_ms > self.quality_thresholds['latency_max_ms']:
            alerts.append({
                'type': 'DATA_LATENCY',
                'severity': 'HIGH' if metrics.data_latency_ms > 5000 else 'MEDIUM',
                'message': f"Data latency too high: {metrics.data_latency_ms:.0f}ms",
                'details': f"Symbol: {metrics.symbol}, Timeframe: {metrics.timeframe}"
            })
        
        # 检查异常率
        total_anomalies = metrics.price_anomalies + metrics.volume_anomalies
        anomaly_rate = total_anomalies / metrics.total_records if metrics.total_records > 0 else 0
        if anomaly_rate > self.quality_thresholds['anomaly_threshold']:
            alerts.append({
                'type': 'DATA_ANOMALIES',
                'severity': 'MEDIUM',
                'message': f"High anomaly rate: {anomaly_rate:.2%}",
                'details': f"Symbol: {metrics.symbol}, Price anomalies: {metrics.price_anomalies}, Volume anomalies: {metrics.volume_anomalies}"
            })
        
        # 发送告警
        for alert in alerts:
            self._send_alert(alert)
    
    def _check_performance_alerts(self, metrics: PipelinePerformanceMetrics):
        """检查性能告警"""
        alerts = []
        
        # 检查吞吐量
        if metrics.throughput_records_per_second < self.performance_targets['throughput_min_rps']:
            alerts.append({
                'type': 'LOW_THROUGHPUT',
                'severity': 'MEDIUM',
                'message': f"Throughput below target: {metrics.throughput_records_per_second:.1f} rps",
                'details': f"Stage: {metrics.stage}"
            })
        
        # 检查成功率
        if metrics.success_rate < self.performance_targets['success_rate_min']:
            alerts.append({
                'type': 'LOW_SUCCESS_RATE',
                'severity': 'HIGH',
                'message': f"Success rate below target: {metrics.success_rate:.2%}",
                'details': f"Stage: {metrics.stage}, Errors: {metrics.error_count}"
            })
        
        # 检查资源使用
        if metrics.cpu_usage_percent > self.performance_targets['cpu_usage_max'] * 100:
            alerts.append({
                'type': 'HIGH_CPU_USAGE',
                'severity': 'MEDIUM',
                'message': f"CPU usage too high: {metrics.cpu_usage_percent:.1f}%",
                'details': f"Stage: {metrics.stage}"
            })
        
        # 发送告警
        for alert in alerts:
            self._send_alert(alert)
    
    def _send_alert(self, alert: Dict[str, Any]):
        """发送告警"""
        timestamp = datetime.now()
        
        # 保存到数据库
        with sqlite3.connect(self.monitoring_db) as conn:
            conn.execute("""
                INSERT INTO alerts (timestamp, alert_type, severity, message, details)
                VALUES (?, ?, ?, ?, ?)
            """, (
                timestamp.isoformat(),
                alert['type'],
                alert['severity'],
                alert['message'],
                alert['details']
            ))
        
        # 发送到配置的告警渠道
        if 'log' in self.alert_channels:
            if alert['severity'] == 'HIGH':
                logger.error(f"ALERT [{alert['type']}]: {alert['message']} - {alert['details']}")
            else:
                logger.warning(f"ALERT [{alert['type']}]: {alert['message']} - {alert['details']}")
    
    def _periodic_health_check(self):
        """定期健康检查"""
        try:
            # 检查数据库大小
            db_size_mb = self.monitoring_db.stat().st_size / (1024 * 1024)
            if db_size_mb > 1000:  # 1GB
                logger.warning(f"Monitoring database size is large: {db_size_mb:.1f}MB")
            
            # 清理旧数据
            self._cleanup_old_data()
            
        except Exception as e:
            logger.error(f"Periodic health check error: {e}")
    
    def _cleanup_old_data(self):
        """清理旧数据"""
        cutoff_date = datetime.now() - timedelta(days=30)
        
        with sqlite3.connect(self.monitoring_db) as conn:
            # 清理30天前的数据
            conn.execute("DELETE FROM data_quality WHERE timestamp < ?", (cutoff_date.isoformat(),))
            conn.execute("DELETE FROM pipeline_performance WHERE timestamp < ?", (cutoff_date.isoformat(),))
            conn.execute("DELETE FROM alerts WHERE timestamp < ? AND resolved = TRUE", (cutoff_date.isoformat(),))
            
            conn.commit()
    
    def get_quality_summary(self, hours: int = 24) -> Dict[str, Any]:
        """获取质量摘要"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.monitoring_db) as conn:
            conn.row_factory = sqlite3.Row
            
            # 获取质量统计
            quality_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_checks,
                    AVG(quality_score) as avg_quality_score,
                    MIN(quality_score) as min_quality_score,
                    MAX(quality_score) as max_quality_score,
                    SUM(CASE WHEN status = 'CRITICAL' THEN 1 ELSE 0 END) as critical_count,
                    SUM(CASE WHEN status = 'POOR' THEN 1 ELSE 0 END) as poor_count
                FROM data_quality 
                WHERE timestamp > ?
            """, (cutoff_time.isoformat(),)).fetchone()
            
            # 获取告警统计
            alert_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_alerts,
                    SUM(CASE WHEN severity = 'HIGH' THEN 1 ELSE 0 END) as high_severity_alerts
                FROM alerts 
                WHERE timestamp > ?
            """, (cutoff_time.isoformat(),)).fetchone()
            
            return {
                'period_hours': hours,
                'quality_stats': dict(quality_stats) if quality_stats else {},
                'alert_stats': dict(alert_stats) if alert_stats else {},
                'timestamp': datetime.now().isoformat()
            }
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """获取性能摘要"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.monitoring_db) as conn:
            conn.row_factory = sqlite3.Row
            
            # 按阶段获取性能统计
            performance_stats = conn.execute("""
                SELECT 
                    stage,
                    COUNT(*) as measurement_count,
                    AVG(throughput_rps) as avg_throughput,
                    AVG(latency_ms) as avg_latency,
                    AVG(success_rate) as avg_success_rate,
                    MAX(cpu_usage_percent) as max_cpu_usage,
                    MAX(memory_usage_mb) as max_memory_usage
                FROM pipeline_performance 
                WHERE timestamp > ?
                GROUP BY stage
            """, (cutoff_time.isoformat(),)).fetchall()
            
            return {
                'period_hours': hours,
                'performance_by_stage': [dict(row) for row in performance_stats],
                'timestamp': datetime.now().isoformat()
            }


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    # 配置监控器
    config = {
        'quality_thresholds': {
            'completeness_min': 0.95,
            'latency_max_ms': 1000,
            'error_rate_max': 0.05,
            'anomaly_threshold': 0.02
        },
        'performance_targets': {
            'throughput_min_rps': 100,
            'cpu_usage_max': 0.8,
            'memory_usage_max_mb': 2048,
            'success_rate_min': 0.95
        },
        'alert_channels': ['log'],
        'monitoring_interval': 30
    }
    
    # 创建监控器
    monitor = DataPipelineMonitor(config)
    
    # 生成测试数据
    dates = pd.date_range(start='2024-08-01', end='2024-08-02', freq='5min')
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(50000, 52000, len(dates)),
        'high': np.random.uniform(50500, 52500, len(dates)),
        'low': np.random.uniform(49500, 51500, len(dates)),
        'close': np.random.uniform(50000, 52000, len(dates)),
        'volume': np.random.uniform(100, 1000, len(dates))
    })
    
    # 确保OHLC一致性
    test_data['high'] = test_data[['open', 'close']].max(axis=1) + np.random.uniform(0, 100, len(dates))
    test_data['low'] = test_data[['open', 'close']].min(axis=1) - np.random.uniform(0, 100, len(dates))
    
    print("Testing data pipeline monitor...")
    
    # 测试数据质量监控
    start_time = time.time()
    quality_metrics = monitor.monitor_data_quality(test_data, "BTCUSDT", "5m")
    print(f"Quality Score: {quality_metrics.quality_score:.3f}")
    print(f"Status: {quality_metrics.status}")
    
    # 测试性能监控
    performance_metrics = monitor.monitor_pipeline_performance(
        PipelineStage.PROCESSING,
        start_time,
        len(test_data),
        error_count=0
    )
    print(f"Throughput: {performance_metrics.throughput_records_per_second:.1f} rps")
    print(f"Latency: {performance_metrics.latency_ms:.1f}ms")
    
    # 获取摘要
    quality_summary = monitor.get_quality_summary(hours=1)
    performance_summary = monitor.get_performance_summary(hours=1)
    
    print(f"Quality Summary: {quality_summary}")
    print(f"Performance Summary: {performance_summary}")
    
    print("Data pipeline monitor test completed!")