"""
Real-time Data Quality Monitor for DipMaster Trading System
实时数据质量监控和异常检测系统
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import websocket
import threading
import queue
import time
from collections import deque, defaultdict
import statistics
from enum import Enum
import sqlite3
from pathlib import Path
import redis
import psutil
import warnings

warnings.filterwarnings('ignore')

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class QualityMetric(Enum):
    """质量指标类型"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    FRESHNESS = "freshness"
    CONTINUITY = "continuity"

@dataclass
class QualityAlert:
    """质量告警"""
    timestamp: datetime
    symbol: str
    metric: QualityMetric
    level: AlertLevel
    current_value: float
    threshold: float
    message: str
    metadata: Dict[str, Any] = None

@dataclass
class QualityMetrics:
    """质量指标"""
    symbol: str
    timestamp: datetime
    completeness: float
    accuracy: float
    consistency: float
    validity: float
    freshness: float
    continuity: float
    overall_score: float
    anomaly_count: int

class RealTimeQualityMonitor:
    """实时数据质量监控器"""
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # 配置参数
        self.config = config or self.get_default_config()
        
        # 质量阈值
        self.quality_thresholds = {
            QualityMetric.COMPLETENESS: 0.995,
            QualityMetric.ACCURACY: 0.999,
            QualityMetric.CONSISTENCY: 0.995,
            QualityMetric.VALIDITY: 0.999,
            QualityMetric.FRESHNESS: 300,  # 5分钟
            QualityMetric.CONTINUITY: 0.98
        }
        
        # 数据缓冲区 - 存储最近的数据用于质量分析
        self.data_buffers = defaultdict(lambda: deque(maxlen=100))  # 每个symbol最多保存100条记录
        self.quality_history = defaultdict(lambda: deque(maxlen=1000))  # 质量历史
        self.alert_history = deque(maxlen=10000)  # 告警历史
        
        # 实时统计
        self.realtime_stats = defaultdict(dict)
        
        # 异常检测器
        self.anomaly_detectors = {}
        self.initialize_anomaly_detectors()
        
        # 告警处理器
        self.alert_handlers = []
        self.setup_alert_handlers()
        
        # 数据库连接
        self.init_quality_database()
        
        # Redis连接 (可选)
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=1, decode_responses=False)
            self.redis_client.ping()
            self.redis_enabled = True
        except:
            self.logger.warning("Redis未连接，禁用实时缓存功能")
            self.redis_enabled = False
        
        # 监控状态
        self.monitoring_active = False
        self.monitor_thread = None
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.FileHandler('logs/quality_monitor.log'),
                logging.StreamHandler()
            ]
        )
    
    def get_default_config(self) -> Dict:
        """默认配置"""
        return {
            "monitoring_interval": 10,  # 监控间隔(秒)
            "buffer_size": 100,
            "alert_cooldown": 300,  # 告警冷却时间(秒)
            "quality_check_window": 50,  # 质量检查窗口大小
            "enable_anomaly_detection": True,
            "enable_cross_validation": True,
            "enable_predictive_alerts": True
        }
    
    def init_quality_database(self):
        """初始化质量数据库"""
        db_path = Path("data/quality_monitor.db")
        db_path.parent.mkdir(exist_ok=True, parents=True)
        
        self.quality_db = sqlite3.connect(str(db_path), check_same_thread=False)
        cursor = self.quality_db.cursor()
        
        # 质量指标表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp TEXT,
                completeness REAL,
                accuracy REAL,
                consistency REAL,
                validity REAL,
                freshness REAL,
                continuity REAL,
                overall_score REAL,
                anomaly_count INTEGER
            )
        """)
        
        # 告警记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                metric TEXT,
                level TEXT,
                current_value REAL,
                threshold REAL,
                message TEXT,
                metadata TEXT
            )
        """)
        
        self.quality_db.commit()
    
    def initialize_anomaly_detectors(self):
        """初始化异常检测器"""
        self.anomaly_detectors = {
            'price_spike': PriceSpikeDetector(),
            'volume_anomaly': VolumeAnomalyDetector(),
            'gap_detector': DataGapDetector(),
            'pattern_anomaly': PatternAnomalyDetector()
        }
    
    def setup_alert_handlers(self):
        """设置告警处理器"""
        self.alert_handlers = [
            DatabaseAlertHandler(self.quality_db),
            LogAlertHandler(self.logger),
            # EmailAlertHandler(),  # 可选
            # SlackAlertHandler(),  # 可选
        ]
        
        if self.redis_enabled:
            self.alert_handlers.append(RedisAlertHandler(self.redis_client))
    
    def ingest_market_data(self, symbol: str, data: Dict):
        """接收市场数据"""
        try:
            # 转换数据格式
            if isinstance(data, dict):
                # 假设数据格式: {'timestamp': ..., 'open': ..., 'high': ..., 'low': ..., 'close': ..., 'volume': ...}
                row = {
                    'timestamp': pd.to_datetime(data.get('timestamp', datetime.now())),
                    'open': float(data.get('open', 0)),
                    'high': float(data.get('high', 0)),
                    'low': float(data.get('low', 0)),
                    'close': float(data.get('close', 0)),
                    'volume': float(data.get('volume', 0))
                }
            else:
                return
            
            # 添加到缓冲区
            self.data_buffers[symbol].append(row)
            
            # 更新实时统计
            self.update_realtime_stats(symbol, row)
            
            # 执行质量检查
            if len(self.data_buffers[symbol]) >= self.config["quality_check_window"]:
                asyncio.create_task(self.check_data_quality(symbol))
            
        except Exception as e:
            self.logger.error(f"接收 {symbol} 数据失败: {e}")
    
    def update_realtime_stats(self, symbol: str, row: Dict):
        """更新实时统计"""
        if symbol not in self.realtime_stats:
            self.realtime_stats[symbol] = {
                'last_update': None,
                'price_changes': deque(maxlen=100),
                'volume_changes': deque(maxlen=100),
                'data_points_count': 0,
                'average_interval': 0
            }
        
        stats = self.realtime_stats[symbol]
        current_time = row['timestamp']
        
        # 更新时间间隔统计
        if stats['last_update']:
            interval = (current_time - stats['last_update']).total_seconds()
            if stats['average_interval'] == 0:
                stats['average_interval'] = interval
            else:
                stats['average_interval'] = (stats['average_interval'] * 0.9 + interval * 0.1)
        
        stats['last_update'] = current_time
        stats['data_points_count'] += 1
        
        # 价格变化统计
        if len(self.data_buffers[symbol]) > 1:
            prev_close = list(self.data_buffers[symbol])[-2]['close']
            price_change = (row['close'] - prev_close) / prev_close
            stats['price_changes'].append(price_change)
            
            prev_volume = list(self.data_buffers[symbol])[-2]['volume']
            if prev_volume > 0:
                volume_change = (row['volume'] - prev_volume) / prev_volume
                stats['volume_changes'].append(volume_change)
    
    async def check_data_quality(self, symbol: str):
        """检查数据质量"""
        try:
            buffer = list(self.data_buffers[symbol])
            if len(buffer) < 10:  # 需要足够的数据点
                return
            
            # 转换为DataFrame
            df = pd.DataFrame(buffer)
            df.set_index('timestamp', inplace=True)
            
            # 计算质量指标
            metrics = self.calculate_quality_metrics(symbol, df)
            
            # 保存质量指标
            self.quality_history[symbol].append(metrics)
            self.save_quality_metrics(metrics)
            
            # 检查是否需要告警
            alerts = self.check_quality_thresholds(metrics)
            for alert in alerts:
                await self.handle_alert(alert)
            
            # 异常检测
            if self.config["enable_anomaly_detection"]:
                anomaly_alerts = await self.detect_anomalies(symbol, df)
                for alert in anomaly_alerts:
                    await self.handle_alert(alert)
            
        except Exception as e:
            self.logger.error(f"检查 {symbol} 数据质量失败: {e}")
    
    def calculate_quality_metrics(self, symbol: str, df: pd.DataFrame) -> QualityMetrics:
        """计算质量指标"""
        now = datetime.now()
        
        # 1. 完整性 - 检查缺失值
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
        
        # 2. 一致性 - 检查OHLC关系
        consistency_violations = 0
        if len(df) > 0:
            high_violations = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
            low_violations = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
            consistency_violations = high_violations + low_violations
        
        consistency = max(0, 1 - (consistency_violations / len(df))) if len(df) > 0 else 1
        
        # 3. 有效性 - 检查数值范围
        invalid_prices = (df[['open', 'high', 'low', 'close']] <= 0).sum().sum()
        invalid_volumes = (df['volume'] < 0).sum()
        total_values = len(df) * 5
        validity = max(0, 1 - ((invalid_prices + invalid_volumes) / total_values)) if total_values > 0 else 1
        
        # 4. 精度 - 检查异常波动
        if len(df) > 1:
            price_changes = df['close'].pct_change().abs()
            extreme_changes = (price_changes > 0.5).sum()  # 超过50%变化
            accuracy = max(0, 1 - (extreme_changes / len(df)))
        else:
            accuracy = 1.0
        
        # 5. 新鲜度 - 检查数据时效
        if len(df) > 0:
            latest_time = df.index.max()
            freshness_seconds = (now - latest_time).total_seconds()
            freshness = max(0, 1 - (freshness_seconds / 3600))  # 1小时为基准
        else:
            freshness = 0.0
        
        # 6. 连续性 - 检查时间序列连续性
        if len(df) > 1:
            time_diffs = df.index.to_series().diff().dt.total_seconds()
            expected_interval = time_diffs.median()
            large_gaps = (time_diffs > expected_interval * 3).sum()
            continuity = max(0, 1 - (large_gaps / len(df)))
        else:
            continuity = 1.0
        
        # 异常计数
        anomaly_count = self.count_anomalies(df)
        
        # 总体评分
        overall_score = np.mean([completeness, accuracy, consistency, validity, freshness, continuity])
        
        return QualityMetrics(
            symbol=symbol,
            timestamp=now,
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            validity=validity,
            freshness=freshness,
            continuity=continuity,
            overall_score=overall_score,
            anomaly_count=anomaly_count
        )
    
    def count_anomalies(self, df: pd.DataFrame) -> int:
        """计算异常数量"""
        anomaly_count = 0
        
        if len(df) > 1:
            # 价格异常
            price_changes = df['close'].pct_change().abs()
            anomaly_count += (price_changes > 0.2).sum()
            
            # 成交量异常
            volume_changes = df['volume'].pct_change().abs()
            anomaly_count += (volume_changes > 5.0).sum()
            
            # 零成交量
            anomaly_count += (df['volume'] == 0).sum()
        
        return anomaly_count
    
    def check_quality_thresholds(self, metrics: QualityMetrics) -> List[QualityAlert]:
        """检查质量阈值"""
        alerts = []
        
        quality_checks = [
            (QualityMetric.COMPLETENESS, metrics.completeness, self.quality_thresholds[QualityMetric.COMPLETENESS]),
            (QualityMetric.ACCURACY, metrics.accuracy, self.quality_thresholds[QualityMetric.ACCURACY]),
            (QualityMetric.CONSISTENCY, metrics.consistency, self.quality_thresholds[QualityMetric.CONSISTENCY]),
            (QualityMetric.VALIDITY, metrics.validity, self.quality_thresholds[QualityMetric.VALIDITY]),
            (QualityMetric.CONTINUITY, metrics.continuity, self.quality_thresholds[QualityMetric.CONTINUITY])
        ]
        
        for metric_type, current_value, threshold in quality_checks:
            if current_value < threshold:
                level = self.determine_alert_level(current_value, threshold)
                
                alert = QualityAlert(
                    timestamp=metrics.timestamp,
                    symbol=metrics.symbol,
                    metric=metric_type,
                    level=level,
                    current_value=current_value,
                    threshold=threshold,
                    message=f"{metric_type.value} 低于阈值: {current_value:.3f} < {threshold:.3f}",
                    metadata={'overall_score': metrics.overall_score}
                )
                alerts.append(alert)
        
        # 新鲜度检查（使用不同逻辑）
        freshness_threshold = self.quality_thresholds[QualityMetric.FRESHNESS]
        if metrics.freshness < 0.5:  # 新鲜度低于50%
            level = AlertLevel.CRITICAL if metrics.freshness < 0.2 else AlertLevel.WARNING
            
            alert = QualityAlert(
                timestamp=metrics.timestamp,
                symbol=metrics.symbol,
                metric=QualityMetric.FRESHNESS,
                level=level,
                current_value=metrics.freshness,
                threshold=freshness_threshold,
                message=f"数据新鲜度过低: {metrics.freshness:.3f}",
                metadata={'last_update_minutes_ago': (1 - metrics.freshness) * 60}
            )
            alerts.append(alert)
        
        return alerts
    
    def determine_alert_level(self, current_value: float, threshold: float) -> AlertLevel:
        """确定告警级别"""
        ratio = current_value / threshold
        
        if ratio < 0.8:
            return AlertLevel.CRITICAL
        elif ratio < 0.9:
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO
    
    async def detect_anomalies(self, symbol: str, df: pd.DataFrame) -> List[QualityAlert]:
        """异常检测"""
        alerts = []
        
        for detector_name, detector in self.anomaly_detectors.items():
            try:
                anomalies = detector.detect(symbol, df)
                for anomaly in anomalies:
                    alert = QualityAlert(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        metric=QualityMetric.ACCURACY,  # 异常归类为精度问题
                        level=AlertLevel.WARNING,
                        current_value=0.0,
                        threshold=1.0,
                        message=f"检测到 {detector_name} 异常: {anomaly['description']}",
                        metadata=anomaly
                    )
                    alerts.append(alert)
            except Exception as e:
                self.logger.error(f"{detector_name} 异常检测失败: {e}")
        
        return alerts
    
    async def handle_alert(self, alert: QualityAlert):
        """处理告警"""
        try:
            # 添加到历史记录
            self.alert_history.append(alert)
            
            # 调用所有告警处理器
            for handler in self.alert_handlers:
                try:
                    await handler.handle(alert)
                except Exception as e:
                    self.logger.error(f"告警处理器失败: {e}")
            
            # 记录日志
            self.logger.warning(f"质量告警 [{alert.level.value.upper()}] {alert.symbol}: {alert.message}")
            
        except Exception as e:
            self.logger.error(f"处理告警失败: {e}")
    
    def save_quality_metrics(self, metrics: QualityMetrics):
        """保存质量指标到数据库"""
        try:
            cursor = self.quality_db.cursor()
            cursor.execute("""
                INSERT INTO quality_metrics 
                (symbol, timestamp, completeness, accuracy, consistency, validity, 
                 freshness, continuity, overall_score, anomaly_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.symbol,
                metrics.timestamp.isoformat(),
                metrics.completeness,
                metrics.accuracy,
                metrics.consistency,
                metrics.validity,
                metrics.freshness,
                metrics.continuity,
                metrics.overall_score,
                metrics.anomaly_count
            ))
            self.quality_db.commit()
            
        except Exception as e:
            self.logger.error(f"保存质量指标失败: {e}")
    
    def get_quality_report(self, symbol: str = None, hours: int = 24) -> Dict:
        """获取质量报告"""
        try:
            cursor = self.quality_db.cursor()
            
            # 查询条件
            since = datetime.now() - timedelta(hours=hours)
            
            if symbol:
                cursor.execute("""
                    SELECT * FROM quality_metrics 
                    WHERE symbol = ? AND timestamp > ?
                    ORDER BY timestamp DESC
                """, (symbol, since.isoformat()))
            else:
                cursor.execute("""
                    SELECT * FROM quality_metrics 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                """, (since.isoformat(),))
            
            rows = cursor.fetchall()
            
            if not rows:
                return {"message": "暂无质量数据"}
            
            # 分析质量趋势
            df = pd.DataFrame(rows, columns=[
                'id', 'symbol', 'timestamp', 'completeness', 'accuracy', 
                'consistency', 'validity', 'freshness', 'continuity', 
                'overall_score', 'anomaly_count'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            report = {
                'summary': {
                    'total_records': len(df),
                    'symbols_count': df['symbol'].nunique(),
                    'time_range': {
                        'start': df['timestamp'].min().isoformat(),
                        'end': df['timestamp'].max().isoformat()
                    },
                    'average_quality': {
                        'overall_score': df['overall_score'].mean(),
                        'completeness': df['completeness'].mean(),
                        'accuracy': df['accuracy'].mean(),
                        'consistency': df['consistency'].mean(),
                        'validity': df['validity'].mean(),
                        'freshness': df['freshness'].mean(),
                        'continuity': df['continuity'].mean()
                    },
                    'total_anomalies': df['anomaly_count'].sum()
                },
                'by_symbol': {}
            }
            
            # 按币种统计
            for sym in df['symbol'].unique():
                symbol_data = df[df['symbol'] == sym]
                report['by_symbol'][sym] = {
                    'records_count': len(symbol_data),
                    'latest_score': symbol_data.iloc[0]['overall_score'],
                    'average_score': symbol_data['overall_score'].mean(),
                    'quality_trend': 'improving' if symbol_data['overall_score'].diff().mean() > 0 else 'declining',
                    'total_anomalies': symbol_data['anomaly_count'].sum()
                }
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成质量报告失败: {e}")
            return {"error": str(e)}
    
    def start_monitoring(self):
        """启动监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("实时质量监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("实时质量监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 系统健康检查
                self._check_system_health()
                
                # 清理过期数据
                self._cleanup_old_data()
                
                # 等待下一次检查
                time.sleep(self.config["monitoring_interval"])
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
    
    def _check_system_health(self):
        """检查系统健康状态"""
        try:
            # 检查内存使用
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                self.logger.warning(f"系统内存使用过高: {memory_percent}%")
            
            # 检查缓冲区大小
            total_buffer_size = sum(len(buffer) for buffer in self.data_buffers.values())
            if total_buffer_size > 10000:
                self.logger.warning(f"数据缓冲区过大: {total_buffer_size} 条记录")
            
            # 检查Redis连接
            if self.redis_enabled:
                try:
                    self.redis_client.ping()
                except:
                    self.logger.warning("Redis连接丢失")
                    self.redis_enabled = False
            
        except Exception as e:
            self.logger.error(f"系统健康检查失败: {e}")
    
    def _cleanup_old_data(self):
        """清理过期数据"""
        try:
            # 清理数据库中的老数据（保留7天）
            cutoff_time = datetime.now() - timedelta(days=7)
            cursor = self.quality_db.cursor()
            
            cursor.execute("DELETE FROM quality_metrics WHERE timestamp < ?", (cutoff_time.isoformat(),))
            cursor.execute("DELETE FROM quality_alerts WHERE timestamp < ?", (cutoff_time.isoformat(),))
            
            self.quality_db.commit()
            
        except Exception as e:
            self.logger.error(f"清理过期数据失败: {e}")

# 异常检测器基类和实现
class BaseAnomalyDetector:
    """异常检测器基类"""
    
    def detect(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """检测异常，返回异常列表"""
        raise NotImplementedError

class PriceSpikeDetector(BaseAnomalyDetector):
    """价格异常检测器"""
    
    def detect(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        anomalies = []
        
        if len(df) < 20:
            return anomalies
        
        # 计算价格变化
        price_changes = df['close'].pct_change().abs()
        threshold = price_changes.quantile(0.99)
        
        # 找出异常点
        spikes = df[price_changes > max(threshold, 0.1)]  # 至少10%变化
        
        for idx, row in spikes.iterrows():
            anomalies.append({
                'type': 'price_spike',
                'timestamp': idx,
                'description': f"价格异常波动 {price_changes.loc[idx]:.2%}",
                'severity': 'high' if price_changes.loc[idx] > 0.2 else 'medium',
                'value': row['close'],
                'change_percent': price_changes.loc[idx]
            })
        
        return anomalies

class VolumeAnomalyDetector(BaseAnomalyDetector):
    """成交量异常检测器"""
    
    def detect(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        anomalies = []
        
        if len(df) < 20:
            return anomalies
        
        # 成交量异常
        volume_ma = df['volume'].rolling(20).mean()
        volume_std = df['volume'].rolling(20).std()
        
        # Z-score方法检测异常
        z_scores = (df['volume'] - volume_ma) / volume_std
        
        volume_anomalies = df[z_scores.abs() > 3]  # 3倍标准差
        
        for idx, row in volume_anomalies.iterrows():
            anomalies.append({
                'type': 'volume_anomaly',
                'timestamp': idx,
                'description': f"成交量异常: {row['volume']:.0f} (Z-score: {z_scores.loc[idx]:.2f})",
                'severity': 'high' if abs(z_scores.loc[idx]) > 5 else 'medium',
                'volume': row['volume'],
                'z_score': z_scores.loc[idx]
            })
        
        return anomalies

class DataGapDetector(BaseAnomalyDetector):
    """数据缺口检测器"""
    
    def detect(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        anomalies = []
        
        if len(df) < 5:
            return anomalies
        
        # 时间间隔分析
        time_diffs = df.index.to_series().diff()
        expected_interval = time_diffs.median()
        
        # 检测大缺口
        large_gaps = time_diffs[time_diffs > expected_interval * 3]
        
        for idx, gap in large_gaps.items():
            anomalies.append({
                'type': 'data_gap',
                'timestamp': idx,
                'description': f"数据缺口: {gap.total_seconds()/60:.1f} 分钟",
                'severity': 'high' if gap > expected_interval * 10 else 'medium',
                'gap_minutes': gap.total_seconds() / 60,
                'expected_interval_minutes': expected_interval.total_seconds() / 60
            })
        
        return anomalies

class PatternAnomalyDetector(BaseAnomalyDetector):
    """模式异常检测器"""
    
    def detect(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        anomalies = []
        
        if len(df) < 50:
            return anomalies
        
        # 检测连续相同价格（可能的数据冻结）
        close_diff = df['close'].diff()
        consecutive_same = (close_diff == 0).rolling(5).sum()
        
        frozen_data = df[consecutive_same >= 5]
        
        for idx, row in frozen_data.iterrows():
            anomalies.append({
                'type': 'frozen_price',
                'timestamp': idx,
                'description': f"价格冻结: 连续相同价格 {row['close']}",
                'severity': 'medium',
                'price': row['close']
            })
        
        return anomalies

# 告警处理器
class BaseAlertHandler:
    """告警处理器基类"""
    
    async def handle(self, alert: QualityAlert):
        """处理告警"""
        raise NotImplementedError

class DatabaseAlertHandler(BaseAlertHandler):
    """数据库告警处理器"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    async def handle(self, alert: QualityAlert):
        """保存告警到数据库"""
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO quality_alerts 
            (timestamp, symbol, metric, level, current_value, threshold, message, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.timestamp.isoformat(),
            alert.symbol,
            alert.metric.value,
            alert.level.value,
            alert.current_value,
            alert.threshold,
            alert.message,
            json.dumps(alert.metadata or {})
        ))
        self.db.commit()

class LogAlertHandler(BaseAlertHandler):
    """日志告警处理器"""
    
    def __init__(self, logger):
        self.logger = logger
    
    async def handle(self, alert: QualityAlert):
        """记录告警到日志"""
        log_level = {
            AlertLevel.INFO: self.logger.info,
            AlertLevel.WARNING: self.logger.warning,
            AlertLevel.CRITICAL: self.logger.error,
            AlertLevel.EMERGENCY: self.logger.critical
        }
        
        log_func = log_level.get(alert.level, self.logger.info)
        log_func(f"[{alert.symbol}] {alert.message}")

class RedisAlertHandler(BaseAlertHandler):
    """Redis告警处理器"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def handle(self, alert: QualityAlert):
        """发布告警到Redis"""
        alert_data = {
            'timestamp': alert.timestamp.isoformat(),
            'symbol': alert.symbol,
            'metric': alert.metric.value,
            'level': alert.level.value,
            'message': alert.message,
            'current_value': alert.current_value,
            'threshold': alert.threshold
        }
        
        # 发布到Redis频道
        self.redis.publish('quality_alerts', json.dumps(alert_data))
        
        # 同时保存到Redis列表（最近1000条）
        self.redis.lpush('recent_alerts', json.dumps(alert_data))
        self.redis.ltrim('recent_alerts', 0, 999)

# 使用示例
async def main():
    """质量监控系统演示"""
    monitor = RealTimeQualityMonitor()
    
    # 启动监控
    monitor.start_monitoring()
    
    # 模拟数据接收
    import random
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    for i in range(100):
        for symbol in symbols:
            # 模拟市场数据
            base_price = 50000 if symbol == 'BTCUSDT' else 3000 if symbol == 'ETHUSDT' else 100
            
            data = {
                'timestamp': datetime.now(),
                'open': base_price * (1 + random.uniform(-0.01, 0.01)),
                'high': base_price * (1 + random.uniform(0, 0.02)),
                'low': base_price * (1 + random.uniform(-0.02, 0)),
                'close': base_price * (1 + random.uniform(-0.01, 0.01)),
                'volume': random.uniform(1000, 10000)
            }
            
            # 偶尔添加异常数据
            if random.random() < 0.1:
                data['close'] *= random.choice([0.5, 1.5])  # 价格异常
            
            monitor.ingest_market_data(symbol, data)
        
        await asyncio.sleep(1)
    
    # 生成质量报告
    report = monitor.get_quality_report()
    print("\n=== 质量报告 ===")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    
    # 停止监控
    monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())