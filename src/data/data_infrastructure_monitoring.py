"""
Data Infrastructure Monitoring System
DipMaster Trading System - 数据基础设施监控系统

Features:
- 实时数据质量监控
- 数据completeness跟踪
- Gap检测和告警
- 性能指标收集
- 自动化报告生成
- 异常检测和告警
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sqlite3
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

warnings.filterwarnings('ignore')

@dataclass
class QualityAlert:
    """质量告警"""
    symbol: str
    timeframe: str
    alert_type: str  # 'gap', 'quality_degradation', 'missing_data', 'anomaly'
    severity: str    # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class PerformanceMetric:
    """性能指标"""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    symbol: Optional[str] = None
    timeframe: Optional[str] = None

class DataInfrastructureMonitor:
    """数据基础设施监控器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # 监控配置
        self.base_path = Path(self.config.get('base_path', 'data/enhanced_market_data'))
        self.monitoring_interval = self.config.get('monitoring_interval_seconds', 300)  # 5分钟
        
        # 质量阈值
        self.quality_thresholds = {
            'completeness_warning': 0.99,
            'completeness_critical': 0.95,
            'gap_warning_minutes': 30,
            'gap_critical_minutes': 120,
            'quality_degradation_threshold': 0.05
        }
        
        # 监控状态
        self.alerts = deque(maxlen=1000)  # 最近1000个告警
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.quality_history = defaultdict(lambda: deque(maxlen=100))
        
        # 数据库连接
        self.init_monitoring_database()
        
        # 告警配置
        self.email_config = self.config.get('email', {})
        self.webhook_config = self.config.get('webhook', {})
        
        # 监控线程控制
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # TOP30符号
        self.symbols_to_monitor = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "TRXUSDT", "LINKUSDT",
            "LTCUSDT", "DOTUSDT", "MATICUSDT", "UNIUSDT", "ICPUSDT",
            "NEARUSDT", "XLMUSDT", "ATOMUSDT", "VETUSDT", "FILUSDT",
            "APTUSDT", "ARBUSDT", "OPUSDT", "GRTUSDT", "MKRUSDT",
            "AAVEUSDT", "COMPUSDT", "ALGOUSDT", "TONUSDT", "INJUSDT"
        ]
        
        self.timeframes_to_monitor = ['1m', '5m', '15m', '1h', '4h', '1d']
    
    def init_monitoring_database(self):
        """初始化监控数据库"""
        db_path = Path("data/monitoring.db")
        db_path.parent.mkdir(exist_ok=True)
        
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                completeness REAL,
                consistency REAL,
                accuracy REAL,
                overall_score REAL,
                record_count INTEGER,
                file_size_mb REAL
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_time DATETIME
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                symbol TEXT,
                timeframe TEXT
            )
        """)
        
        self.conn.commit()
    
    def start_monitoring(self):
        """启动监控"""
        if self.monitoring_active:
            self.logger.warning("监控已经启动")
            return
        
        self.logger.info("启动数据基础设施监控...")
        self.monitoring_active = True
        
        # 启动监控线程
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.logger.info("停止数据基础设施监控...")
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
    
    def _monitoring_loop(self):
        """监控主循环"""
        while self.monitoring_active:
            try:
                # 执行监控检查
                self._perform_monitoring_cycle()
                
                # 等待下一个循环
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                time.sleep(60)  # 错误时等待1分钟
    
    def _perform_monitoring_cycle(self):
        """执行一次监控循环"""
        cycle_start = time.time()
        
        # 1. 数据质量检查
        quality_issues = self._check_data_quality()
        
        # 2. Gap检测
        gap_issues = self._detect_data_gaps()
        
        # 3. 文件完整性检查
        file_issues = self._check_file_integrity()
        
        # 4. 性能指标收集
        self._collect_performance_metrics()
        
        # 5. 处理告警
        all_issues = quality_issues + gap_issues + file_issues
        for issue in all_issues:
            self._process_alert(issue)
        
        # 6. 更新监控指标
        cycle_time = time.time() - cycle_start
        self._record_metric("monitoring_cycle_time", cycle_time, "seconds")
        
        self.logger.debug(f"监控循环完成: {cycle_time:.2f}秒, 发现 {len(all_issues)} 个问题")
    
    def _check_data_quality(self) -> List[QualityAlert]:
        """检查数据质量"""
        alerts = []
        
        for symbol in self.symbols_to_monitor:
            for timeframe in self.timeframes_to_monitor:
                try:
                    quality_metrics = self._assess_file_quality(symbol, timeframe)
                    
                    if quality_metrics:
                        # 检查质量阈值
                        if quality_metrics['overall_score'] < self.quality_thresholds['completeness_critical']:
                            alerts.append(QualityAlert(
                                symbol=symbol,
                                timeframe=timeframe,
                                alert_type='quality_degradation',
                                severity='critical',
                                message=f"数据质量严重下降: {quality_metrics['overall_score']:.3f}",
                                timestamp=datetime.now(timezone.utc)
                            ))
                        elif quality_metrics['overall_score'] < self.quality_thresholds['completeness_warning']:
                            alerts.append(QualityAlert(
                                symbol=symbol,
                                timeframe=timeframe,
                                alert_type='quality_degradation',
                                severity='warning',
                                message=f"数据质量轻微下降: {quality_metrics['overall_score']:.3f}",
                                timestamp=datetime.now(timezone.utc)
                            ))
                        
                        # 存储质量指标到数据库
                        self._save_quality_metrics(symbol, timeframe, quality_metrics)
                
                except Exception as e:
                    self.logger.error(f"质量检查失败 {symbol} {timeframe}: {e}")
        
        return alerts
    
    def _assess_file_quality(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """评估文件质量"""
        file_path = self.base_path / f"{symbol}_{timeframe}_2years.parquet"
        
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_parquet(file_path)
            if df.empty:
                return None
            
            # 确保时间索引
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df.set_index('timestamp', inplace=True)
            
            # 计算质量指标
            metrics = {
                'completeness': self._calculate_completeness(df, timeframe),
                'consistency': self._calculate_consistency(df),
                'accuracy': self._calculate_accuracy(df),
                'record_count': len(df),
                'file_size_mb': file_path.stat().st_size / 1024 / 1024,
                'date_range_start': df.index.min().isoformat() if not df.empty else None,
                'date_range_end': df.index.max().isoformat() if not df.empty else None
            }
            
            metrics['overall_score'] = np.mean([
                metrics['completeness'],
                metrics['consistency'], 
                metrics['accuracy']
            ])
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"质量评估失败 {symbol} {timeframe}: {e}")
            return None
    
    def _calculate_completeness(self, df: pd.DataFrame, timeframe: str) -> float:
        """计算完整性"""
        if df.empty:
            return 0.0
        
        # 缺失值检查
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        
        # 时间序列连续性检查
        expected_interval = self._get_expected_interval(timeframe)
        time_diffs = df.index.to_series().diff()[1:]  # 跳过第一个NaT
        
        if len(time_diffs) > 0:
            # 计算gap比例
            gaps = time_diffs > expected_interval * 2
            gap_ratio = gaps.sum() / len(time_diffs)
        else:
            gap_ratio = 0
        
        return max(0, 1 - missing_ratio - gap_ratio * 0.1)
    
    def _calculate_consistency(self, df: pd.DataFrame) -> float:
        """计算一致性"""
        if df.empty:
            return 1.0
        
        violations = 0
        total_checks = len(df)
        
        if total_checks > 0:
            # OHLC关系检查
            violations += (df['high'] < df[['open', 'close']].max(axis=1)).sum()
            violations += (df['low'] > df[['open', 'close']].min(axis=1)).sum()
            violations += (df['high'] < df['low']).sum()
            
            # 价格正值检查
            violations += (df[['open', 'high', 'low', 'close']] <= 0).sum().sum()
            violations += (df['volume'] < 0).sum()
        
        return max(0, 1 - violations / (total_checks * 6))
    
    def _calculate_accuracy(self, df: pd.DataFrame) -> float:
        """计算准确性"""
        if df.empty or len(df) < 10:
            return 1.0
        
        # 异常价格变动检测
        returns = df['close'].pct_change().abs()
        extreme_moves = (returns > 0.5).sum()  # 超过50%变动
        
        # 价格跳跃检测
        price_gaps = (df['open'] / df['close'].shift(1) - 1).abs()
        large_gaps = (price_gaps > 0.1).sum()  # 超过10%缺口
        
        anomaly_ratio = (extreme_moves + large_gaps) / len(df)
        return max(0, 1 - anomaly_ratio * 2)  # 异常对准确性影响更大
    
    def _get_expected_interval(self, timeframe: str) -> pd.Timedelta:
        """获取预期时间间隔"""
        intervals = {
            '1m': pd.Timedelta(minutes=1),
            '5m': pd.Timedelta(minutes=5),
            '15m': pd.Timedelta(minutes=15),
            '1h': pd.Timedelta(hours=1),
            '4h': pd.Timedelta(hours=4),
            '1d': pd.Timedelta(days=1)
        }
        return intervals.get(timeframe, pd.Timedelta(minutes=5))
    
    def _detect_data_gaps(self) -> List[QualityAlert]:
        """检测数据缺口"""
        alerts = []
        
        for symbol in self.symbols_to_monitor:
            for timeframe in self.timeframes_to_monitor:
                try:
                    gaps = self._find_gaps_in_file(symbol, timeframe)
                    
                    for gap_start, gap_end, gap_duration in gaps:
                        gap_minutes = gap_duration.total_seconds() / 60
                        
                        if gap_minutes > self.quality_thresholds['gap_critical_minutes']:
                            severity = 'critical'
                        elif gap_minutes > self.quality_thresholds['gap_warning_minutes']:
                            severity = 'warning'
                        else:
                            continue  # 忽略小gaps
                        
                        alerts.append(QualityAlert(
                            symbol=symbol,
                            timeframe=timeframe,
                            alert_type='gap',
                            severity=severity,
                            message=f"数据缺口: {gap_start} 到 {gap_end} ({gap_minutes:.0f}分钟)",
                            timestamp=datetime.now(timezone.utc)
                        ))
                
                except Exception as e:
                    self.logger.error(f"Gap检测失败 {symbol} {timeframe}: {e}")
        
        return alerts
    
    def _find_gaps_in_file(self, symbol: str, timeframe: str) -> List[Tuple[datetime, datetime, timedelta]]:
        """在文件中查找gaps"""
        file_path = self.base_path / f"{symbol}_{timeframe}_2years.parquet"
        
        if not file_path.exists():
            return []
        
        try:
            df = pd.read_parquet(file_path)
            if df.empty:
                return []
            
            # 确保时间索引
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df.set_index('timestamp', inplace=True)
            
            expected_interval = self._get_expected_interval(timeframe)
            time_diffs = df.index.to_series().diff()[1:]
            
            # 查找大gaps
            gaps = []
            for i, diff in enumerate(time_diffs):
                if pd.notna(diff) and diff > expected_interval * 2:
                    gap_start = df.index[i]
                    gap_end = df.index[i + 1]
                    gaps.append((gap_start, gap_end, diff))
            
            return gaps
            
        except Exception as e:
            self.logger.error(f"文件gap检测失败 {file_path}: {e}")
            return []
    
    def _check_file_integrity(self) -> List[QualityAlert]:
        """检查文件完整性"""
        alerts = []
        
        for symbol in self.symbols_to_monitor:
            for timeframe in self.timeframes_to_monitor:
                file_path = self.base_path / f"{symbol}_{timeframe}_2years.parquet"
                
                # 检查文件是否存在
                if not file_path.exists():
                    alerts.append(QualityAlert(
                        symbol=symbol,
                        timeframe=timeframe,
                        alert_type='missing_data',
                        severity='critical',
                        message=f"数据文件不存在: {file_path.name}",
                        timestamp=datetime.now(timezone.utc)
                    ))
                    continue
                
                # 检查文件大小
                file_size = file_path.stat().st_size
                if file_size < 1024:  # 小于1KB
                    alerts.append(QualityAlert(
                        symbol=symbol,
                        timeframe=timeframe,
                        alert_type='file_corruption',
                        severity='high',
                        message=f"文件大小异常: {file_size} bytes",
                        timestamp=datetime.now(timezone.utc)
                    ))
                
                # 检查文件修改时间
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                if datetime.now(timezone.utc) - mtime > timedelta(hours=24):
                    alerts.append(QualityAlert(
                        symbol=symbol,
                        timeframe=timeframe,
                        alert_type='stale_data',
                        severity='medium',
                        message=f"数据文件过期: 最后修改 {mtime.strftime('%Y-%m-%d %H:%M:%S')}",
                        timestamp=datetime.now(timezone.utc)
                    ))
        
        return alerts
    
    def _collect_performance_metrics(self):
        """收集性能指标"""
        # 总体统计
        total_files = len(list(self.base_path.glob("*.parquet")))
        total_size = sum(f.stat().st_size for f in self.base_path.glob("*.parquet")) / 1024 / 1024 / 1024  # GB
        
        self._record_metric("total_data_files", total_files, "count")
        self._record_metric("total_data_size_gb", total_size, "GB")
        
        # 按时间框架统计
        for timeframe in self.timeframes_to_monitor:
            tf_files = len(list(self.base_path.glob(f"*_{timeframe}_*.parquet")))
            self._record_metric("data_files_by_timeframe", tf_files, "count", timeframe=timeframe)
        
        # 按币种统计
        for symbol in self.symbols_to_monitor:
            symbol_files = len(list(self.base_path.glob(f"{symbol}_*.parquet")))
            self._record_metric("data_files_by_symbol", symbol_files, "count", symbol=symbol)
    
    def _record_metric(self, metric_name: str, value: float, unit: str, 
                      symbol: str = None, timeframe: str = None):
        """记录指标"""
        timestamp = datetime.now(timezone.utc)
        
        # 内存存储
        key = f"{metric_name}_{symbol or ''}_{timeframe or ''}".strip('_')
        self.metrics_history[key].append({
            'timestamp': timestamp,
            'value': value,
            'unit': unit
        })
        
        # 数据库存储
        try:
            self.conn.execute("""
                INSERT INTO performance_metrics 
                (metric_name, value, unit, timestamp, symbol, timeframe)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (metric_name, value, unit, timestamp, symbol, timeframe))
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"保存性能指标失败: {e}")
    
    def _save_quality_metrics(self, symbol: str, timeframe: str, metrics: Dict[str, Any]):
        """保存质量指标"""
        try:
            self.conn.execute("""
                INSERT INTO quality_metrics 
                (symbol, timeframe, timestamp, completeness, consistency, accuracy, 
                 overall_score, record_count, file_size_mb)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, timeframe, datetime.now(timezone.utc),
                metrics['completeness'], metrics['consistency'], metrics['accuracy'],
                metrics['overall_score'], metrics['record_count'], metrics['file_size_mb']
            ))
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"保存质量指标失败: {e}")
    
    def _process_alert(self, alert: QualityAlert):
        """处理告警"""
        # 添加到告警队列
        self.alerts.append(alert)
        
        # 保存到数据库
        try:
            self.conn.execute("""
                INSERT INTO alerts 
                (symbol, timeframe, alert_type, severity, message, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (alert.symbol, alert.timeframe, alert.alert_type, 
                  alert.severity, alert.message, alert.timestamp))
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"保存告警失败: {e}")
        
        # 发送通知
        if alert.severity in ['high', 'critical']:
            self._send_alert_notification(alert)
        
        # 日志记录
        log_level = {
            'low': logging.DEBUG,
            'medium': logging.INFO,
            'warning': logging.WARNING,
            'high': logging.WARNING,
            'critical': logging.ERROR
        }.get(alert.severity, logging.INFO)
        
        self.logger.log(log_level, f"[{alert.severity.upper()}] {alert.symbol} {alert.timeframe}: {alert.message}")
    
    def _send_alert_notification(self, alert: QualityAlert):
        """发送告警通知"""
        # 邮件通知
        if self.email_config.get('enabled', False):
            self._send_email_alert(alert)
        
        # Webhook通知
        if self.webhook_config.get('enabled', False):
            self._send_webhook_alert(alert)
    
    def _send_email_alert(self, alert: QualityAlert):
        """发送邮件告警"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.email_config['smtp_user']
            msg['To'] = ', '.join(self.email_config['recipients'])
            msg['Subject'] = f"[{alert.severity.upper()}] DipMaster数据告警: {alert.symbol}"
            
            body = f"""
DipMaster Trading System 数据基础设施告警

交易对: {alert.symbol}
时间框架: {alert.timeframe}
告警类型: {alert.alert_type}
严重程度: {alert.severity}
消息: {alert.message}
时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

请及时处理此告警。
            """
            
            msg.attach(MimeText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['smtp_user'], self.email_config['smtp_password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"邮件告警已发送: {alert.symbol} {alert.timeframe}")
            
        except Exception as e:
            self.logger.error(f"发送邮件告警失败: {e}")
    
    def _send_webhook_alert(self, alert: QualityAlert):
        """发送Webhook告警"""
        try:
            payload = {
                'symbol': alert.symbol,
                'timeframe': alert.timeframe,
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'system': 'DipMaster Data Infrastructure'
            }
            
            response = requests.post(
                self.webhook_config['url'],
                json=payload,
                headers=self.webhook_config.get('headers', {}),
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"Webhook告警已发送: {alert.symbol} {alert.timeframe}")
            else:
                self.logger.warning(f"Webhook告警发送失败: HTTP {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"发送Webhook告警失败: {e}")
    
    def generate_monitoring_dashboard(self) -> str:
        """生成监控仪表板"""
        # 创建仪表板HTML
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '数据质量趋势', '告警数量统计',
                '文件大小分布', '数据完整性热力图',
                '性能指标', '系统状态'
            ),
            specs=[
                [{"secondary_y": True}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "heatmap"}],
                [{"secondary_y": True}, {"type": "indicator"}]
            ]
        )
        
        # 1. 质量趋势图
        quality_data = self._get_quality_trend_data()
        if quality_data:
            fig.add_trace(
                go.Scatter(
                    x=quality_data['timestamps'],
                    y=quality_data['scores'],
                    name='质量评分',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        # 2. 告警统计
        alert_stats = self._get_alert_statistics()
        if alert_stats:
            fig.add_trace(
                go.Bar(
                    x=list(alert_stats.keys()),
                    y=list(alert_stats.values()),
                    name='告警数量',
                    marker_color='red'
                ),
                row=1, col=2
            )
        
        # 3. 文件大小分布
        file_sizes = self._get_file_size_distribution()
        if file_sizes:
            fig.add_trace(
                go.Histogram(
                    x=file_sizes,
                    name='文件大小分布',
                    nbinsx=20
                ),
                row=2, col=1
            )
        
        # 4. 完整性热力图
        completeness_matrix = self._get_completeness_heatmap_data()
        if completeness_matrix is not None:
            fig.add_trace(
                go.Heatmap(
                    z=completeness_matrix['values'],
                    x=completeness_matrix['timeframes'],
                    y=completeness_matrix['symbols'],
                    colorscale='RdYlGn',
                    name='数据完整性'
                ),
                row=2, col=2
            )
        
        # 5. 系统指标
        system_metrics = self._get_current_system_metrics()
        if system_metrics:
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=system_metrics.get('overall_health', 0),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "系统健康度"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 50], 'color': "lightgray"},
                               {'range': [50, 80], 'color': "yellow"},
                               {'range': [80, 100], 'color': "green"}
                           ]}
                ),
                row=3, col=2
            )
        
        # 更新布局
        fig.update_layout(
            title="DipMaster 数据基础设施监控仪表板",
            height=1200,
            showlegend=False
        )
        
        # 保存仪表板
        dashboard_path = Path("data/monitoring_dashboard.html")
        fig.write_html(str(dashboard_path))
        
        return str(dashboard_path)
    
    def _get_quality_trend_data(self) -> Optional[Dict[str, List]]:
        """获取质量趋势数据"""
        try:
            cursor = self.conn.execute("""
                SELECT timestamp, AVG(overall_score) as avg_score
                FROM quality_metrics 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY datetime(timestamp, 'start of hour')
                ORDER BY timestamp
            """)
            
            results = cursor.fetchall()
            if not results:
                return None
            
            timestamps = [datetime.fromisoformat(row[0]) for row in results]
            scores = [row[1] for row in results]
            
            return {'timestamps': timestamps, 'scores': scores}
        except:
            return None
    
    def _get_alert_statistics(self) -> Dict[str, int]:
        """获取告警统计"""
        try:
            cursor = self.conn.execute("""
                SELECT severity, COUNT(*) 
                FROM alerts 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY severity
            """)
            
            return dict(cursor.fetchall())
        except:
            return {}
    
    def _get_file_size_distribution(self) -> List[float]:
        """获取文件大小分布"""
        try:
            file_sizes = []
            for file_path in self.base_path.glob("*.parquet"):
                size_mb = file_path.stat().st_size / 1024 / 1024
                file_sizes.append(size_mb)
            return file_sizes
        except:
            return []
    
    def _get_completeness_heatmap_data(self) -> Optional[Dict[str, Any]]:
        """获取完整性热力图数据"""
        try:
            # 获取最新的完整性数据
            cursor = self.conn.execute("""
                SELECT symbol, timeframe, completeness
                FROM quality_metrics qm1
                WHERE timestamp = (
                    SELECT MAX(timestamp) 
                    FROM quality_metrics qm2 
                    WHERE qm2.symbol = qm1.symbol AND qm2.timeframe = qm1.timeframe
                )
            """)
            
            results = cursor.fetchall()
            if not results:
                return None
            
            # 组织数据为矩阵
            symbols = sorted(set(row[0] for row in results))
            timeframes = sorted(set(row[1] for row in results))
            
            matrix = np.zeros((len(symbols), len(timeframes)))
            
            for symbol, timeframe, completeness in results:
                i = symbols.index(symbol)
                j = timeframes.index(timeframe)
                matrix[i, j] = completeness
            
            return {
                'values': matrix,
                'symbols': symbols,
                'timeframes': timeframes
            }
        except:
            return None
    
    def _get_current_system_metrics(self) -> Dict[str, float]:
        """获取当前系统指标"""
        try:
            # 计算整体健康度
            total_files = len(list(self.base_path.glob("*.parquet")))
            expected_files = len(self.symbols_to_monitor) * len(self.timeframes_to_monitor)
            
            file_coverage = (total_files / expected_files) * 100 if expected_files > 0 else 0
            
            # 获取最近的平均质量分数
            cursor = self.conn.execute("""
                SELECT AVG(overall_score) 
                FROM quality_metrics 
                WHERE timestamp > datetime('now', '-1 hour')
            """)
            
            result = cursor.fetchone()
            avg_quality = result[0] * 100 if result[0] else 0
            
            # 计算告警权重
            cursor = self.conn.execute("""
                SELECT COUNT(*) 
                FROM alerts 
                WHERE timestamp > datetime('now', '-1 hour') AND severity IN ('high', 'critical')
            """)
            
            critical_alerts = cursor.fetchone()[0]
            alert_penalty = min(critical_alerts * 10, 50)  # 每个严重告警扣10分，最多扣50分
            
            overall_health = min(100, max(0, (file_coverage * 0.3 + avg_quality * 0.7 - alert_penalty)))
            
            return {
                'overall_health': overall_health,
                'file_coverage': file_coverage,
                'avg_quality': avg_quality,
                'critical_alerts': critical_alerts
            }
        except:
            return {'overall_health': 0}
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        now = datetime.now(timezone.utc)
        
        # 获取最近的统计数据
        recent_alerts = [alert for alert in self.alerts if (now - alert.timestamp).total_seconds() < 3600]
        
        # 按严重程度分组
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[alert.severity] += 1
        
        # 计算总体状态
        if alert_counts['critical'] > 0:
            overall_status = 'critical'
        elif alert_counts['high'] > 0:
            overall_status = 'degraded'
        elif alert_counts.get('warning', 0) + alert_counts.get('medium', 0) > 5:
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'
        
        return {
            'timestamp': now.isoformat(),
            'monitoring_active': self.monitoring_active,
            'overall_status': overall_status,
            'recent_alerts': {
                'total': len(recent_alerts),
                'by_severity': dict(alert_counts)
            },
            'data_coverage': {
                'expected_files': len(self.symbols_to_monitor) * len(self.timeframes_to_monitor),
                'actual_files': len(list(self.base_path.glob("*.parquet"))),
                'coverage_percentage': (len(list(self.base_path.glob("*.parquet"))) / 
                                       (len(self.symbols_to_monitor) * len(self.timeframes_to_monitor)) * 100)
            },
            'system_metrics': self._get_current_system_metrics()
        }

# 使用示例
def create_monitor_with_config():
    """创建带配置的监控器"""
    config = {
        'base_path': 'data/enhanced_market_data',
        'monitoring_interval_seconds': 300,
        'email': {
            'enabled': False,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'smtp_user': 'your_email@gmail.com',
            'smtp_password': 'your_password',
            'recipients': ['admin@example.com']
        },
        'webhook': {
            'enabled': False,
            'url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
            'headers': {'Content-Type': 'application/json'}
        }
    }
    
    return DataInfrastructureMonitor(config)