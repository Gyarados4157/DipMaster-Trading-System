"""
数据质量监控系统 - Data Quality Monitoring System
为DipMaster Trading System提供全面的数据质量保障

Features:
- 实时数据质量监控
- 异常检测和自动修复
- 数据完整性验证
- 质量报告生成
- 警报和通知系统
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import threading
from collections import defaultdict, deque
import statistics
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

@dataclass
class QualityIssue:
    """质量问题记录"""
    issue_id: str
    timestamp: datetime
    symbol: str
    issue_type: str  # 'missing_data', 'outlier', 'inconsistency', 'latency'
    severity: str    # 'low', 'medium', 'high', 'critical'
    description: str
    value: Any
    expected_value: Any = None
    auto_fixed: bool = False
    
@dataclass 
class QualityMetrics:
    """质量指标"""
    timestamp: datetime
    symbol: str
    timeframe: str
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    timeliness_score: float
    overall_score: float
    record_count: int
    anomaly_count: int
    
@dataclass
class DataHealthStatus:
    """数据健康状态"""
    timestamp: datetime
    overall_health: str  # 'healthy', 'warning', 'critical'
    active_symbols: int
    quality_score: float
    issues_count: Dict[str, int]
    last_update: datetime

class OutlierDetector:
    """异常值检测器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.OutlierDetector")
        self.models = {}  # 每个symbol维护一个模型
        
    def detect_price_outliers(self, df: pd.DataFrame, symbol: str) -> List[int]:
        """检测价格异常值"""
        if df.empty or len(df) < 10:
            return []
            
        # 计算价格变化率
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_zscore'] = np.abs((df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std())
        
        outliers = []
        
        # 1. 极端价格变化
        extreme_returns = df[np.abs(df['returns']) > 0.5].index.tolist()
        outliers.extend(extreme_returns)
        
        # 2. 价格Z-score异常
        zscore_outliers = df[df['price_zscore'] > 4].index.tolist()
        outliers.extend(zscore_outliers)
        
        # 3. 使用Isolation Forest
        try:
            features = df[['open', 'high', 'low', 'close', 'volume']].fillna(method='ffill')
            if len(features) >= 10:
                if symbol not in self.models:
                    self.models[symbol] = IsolationForest(contamination=0.1, random_state=42)
                    
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                outlier_scores = self.models[symbol].fit_predict(features_scaled)
                isolation_outliers = df[outlier_scores == -1].index.tolist()
                outliers.extend(isolation_outliers)
                
        except Exception as e:
            self.logger.warning(f"Isolation Forest检测失败 {symbol}: {e}")
            
        return list(set(outliers))
        
    def detect_volume_outliers(self, df: pd.DataFrame) -> List[int]:
        """检测成交量异常值"""
        if df.empty or len(df) < 10:
            return []
            
        df = df.copy()
        
        # 计算成交量统计
        volume_mean = df['volume'].rolling(20).mean()
        volume_std = df['volume'].rolling(20).std()
        volume_zscore = np.abs((df['volume'] - volume_mean) / volume_std)
        
        # 异常成交量（Z-score > 5）
        volume_outliers = df[volume_zscore > 5].index.tolist()
        
        # 零成交量检测
        zero_volume = df[df['volume'] == 0].index.tolist()
        
        return list(set(volume_outliers + zero_volume))
        
    def detect_ohlc_inconsistencies(self, df: pd.DataFrame) -> List[int]:
        """检测OHLC不一致性"""
        if df.empty:
            return []
            
        inconsistencies = []
        
        # High < max(Open, Close)
        high_issues = df[df['high'] < df[['open', 'close']].max(axis=1)].index.tolist()
        inconsistencies.extend(high_issues)
        
        # Low > min(Open, Close)
        low_issues = df[df['low'] > df[['open', 'close']].min(axis=1)].index.tolist()
        inconsistencies.extend(low_issues)
        
        # High < Low
        hl_issues = df[df['high'] < df['low']].index.tolist()
        inconsistencies.extend(hl_issues)
        
        return list(set(inconsistencies))

class DataCompleteness:
    """数据完整性检查器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DataCompleteness")
        
    def check_time_gaps(self, df: pd.DataFrame, expected_interval: str = '5T') -> List[Tuple[datetime, datetime]]:
        """检查时间缺口"""
        if df.empty or len(df) < 2:
            return []
            
        # 生成期望的时间序列
        start_time = df.index.min()
        end_time = df.index.max()
        expected_times = pd.date_range(start=start_time, end=end_time, freq=expected_interval)
        
        # 找出缺失的时间点
        missing_times = expected_times.difference(df.index)
        
        # 将连续的缺失时间合并为时间段
        gaps = []
        if len(missing_times) > 0:
            current_start = missing_times[0]
            current_end = missing_times[0]
            
            for i in range(1, len(missing_times)):
                if missing_times[i] - current_end <= pd.Timedelta(expected_interval):
                    current_end = missing_times[i]
                else:
                    gaps.append((current_start, current_end))
                    current_start = missing_times[i]
                    current_end = missing_times[i]
            
            gaps.append((current_start, current_end))
            
        return gaps
        
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """检查缺失值"""
        if df.empty:
            return {}
            
        missing_counts = df.isnull().sum().to_dict()
        return {col: count for col, count in missing_counts.items() if count > 0}
        
    def check_duplicate_records(self, df: pd.DataFrame) -> int:
        """检查重复记录"""
        if df.empty:
            return 0
            
        return df.duplicated().sum()
        
    def calculate_completeness_score(self, df: pd.DataFrame, expected_interval: str = '5T') -> float:
        """计算完整性评分"""
        if df.empty:
            return 0.0
            
        # 时间完整性
        gaps = self.check_time_gaps(df, expected_interval)
        total_expected_periods = len(pd.date_range(
            start=df.index.min(), 
            end=df.index.max(), 
            freq=expected_interval
        ))
        missing_periods = sum((gap[1] - gap[0]).total_seconds() / 
                            pd.Timedelta(expected_interval).total_seconds() + 1 
                            for gap in gaps)
        time_completeness = 1 - (missing_periods / total_expected_periods) if total_expected_periods > 0 else 1
        
        # 数据完整性
        missing_values = self.check_missing_values(df)
        total_values = len(df) * len(df.columns)
        missing_value_count = sum(missing_values.values())
        data_completeness = 1 - (missing_value_count / total_values) if total_values > 0 else 1
        
        # 综合评分
        return (time_completeness * 0.6 + data_completeness * 0.4)

class DataQualityMonitor:
    """数据质量监控器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 检测器
        self.outlier_detector = OutlierDetector()
        self.completeness_checker = DataCompleteness()
        
        # 质量阈值
        self.quality_thresholds = self.config.get('quality_thresholds', {
            'completeness': 0.95,
            'accuracy': 0.98,
            'consistency': 0.99,
            'timeliness': 0.90,
            'overall': 0.95
        })
        
        # 问题存储
        self.quality_issues = deque(maxlen=10000)
        self.quality_metrics_history = deque(maxlen=5000)
        
        # 数据库连接
        self.db_path = Path(self.config.get('db_path', 'data/quality_monitor.db'))
        self._init_database()
        
        # 自动修复开关
        self.auto_repair_enabled = self.config.get('auto_repair', True)
        
        # 监控状态
        self.monitoring_active = False
        self.last_check = {}
        
    def _init_database(self):
        """初始化数据库"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # 质量问题表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS quality_issues (
                    issue_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    symbol TEXT,
                    issue_type TEXT,
                    severity TEXT,
                    description TEXT,
                    value TEXT,
                    expected_value TEXT,
                    auto_fixed INTEGER
                )
            ''')
            
            # 质量指标表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    timeframe TEXT,
                    completeness_score REAL,
                    accuracy_score REAL,
                    consistency_score REAL,
                    timeliness_score REAL,
                    overall_score REAL,
                    record_count INTEGER,
                    anomaly_count INTEGER
                )
            ''')
            
            conn.commit()
            
    def assess_data_quality(self, df: pd.DataFrame, symbol: str, timeframe: str = '5m') -> QualityMetrics:
        """评估数据质量"""
        timestamp = datetime.now(timezone.utc)
        
        if df.empty:
            return QualityMetrics(
                timestamp=timestamp,
                symbol=symbol,
                timeframe=timeframe,
                completeness_score=0.0,
                accuracy_score=0.0,
                consistency_score=0.0,
                timeliness_score=0.0,
                overall_score=0.0,
                record_count=0,
                anomaly_count=0
            )
        
        # 1. 完整性检查
        completeness_score = self.completeness_checker.calculate_completeness_score(df)
        
        # 2. 准确性检查（异常值检测）
        price_outliers = self.outlier_detector.detect_price_outliers(df, symbol)
        volume_outliers = self.outlier_detector.detect_volume_outliers(df)
        total_outliers = len(set(price_outliers + volume_outliers))
        accuracy_score = max(0, 1 - (total_outliers / len(df))) if len(df) > 0 else 1
        
        # 3. 一致性检查
        ohlc_issues = self.outlier_detector.detect_ohlc_inconsistencies(df)
        consistency_score = max(0, 1 - (len(ohlc_issues) / len(df))) if len(df) > 0 else 1
        
        # 4. 时效性检查
        latest_time = df.index.max()
        current_time = pd.Timestamp.now(tz='UTC')
        delay_hours = (current_time - latest_time).total_seconds() / 3600
        
        if delay_hours <= 1:
            timeliness_score = 1.0
        elif delay_hours <= 24:
            timeliness_score = 1.0 - (delay_hours - 1) * 0.5 / 23
        else:
            timeliness_score = 0.0
            
        # 5. 综合评分
        scores = [completeness_score, accuracy_score, consistency_score, timeliness_score]
        overall_score = np.mean(scores)
        
        metrics = QualityMetrics(
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            timeliness_score=timeliness_score,
            overall_score=overall_score,
            record_count=len(df),
            anomaly_count=total_outliers
        )
        
        # 记录历史
        self.quality_metrics_history.append(metrics)
        
        # 保存到数据库
        self._save_quality_metrics(metrics)
        
        # 检查问题
        self._check_quality_issues(df, symbol, timeframe, metrics)
        
        return metrics
        
    def _check_quality_issues(self, df: pd.DataFrame, symbol: str, timeframe: str, metrics: QualityMetrics):
        """检查质量问题"""
        issues = []
        
        # 检查各项评分
        if metrics.completeness_score < self.quality_thresholds['completeness']:
            issue = QualityIssue(
                issue_id=f"{symbol}_{timeframe}_completeness_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                issue_type='completeness',
                severity='high' if metrics.completeness_score < 0.8 else 'medium',
                description=f"数据完整性低于阈值: {metrics.completeness_score:.3f} < {self.quality_thresholds['completeness']}",
                value=metrics.completeness_score,
                expected_value=self.quality_thresholds['completeness']
            )
            issues.append(issue)
            
        if metrics.accuracy_score < self.quality_thresholds['accuracy']:
            issue = QualityIssue(
                issue_id=f"{symbol}_{timeframe}_accuracy_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                issue_type='accuracy',
                severity='high' if metrics.accuracy_score < 0.9 else 'medium',
                description=f"数据准确性低于阈值: {metrics.accuracy_score:.3f} < {self.quality_thresholds['accuracy']}",
                value=metrics.accuracy_score,
                expected_value=self.quality_thresholds['accuracy']
            )
            issues.append(issue)
            
        if metrics.consistency_score < self.quality_thresholds['consistency']:
            issue = QualityIssue(
                issue_id=f"{symbol}_{timeframe}_consistency_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                issue_type='consistency',
                severity='critical' if metrics.consistency_score < 0.95 else 'high',
                description=f"数据一致性低于阈值: {metrics.consistency_score:.3f} < {self.quality_thresholds['consistency']}",
                value=metrics.consistency_score,
                expected_value=self.quality_thresholds['consistency']
            )
            issues.append(issue)
            
        if metrics.timeliness_score < self.quality_thresholds['timeliness']:
            issue = QualityIssue(
                issue_id=f"{symbol}_{timeframe}_timeliness_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                issue_type='timeliness',
                severity='medium',
                description=f"数据时效性低于阈值: {metrics.timeliness_score:.3f} < {self.quality_thresholds['timeliness']}",
                value=metrics.timeliness_score,
                expected_value=self.quality_thresholds['timeliness']
            )
            issues.append(issue)
            
        # 保存问题
        for issue in issues:
            self.quality_issues.append(issue)
            self._save_quality_issue(issue)
            
            # 如果启用自动修复
            if self.auto_repair_enabled and issue.issue_type in ['completeness', 'consistency']:
                try:
                    self._attempt_auto_repair(df, issue)
                except Exception as e:
                    self.logger.error(f"自动修复失败 {issue.issue_id}: {e}")
                    
    def _attempt_auto_repair(self, df: pd.DataFrame, issue: QualityIssue) -> bool:
        """尝试自动修复"""
        self.logger.info(f"尝试自动修复: {issue.issue_id}")
        
        success = False
        
        if issue.issue_type == 'completeness':
            # 填充缺失值
            df_repaired = df.fillna(method='ffill').fillna(method='bfill')
            success = not df_repaired.isnull().any().any()
            
        elif issue.issue_type == 'consistency':
            # 修复OHLC关系
            df_repaired = df.copy()
            df_repaired['high'] = df_repaired[['high', 'open', 'close']].max(axis=1)
            df_repaired['low'] = df_repaired[['low', 'open', 'close']].min(axis=1)
            success = len(self.outlier_detector.detect_ohlc_inconsistencies(df_repaired)) == 0
            
        if success:
            issue.auto_fixed = True
            self.logger.info(f"自动修复成功: {issue.issue_id}")
        else:
            self.logger.warning(f"自动修复失败: {issue.issue_id}")
            
        return success
        
    def _save_quality_metrics(self, metrics: QualityMetrics):
        """保存质量指标"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO quality_metrics 
                (timestamp, symbol, timeframe, completeness_score, accuracy_score, 
                 consistency_score, timeliness_score, overall_score, record_count, anomaly_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.symbol,
                metrics.timeframe,
                metrics.completeness_score,
                metrics.accuracy_score,
                metrics.consistency_score,
                metrics.timeliness_score,
                metrics.overall_score,
                metrics.record_count,
                metrics.anomaly_count
            ))
            
    def _save_quality_issue(self, issue: QualityIssue):
        """保存质量问题"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO quality_issues 
                (issue_id, timestamp, symbol, issue_type, severity, description, 
                 value, expected_value, auto_fixed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                issue.issue_id,
                issue.timestamp.isoformat(),
                issue.symbol,
                issue.issue_type,
                issue.severity,
                issue.description,
                json.dumps(issue.value),
                json.dumps(issue.expected_value),
                1 if issue.auto_fixed else 0
            ))
            
    def get_quality_report(self, symbol: str = None, days: int = 7) -> Dict[str, Any]:
        """生成质量报告"""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # 查询质量指标
            query = '''
                SELECT * FROM quality_metrics 
                WHERE timestamp >= ? 
            '''
            params = [start_time.isoformat()]
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
                
            query += ' ORDER BY timestamp DESC'
            
            metrics_df = pd.read_sql_query(query, conn, params=params)
            
            # 查询质量问题
            issues_query = '''
                SELECT * FROM quality_issues 
                WHERE timestamp >= ?
            '''
            issues_params = [start_time.isoformat()]
            
            if symbol:
                issues_query += ' AND symbol = ?'
                issues_params.append(symbol)
                
            issues_query += ' ORDER BY timestamp DESC'
            
            issues_df = pd.read_sql_query(issues_query, conn, params=issues_params)
            
        # 生成报告
        report = {
            'report_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'days': days
            },
            'summary': {},
            'metrics_by_symbol': {},
            'issues_summary': {},
            'recommendations': []
        }
        
        if not metrics_df.empty:
            # 总体统计
            report['summary'] = {
                'total_assessments': len(metrics_df),
                'average_overall_score': float(metrics_df['overall_score'].mean()),
                'average_completeness': float(metrics_df['completeness_score'].mean()),
                'average_accuracy': float(metrics_df['accuracy_score'].mean()),
                'average_consistency': float(metrics_df['consistency_score'].mean()),
                'average_timeliness': float(metrics_df['timeliness_score'].mean()),
                'symbols_monitored': len(metrics_df['symbol'].unique())
            }
            
            # 按币种统计
            for sym in metrics_df['symbol'].unique():
                sym_data = metrics_df[metrics_df['symbol'] == sym]
                report['metrics_by_symbol'][sym] = {
                    'assessments_count': len(sym_data),
                    'overall_score': float(sym_data['overall_score'].mean()),
                    'completeness_score': float(sym_data['completeness_score'].mean()),
                    'accuracy_score': float(sym_data['accuracy_score'].mean()),
                    'consistency_score': float(sym_data['consistency_score'].mean()),
                    'timeliness_score': float(sym_data['timeliness_score'].mean()),
                    'total_anomalies': int(sym_data['anomaly_count'].sum())
                }
                
        if not issues_df.empty:
            # 问题统计
            report['issues_summary'] = {
                'total_issues': len(issues_df),
                'issues_by_type': issues_df['issue_type'].value_counts().to_dict(),
                'issues_by_severity': issues_df['severity'].value_counts().to_dict(),
                'auto_fixed_count': int(issues_df['auto_fixed'].sum()),
                'recent_issues': issues_df.head(10).to_dict('records')
            }
            
        # 生成建议
        if report['summary']:
            avg_score = report['summary']['average_overall_score']
            if avg_score < 0.8:
                report['recommendations'].append("整体数据质量较低，建议检查数据源和采集流程")
            elif avg_score < 0.9:
                report['recommendations'].append("数据质量有改进空间，建议优化数据清洗流程")
            else:
                report['recommendations'].append("数据质量良好，保持当前监控频率")
                
        return report
        
    def generate_quality_dashboard(self, symbol: str = None, days: int = 7) -> str:
        """生成质量监控仪表板"""
        report = self.get_quality_report(symbol, days)
        
        # 创建图表
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('质量评分趋势', '问题类型分布', '各维度评分', '异常数量趋势'),
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"type": "bar"}, {"secondary_y": False}]]
        )
        
        # 时间序列图需要实际数据
        with sqlite3.connect(self.db_path) as conn:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days)
            
            query = '''
                SELECT timestamp, symbol, overall_score, anomaly_count 
                FROM quality_metrics 
                WHERE timestamp >= ?
            '''
            params = [start_time.isoformat()]
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
                
            query += ' ORDER BY timestamp'
            
            data = pd.read_sql_query(query, conn, params=params)
            
        if not data.empty:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # 质量评分趋势
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['overall_score'],
                    mode='lines+markers',
                    name='质量评分',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # 异常数量趋势
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['anomaly_count'],
                    mode='lines+markers',
                    name='异常数量',
                    line=dict(color='red')
                ),
                row=2, col=2
            )
            
        # 问题类型分布
        if 'issues_summary' in report and report['issues_summary']:
            issues_by_type = report['issues_summary']['issues_by_type']
            fig.add_trace(
                go.Pie(
                    labels=list(issues_by_type.keys()),
                    values=list(issues_by_type.values()),
                    name="问题类型"
                ),
                row=1, col=2
            )
            
        # 各维度评分
        if 'summary' in report:
            dimensions = ['completeness', 'accuracy', 'consistency', 'timeliness']
            scores = [report['summary'].get(f'average_{dim}', 0) for dim in dimensions]
            
            fig.add_trace(
                go.Bar(
                    x=dimensions,
                    y=scores,
                    name="维度评分",
                    marker_color=['green' if score > 0.9 else 'orange' if score > 0.8 else 'red' for score in scores]
                ),
                row=2, col=1
            )
            
        fig.update_layout(
            title="数据质量监控仪表板",
            height=800,
            showlegend=True
        )
        
        # 保存到文件
        dashboard_path = Path("data/quality_dashboard.html")
        fig.write_html(dashboard_path)
        
        return str(dashboard_path)
        
    def get_health_status(self) -> DataHealthStatus:
        """获取数据健康状态"""
        # 获取最近的质量指标
        recent_metrics = list(self.quality_metrics_history)[-100:] if self.quality_metrics_history else []
        
        if not recent_metrics:
            return DataHealthStatus(
                timestamp=datetime.now(timezone.utc),
                overall_health='unknown',
                active_symbols=0,
                quality_score=0.0,
                issues_count={},
                last_update=datetime.now(timezone.utc)
            )
            
        # 计算平均质量分数
        avg_quality = np.mean([m.overall_score for m in recent_metrics])
        
        # 确定健康状态
        if avg_quality >= 0.95:
            health = 'healthy'
        elif avg_quality >= 0.85:
            health = 'warning'
        else:
            health = 'critical'
            
        # 统计问题
        recent_issues = list(self.quality_issues)[-100:] if self.quality_issues else []
        issues_count = defaultdict(int)
        for issue in recent_issues:
            issues_count[issue.issue_type] += 1
            
        # 活跃币种数
        active_symbols = len(set(m.symbol for m in recent_metrics))
        
        return DataHealthStatus(
            timestamp=datetime.now(timezone.utc),
            overall_health=health,
            active_symbols=active_symbols,
            quality_score=avg_quality,
            issues_count=dict(issues_count),
            last_update=recent_metrics[-1].timestamp if recent_metrics else datetime.now(timezone.utc)
        )

# 使用示例
async def main():
    """主函数示例"""
    # 创建质量监控器
    config = {
        'quality_thresholds': {
            'completeness': 0.95,
            'accuracy': 0.98,
            'consistency': 0.99,
            'timeliness': 0.90,
            'overall': 0.95
        },
        'auto_repair': True,
        'db_path': 'data/quality_monitor.db'
    }
    
    monitor = DataQualityMonitor(config)
    
    # 模拟数据质量评估
    # 创建示例数据
    dates = pd.date_range('2025-01-01', periods=1000, freq='5T')
    data = {
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 101,
        'volume': np.random.exponential(1000, 1000)
    }
    df = pd.DataFrame(data, index=dates)
    
    # 评估质量
    metrics = monitor.assess_data_quality(df, 'BTCUSDT', '5m')
    print(f"质量评分: {metrics.overall_score:.3f}")
    
    # 生成报告
    report = monitor.get_quality_report('BTCUSDT', 1)
    print(f"报告摘要: {report['summary']}")
    
    # 生成仪表板
    dashboard_path = monitor.generate_quality_dashboard('BTCUSDT', 7)
    print(f"仪表板保存至: {dashboard_path}")
    
    # 获取健康状态
    health = monitor.get_health_status()
    print(f"数据健康状态: {health.overall_health}")

if __name__ == "__main__":
    asyncio.run(main())