#!/usr/bin/env python3
"""
DipMaster Trading System - Quality Assurance and Consistency Monitoring System
质量保证系统 - 信号一致性监控、回测对比和模型衰减检测

Author: DipMaster Trading System Monitoring Agent
Date: 2025-08-18
Version: 1.0.0
"""

import asyncio
import json
import time
import logging
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import statistics
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConsistencyLevel(Enum):
    """一致性级别"""
    EXCELLENT = "excellent"    # 95%+
    GOOD = "good"             # 85-95%
    ACCEPTABLE = "acceptable"  # 70-85%
    POOR = "poor"             # 50-70%
    CRITICAL = "critical"     # <50%


class QualityMetric(Enum):
    """质量指标类型"""
    SIGNAL_POSITION_CONSISTENCY = "signal_position_consistency"
    POSITION_EXECUTION_CONSISTENCY = "position_execution_consistency"
    BACKTEST_LIVE_DRIFT = "backtest_live_drift"
    FEATURE_DRIFT = "feature_drift"
    MODEL_PERFORMANCE_DRIFT = "model_performance_drift"
    DATA_QUALITY = "data_quality"
    TIMING_CONSISTENCY = "timing_consistency"
    PRICE_CONSISTENCY = "price_consistency"


@dataclass
class SignalRecord:
    """交易信号记录"""
    signal_id: str
    timestamp: datetime
    symbol: str
    signal_type: str  # BUY/SELL
    confidence: float
    price: float
    technical_indicators: Dict[str, float]
    expected_entry_price: float
    expected_holding_minutes: int
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_hash(self) -> str:
        """生成信号哈希用于去重"""
        data = f"{self.symbol}_{self.signal_type}_{self.price}_{self.confidence}_{self.timestamp.isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()


@dataclass
class PositionRecord:
    """持仓记录"""
    position_id: str
    signal_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    realized: bool = False
    holding_minutes: Optional[int] = None
    execution_slippage: float = 0.0
    
    def calculate_consistency_metrics(self, signal: SignalRecord) -> Dict[str, float]:
        """计算与信号的一致性指标"""
        metrics = {}
        
        # 价格一致性
        if signal.expected_entry_price > 0:
            price_diff = abs(self.entry_price - signal.expected_entry_price) / signal.expected_entry_price
            metrics['price_consistency'] = max(0, 1 - price_diff)
        
        # 时间一致性
        if self.holding_minutes and signal.expected_holding_minutes > 0:
            time_diff = abs(self.holding_minutes - signal.expected_holding_minutes) / signal.expected_holding_minutes
            metrics['timing_consistency'] = max(0, 1 - time_diff)
        
        # 方向一致性
        metrics['direction_consistency'] = 1.0 if self.side == signal.signal_type else 0.0
        
        return metrics


@dataclass
class ExecutionRecord:
    """执行记录"""
    execution_id: str
    position_id: str
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    slippage_bps: float
    latency_ms: float
    venue: str
    status: str


@dataclass
class QualityReport:
    """质量报告"""
    report_id: str
    timestamp: datetime
    time_window: str
    quality_scores: Dict[QualityMetric, float]
    consistency_level: ConsistencyLevel
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    detailed_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'report_id': self.report_id,
            'timestamp': self.timestamp.isoformat(),
            'time_window': self.time_window,
            'quality_scores': {metric.value: score for metric, score in self.quality_scores.items()},
            'consistency_level': self.consistency_level.value,
            'violations': self.violations,
            'recommendations': self.recommendations,
            'detailed_metrics': self.detailed_metrics
        }


class SignalConsistencyMonitor:
    """信号一致性监控器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.signals: Dict[str, SignalRecord] = {}
        self.positions: Dict[str, PositionRecord] = {}
        self.executions: Dict[str, List[ExecutionRecord]] = {}
        self.consistency_history: List[Dict[str, Any]] = []
        
        # 一致性阈值
        self.thresholds = {
            'signal_position_match': self.config.get('signal_position_match_threshold', 0.95),
            'position_execution_match': self.config.get('position_execution_match_threshold', 0.98),
            'price_deviation_bps': self.config.get('price_deviation_bps', 20.0),
            'timing_deviation_minutes': self.config.get('timing_deviation_minutes', 2.0)
        }
    
    async def record_signal(self, signal: SignalRecord) -> None:
        """记录交易信号"""
        self.signals[signal.signal_id] = signal
        logger.debug(f"📊 Recorded signal: {signal.signal_id}")
    
    async def record_position(self, position: PositionRecord) -> None:
        """记录持仓"""
        self.positions[position.position_id] = position
        
        # 如果持仓有对应信号，计算一致性
        if position.signal_id in self.signals:
            await self._check_signal_position_consistency(position)
        
        logger.debug(f"📊 Recorded position: {position.position_id}")
    
    async def record_execution(self, execution: ExecutionRecord) -> None:
        """记录执行"""
        if execution.position_id not in self.executions:
            self.executions[execution.position_id] = []
        
        self.executions[execution.position_id].append(execution)
        
        # 检查执行一致性
        if execution.position_id in self.positions:
            await self._check_position_execution_consistency(execution)
        
        logger.debug(f"📊 Recorded execution: {execution.execution_id}")
    
    async def _check_signal_position_consistency(self, position: PositionRecord) -> Dict[str, Any]:
        """检查信号与持仓的一致性"""
        signal = self.signals.get(position.signal_id)
        if not signal:
            return {}
        
        consistency_metrics = position.calculate_consistency_metrics(signal)
        
        # 记录一致性检查结果
        consistency_result = {
            'timestamp': datetime.now(timezone.utc),
            'signal_id': signal.signal_id,
            'position_id': position.position_id,
            'metrics': consistency_metrics,
            'overall_score': statistics.mean(consistency_metrics.values()) if consistency_metrics else 0.0
        }
        
        self.consistency_history.append(consistency_result)
        
        # 检查是否有违规
        if consistency_result['overall_score'] < self.thresholds['signal_position_match']:
            logger.warning(f"⚠️ Signal-Position consistency violation: {consistency_result['overall_score']:.3f}")
        
        return consistency_result
    
    async def _check_position_execution_consistency(self, execution: ExecutionRecord) -> Dict[str, Any]:
        """检查持仓与执行的一致性"""
        position = self.positions.get(execution.position_id)
        if not position:
            return {}
        
        # 计算执行质量指标
        price_consistency = 1.0 - abs(execution.price - position.entry_price) / position.entry_price
        quantity_consistency = 1.0 if execution.quantity == position.quantity else 0.8
        direction_consistency = 1.0 if execution.side == position.side else 0.0
        
        execution_metrics = {
            'price_consistency': max(0, price_consistency),
            'quantity_consistency': quantity_consistency,
            'direction_consistency': direction_consistency,
            'slippage_quality': max(0, 1 - execution.slippage_bps / 50.0),  # 50bps为满分基准
            'latency_quality': max(0, 1 - execution.latency_ms / 1000.0)    # 1000ms为满分基准
        }
        
        execution_result = {
            'timestamp': datetime.now(timezone.utc),
            'position_id': position.position_id,
            'execution_id': execution.execution_id,
            'metrics': execution_metrics,
            'overall_score': statistics.mean(execution_metrics.values())
        }
        
        # 检查是否有违规
        if execution_result['overall_score'] < self.thresholds['position_execution_match']:
            logger.warning(f"⚠️ Position-Execution consistency violation: {execution_result['overall_score']:.3f}")
        
        return execution_result
    
    def get_consistency_report(self, hours: int = 24) -> Dict[str, Any]:
        """生成一致性报告"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_checks = [
            check for check in self.consistency_history
            if check['timestamp'] >= cutoff_time
        ]
        
        if not recent_checks:
            return {
                'signal_position_consistency': 0.0,
                'average_score': 0.0,
                'total_checks': 0,
                'violations': 0
            }
        
        # 计算平均一致性分数
        average_score = statistics.mean([check['overall_score'] for check in recent_checks])
        
        # 统计违规数量
        violations = sum(
            1 for check in recent_checks
            if check['overall_score'] < self.thresholds['signal_position_match']
        )
        
        return {
            'signal_position_consistency': average_score,
            'average_score': average_score,
            'total_checks': len(recent_checks),
            'violations': violations,
            'violation_rate': violations / len(recent_checks) if recent_checks else 0,
            'recent_checks': recent_checks[-10:]  # 最近10次检查
        }


class BacktestLiveDriftDetector:
    """回测与实盘漂移检测器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.backtest_baseline: Dict[str, Any] = {}
        self.live_performance: Dict[str, List[float]] = {
            'win_rate': [],
            'avg_return': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'avg_holding_time': [],
            'trade_frequency': []
        }
        self.drift_alerts: List[Dict[str, Any]] = []
        
        # 漂移检测阈值
        self.drift_thresholds = {
            'minimal': self.config.get('minimal_drift_threshold', 0.05),    # 5%
            'moderate': self.config.get('moderate_drift_threshold', 0.10),  # 10%
            'significant': self.config.get('significant_drift_threshold', 0.20), # 20%
            'critical': self.config.get('critical_drift_threshold', 0.30)   # 30%
        }
    
    def set_backtest_baseline(self, baseline_metrics: Dict[str, float]) -> None:
        """设置回测基准指标"""
        self.backtest_baseline = baseline_metrics
        logger.info(f"📊 Set backtest baseline: {baseline_metrics}")
    
    async def record_live_performance(self, performance_metrics: Dict[str, float]) -> None:
        """记录实盘性能指标"""
        for metric, value in performance_metrics.items():
            if metric in self.live_performance:
                self.live_performance[metric].append(value)
                
                # 保持数据窗口大小
                max_window = self.config.get('performance_window_size', 1000)
                if len(self.live_performance[metric]) > max_window:
                    self.live_performance[metric] = self.live_performance[metric][-max_window:]
    
    async def detect_drift(self) -> Dict[str, Any]:
        """检测性能漂移"""
        if not self.backtest_baseline:
            return {'error': 'No backtest baseline set'}
        
        drift_analysis = {}
        drift_detected = False
        
        for metric, live_values in self.live_performance.items():
            if not live_values or metric not in self.backtest_baseline:
                continue
            
            baseline_value = self.backtest_baseline[metric]
            recent_avg = statistics.mean(live_values[-min(100, len(live_values)):])  # 最近100个数据点
            
            # 计算相对偏差
            if baseline_value != 0:
                relative_drift = abs(recent_avg - baseline_value) / abs(baseline_value)
            else:
                relative_drift = abs(recent_avg)
            
            # 判断漂移级别
            drift_level = self._classify_drift_level(relative_drift)
            
            drift_analysis[metric] = {
                'baseline': baseline_value,
                'recent_average': recent_avg,
                'absolute_drift': recent_avg - baseline_value,
                'relative_drift': relative_drift,
                'drift_level': drift_level,
                'sample_size': len(live_values)
            }
            
            # 如果检测到显著漂移，记录告警
            if drift_level in ['significant', 'critical']:
                drift_detected = True
                await self._record_drift_alert(metric, drift_analysis[metric])
        
        # 计算整体漂移评分
        overall_drift = statistics.mean([
            analysis['relative_drift']
            for analysis in drift_analysis.values()
        ]) if drift_analysis else 0.0
        
        return {
            'overall_drift_score': overall_drift,
            'overall_drift_level': self._classify_drift_level(overall_drift),
            'drift_detected': drift_detected,
            'metric_analysis': drift_analysis,
            'recommendations': self._generate_drift_recommendations(drift_analysis)
        }
    
    def _classify_drift_level(self, relative_drift: float) -> str:
        """分类漂移级别"""
        if relative_drift >= self.drift_thresholds['critical']:
            return 'critical'
        elif relative_drift >= self.drift_thresholds['significant']:
            return 'significant'
        elif relative_drift >= self.drift_thresholds['moderate']:
            return 'moderate'
        elif relative_drift >= self.drift_thresholds['minimal']:
            return 'minimal'
        else:
            return 'none'
    
    async def _record_drift_alert(self, metric: str, analysis: Dict[str, Any]) -> None:
        """记录漂移告警"""
        alert = {
            'timestamp': datetime.now(timezone.utc),
            'metric': metric,
            'drift_level': analysis['drift_level'],
            'relative_drift': analysis['relative_drift'],
            'baseline': analysis['baseline'],
            'current': analysis['recent_average']
        }
        
        self.drift_alerts.append(alert)
        logger.warning(f"🚨 Drift detected - {metric}: {analysis['drift_level']} level ({analysis['relative_drift']:.3f})")
    
    def _generate_drift_recommendations(self, drift_analysis: Dict[str, Any]) -> List[str]:
        """生成漂移修复建议"""
        recommendations = []
        
        for metric, analysis in drift_analysis.items():
            if analysis['drift_level'] in ['significant', 'critical']:
                if metric == 'win_rate' and analysis['recent_average'] < analysis['baseline']:
                    recommendations.append(f"Win rate degraded: Review signal quality and strategy parameters")
                elif metric == 'sharpe_ratio' and analysis['recent_average'] < analysis['baseline']:
                    recommendations.append(f"Risk-adjusted returns declined: Consider reducing position sizes")
                elif metric == 'max_drawdown' and analysis['recent_average'] > analysis['baseline']:
                    recommendations.append(f"Drawdown exceeded baseline: Implement tighter risk controls")
                elif metric == 'avg_holding_time' and abs(analysis['relative_drift']) > 0.3:
                    recommendations.append(f"Holding time pattern changed: Verify exit timing logic")
        
        # 通用建议
        if any(analysis['drift_level'] == 'critical' for analysis in drift_analysis.values()):
            recommendations.insert(0, "CRITICAL: Stop trading and investigate immediately")
        elif any(analysis['drift_level'] == 'significant' for analysis in drift_analysis.values()):
            recommendations.insert(0, "Reduce position sizes and monitor closely")
        
        return recommendations[:5]  # 限制建议数量


class FeatureDriftMonitor:
    """特征漂移监控器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.feature_baselines: Dict[str, Dict[str, float]] = {}
        self.feature_history: Dict[str, List[float]] = {}
        self.drift_scores: Dict[str, float] = {}
    
    def set_feature_baseline(self, feature_name: str, baseline_stats: Dict[str, float]) -> None:
        """设置特征基准统计"""
        self.feature_baselines[feature_name] = baseline_stats
        logger.debug(f"📊 Set baseline for feature {feature_name}")
    
    async def record_feature_value(self, feature_name: str, value: float) -> None:
        """记录特征值"""
        if feature_name not in self.feature_history:
            self.feature_history[feature_name] = []
        
        self.feature_history[feature_name].append(value)
        
        # 限制历史数据量
        max_history = self.config.get('feature_history_size', 5000)
        if len(self.feature_history[feature_name]) > max_history:
            self.feature_history[feature_name] = self.feature_history[feature_name][-max_history:]
    
    async def detect_feature_drift(self, feature_name: str) -> Dict[str, Any]:
        """检测单个特征的漂移"""
        if (feature_name not in self.feature_baselines or 
            feature_name not in self.feature_history or
            len(self.feature_history[feature_name]) < 100):
            return {'error': 'Insufficient data for drift detection'}
        
        baseline = self.feature_baselines[feature_name]
        recent_values = self.feature_history[feature_name][-1000:]  # 最近1000个值
        
        # 计算当前统计
        current_stats = {
            'mean': statistics.mean(recent_values),
            'std': statistics.stdev(recent_values) if len(recent_values) > 1 else 0,
            'min': min(recent_values),
            'max': max(recent_values),
            'median': statistics.median(recent_values)
        }
        
        # 使用Kolmogorov-Smirnov测试检测分布漂移
        drift_score = self._calculate_ks_statistic(
            baseline.get('sample_data', []),
            recent_values
        )
        
        # 计算均值漂移
        mean_drift = abs(current_stats['mean'] - baseline.get('mean', current_stats['mean']))
        relative_mean_drift = mean_drift / abs(baseline.get('mean', 1)) if baseline.get('mean', 0) != 0 else mean_drift
        
        # 计算方差漂移
        std_drift = abs(current_stats['std'] - baseline.get('std', current_stats['std']))
        relative_std_drift = std_drift / baseline.get('std', 1) if baseline.get('std', 0) != 0 else std_drift
        
        self.drift_scores[feature_name] = drift_score
        
        return {
            'feature_name': feature_name,
            'drift_score': drift_score,
            'drift_level': self._classify_feature_drift(drift_score),
            'mean_drift': relative_mean_drift,
            'std_drift': relative_std_drift,
            'current_stats': current_stats,
            'baseline_stats': baseline,
            'sample_size': len(recent_values)
        }
    
    def _calculate_ks_statistic(self, baseline_sample: List[float], current_sample: List[float]) -> float:
        """计算Kolmogorov-Smirnov统计量（简化版）"""
        if not baseline_sample or not current_sample:
            return 0.0
        
        try:
            # 简化的KS统计量计算
            baseline_sorted = sorted(baseline_sample)
            current_sorted = sorted(current_sample)
            
            # 计算经验分布函数的最大差异
            max_diff = 0.0
            for i, value in enumerate(baseline_sorted):
                # 在当前样本中找到相同值的位置
                current_cdf = sum(1 for x in current_sorted if x <= value) / len(current_sorted)
                baseline_cdf = (i + 1) / len(baseline_sorted)
                diff = abs(current_cdf - baseline_cdf)
                max_diff = max(max_diff, diff)
            
            return max_diff
        except Exception:
            return 0.0
    
    def _classify_feature_drift(self, drift_score: float) -> str:
        """分类特征漂移级别"""
        if drift_score >= 0.3:
            return 'critical'
        elif drift_score >= 0.2:
            return 'significant'
        elif drift_score >= 0.1:
            return 'moderate'
        elif drift_score >= 0.05:
            return 'minimal'
        else:
            return 'none'


class DataQualityMonitor:
    """数据质量监控器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.quality_checks: Dict[str, List[Dict[str, Any]]] = {}
        self.quality_scores: Dict[str, float] = {}
    
    async def check_data_quality(self, data_source: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查数据质量"""
        quality_report = {
            'data_source': data_source,
            'timestamp': datetime.now(timezone.utc),
            'checks': {},
            'overall_score': 0.0,
            'issues': []
        }
        
        # 完整性检查
        completeness_score = self._check_completeness(data)
        quality_report['checks']['completeness'] = completeness_score
        
        # 准确性检查
        accuracy_score = self._check_accuracy(data)
        quality_report['checks']['accuracy'] = accuracy_score
        
        # 一致性检查
        consistency_score = self._check_data_consistency(data)
        quality_report['checks']['consistency'] = consistency_score
        
        # 及时性检查
        timeliness_score = self._check_timeliness(data)
        quality_report['checks']['timeliness'] = timeliness_score
        
        # 计算整体质量分数
        quality_report['overall_score'] = statistics.mean([
            completeness_score, accuracy_score, consistency_score, timeliness_score
        ])
        
        # 记录质量检查结果
        if data_source not in self.quality_checks:
            self.quality_checks[data_source] = []
        self.quality_checks[data_source].append(quality_report)
        
        # 限制历史记录
        max_history = self.config.get('quality_history_size', 1000)
        if len(self.quality_checks[data_source]) > max_history:
            self.quality_checks[data_source] = self.quality_checks[data_source][-max_history:]
        
        return quality_report
    
    def _check_completeness(self, data: Dict[str, Any]) -> float:
        """检查数据完整性"""
        required_fields = self.config.get('required_fields', [])
        if not required_fields:
            return 1.0
        
        present_fields = sum(1 for field in required_fields if field in data and data[field] is not None)
        return present_fields / len(required_fields)
    
    def _check_accuracy(self, data: Dict[str, Any]) -> float:
        """检查数据准确性"""
        score = 1.0
        issues = []
        
        # 检查价格数据的合理性
        for field in ['price', 'open', 'high', 'low', 'close']:
            if field in data:
                value = data[field]
                if not isinstance(value, (int, float)) or value <= 0:
                    score -= 0.2
                    issues.append(f"Invalid {field}: {value}")
        
        # 检查成交量的合理性
        if 'volume' in data:
            volume = data['volume']
            if not isinstance(volume, (int, float)) or volume < 0:
                score -= 0.2
                issues.append(f"Invalid volume: {volume}")
        
        # 检查时间戳的合理性
        if 'timestamp' in data:
            try:
                timestamp = data['timestamp']
                if isinstance(timestamp, str):
                    datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                elif isinstance(timestamp, (int, float)):
                    datetime.fromtimestamp(timestamp, timezone.utc)
            except:
                score -= 0.3
                issues.append("Invalid timestamp format")
        
        return max(0.0, score)
    
    def _check_data_consistency(self, data: Dict[str, Any]) -> float:
        """检查数据一致性"""
        score = 1.0
        
        # 检查OHLC数据的逻辑一致性
        if all(field in data for field in ['open', 'high', 'low', 'close']):
            o, h, l, c = data['open'], data['high'], data['low'], data['close']
            if not (l <= o <= h and l <= c <= h):
                score -= 0.5
        
        return max(0.0, score)
    
    def _check_timeliness(self, data: Dict[str, Any]) -> float:
        """检查数据及时性"""
        if 'timestamp' not in data:
            return 1.0
        
        try:
            if isinstance(data['timestamp'], str):
                data_time = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            else:
                data_time = datetime.fromtimestamp(data['timestamp'], timezone.utc)
            
            # 检查数据是否太旧（超过配置的延迟阈值）
            max_delay_seconds = self.config.get('max_data_delay_seconds', 300)  # 5分钟
            delay = (datetime.now(timezone.utc) - data_time).total_seconds()
            
            if delay > max_delay_seconds:
                return max(0.0, 1.0 - (delay - max_delay_seconds) / max_delay_seconds)
            
            return 1.0
        except:
            return 0.0


class QualityAssuranceSystem:
    """综合质量保证系统"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.consistency_monitor = SignalConsistencyMonitor(config)
        self.drift_detector = BacktestLiveDriftDetector(config)
        self.feature_monitor = FeatureDriftMonitor(config)
        self.data_quality_monitor = DataQualityMonitor(config)
        self.quality_reports: List[QualityReport] = []
    
    async def generate_comprehensive_quality_report(self, time_window: str = "24h") -> QualityReport:
        """生成综合质量报告"""
        report_id = f"qa_report_{int(time.time())}"
        timestamp = datetime.now(timezone.utc)
        
        # 收集各子系统的质量分数
        quality_scores = {}
        violations = []
        recommendations = []
        detailed_metrics = {}
        
        # 1. 信号一致性分析
        try:
            consistency_report = self.consistency_monitor.get_consistency_report()
            quality_scores[QualityMetric.SIGNAL_POSITION_CONSISTENCY] = consistency_report['average_score']
            detailed_metrics['consistency'] = consistency_report
            
            if consistency_report['violation_rate'] > 0.1:  # 10%以上违规率
                violations.append({
                    'type': 'signal_consistency',
                    'severity': 'warning',
                    'message': f"Signal-Position consistency violation rate: {consistency_report['violation_rate']:.2%}"
                })
                recommendations.append("Review signal generation logic and position management")
        except Exception as e:
            logger.error(f"❌ Error in consistency analysis: {e}")
            quality_scores[QualityMetric.SIGNAL_POSITION_CONSISTENCY] = 0.0
        
        # 2. 回测漂移分析
        try:
            drift_report = await self.drift_detector.detect_drift()
            if 'error' not in drift_report:
                quality_scores[QualityMetric.BACKTEST_LIVE_DRIFT] = max(0, 1 - drift_report['overall_drift_score'])
                detailed_metrics['drift'] = drift_report
                
                if drift_report['drift_detected']:
                    violations.append({
                        'type': 'performance_drift',
                        'severity': 'critical' if drift_report['overall_drift_level'] == 'critical' else 'warning',
                        'message': f"Performance drift detected: {drift_report['overall_drift_level']} level"
                    })
                    recommendations.extend(drift_report['recommendations'])
        except Exception as e:
            logger.error(f"❌ Error in drift detection: {e}")
            quality_scores[QualityMetric.BACKTEST_LIVE_DRIFT] = 0.0
        
        # 3. 特征漂移分析
        feature_drift_scores = []
        for feature_name in self.feature_monitor.feature_history.keys():
            try:
                feature_drift = await self.feature_monitor.detect_feature_drift(feature_name)
                if 'error' not in feature_drift:
                    feature_score = max(0, 1 - feature_drift['drift_score'])
                    feature_drift_scores.append(feature_score)
                    
                    if feature_drift['drift_level'] in ['significant', 'critical']:
                        violations.append({
                            'type': 'feature_drift',
                            'severity': 'warning',
                            'message': f"Feature '{feature_name}' drift: {feature_drift['drift_level']}"
                        })
            except Exception as e:
                logger.error(f"❌ Error in feature drift detection for {feature_name}: {e}")
        
        if feature_drift_scores:
            quality_scores[QualityMetric.FEATURE_DRIFT] = statistics.mean(feature_drift_scores)
        else:
            quality_scores[QualityMetric.FEATURE_DRIFT] = 1.0
        
        # 计算整体一致性级别
        overall_score = statistics.mean(quality_scores.values()) if quality_scores else 0.0
        
        if overall_score >= 0.95:
            consistency_level = ConsistencyLevel.EXCELLENT
        elif overall_score >= 0.85:
            consistency_level = ConsistencyLevel.GOOD
        elif overall_score >= 0.70:
            consistency_level = ConsistencyLevel.ACCEPTABLE
        elif overall_score >= 0.50:
            consistency_level = ConsistencyLevel.POOR
        else:
            consistency_level = ConsistencyLevel.CRITICAL
        
        # 生成综合建议
        if consistency_level == ConsistencyLevel.CRITICAL:
            recommendations.insert(0, "CRITICAL: System quality critically degraded - immediate investigation required")
        elif consistency_level == ConsistencyLevel.POOR:
            recommendations.insert(0, "System quality below acceptable levels - reduce trading activity")
        
        # 创建质量报告
        report = QualityReport(
            report_id=report_id,
            timestamp=timestamp,
            time_window=time_window,
            quality_scores=quality_scores,
            consistency_level=consistency_level,
            violations=violations,
            recommendations=recommendations[:10],  # 限制建议数量
            detailed_metrics=detailed_metrics
        )
        
        self.quality_reports.append(report)
        
        # 限制报告历史
        max_reports = self.config.get('max_quality_reports', 100)
        if len(self.quality_reports) > max_reports:
            self.quality_reports = self.quality_reports[-max_reports:]
        
        logger.info(f"📊 Generated quality report: {consistency_level.value} ({overall_score:.3f})")
        return report
    
    def get_latest_quality_report(self) -> Optional[QualityReport]:
        """获取最新的质量报告"""
        return self.quality_reports[-1] if self.quality_reports else None
    
    def get_quality_trend(self, days: int = 7) -> Dict[str, Any]:
        """获取质量趋势分析"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        recent_reports = [
            report for report in self.quality_reports
            if report.timestamp >= cutoff_time
        ]
        
        if not recent_reports:
            return {'error': 'No recent reports available'}
        
        # 计算趋势
        trend_data = {}
        for metric in QualityMetric:
            scores = [
                report.quality_scores.get(metric, 0)
                for report in recent_reports
            ]
            if scores:
                trend_data[metric.value] = {
                    'current': scores[-1],
                    'average': statistics.mean(scores),
                    'trend': 'improving' if len(scores) > 1 and scores[-1] > scores[0] else 'degrading'
                }
        
        return {
            'period_days': days,
            'reports_count': len(recent_reports),
            'trends': trend_data,
            'overall_trend': 'improving' if recent_reports[-1].consistency_level.value > recent_reports[0].consistency_level.value else 'stable' if recent_reports[-1].consistency_level == recent_reports[0].consistency_level else 'degrading'
        }


# 工厂函数
def create_quality_assurance_system(config: Dict[str, Any] = None) -> QualityAssuranceSystem:
    """创建质量保证系统"""
    return QualityAssuranceSystem(config)


# 演示函数
async def quality_assurance_demo():
    """质量保证系统演示"""
    print("🚀 DipMaster Quality Assurance System Demo")
    
    # 创建质量保证系统
    qa_system = create_quality_assurance_system()
    
    # 设置回测基准
    qa_system.drift_detector.set_backtest_baseline({
        'win_rate': 0.82,
        'avg_return': 0.008,
        'sharpe_ratio': 1.85,
        'max_drawdown': 0.12,
        'avg_holding_time': 96.5
    })
    print("📊 Set backtest baseline")
    
    # 模拟信号和持仓数据
    signal = SignalRecord(
        signal_id="sig_demo_001",
        timestamp=datetime.now(timezone.utc),
        symbol="BTCUSDT",
        signal_type="BUY",
        confidence=0.87,
        price=43250.50,
        technical_indicators={'rsi': 34.2, 'ma20_distance': -0.008},
        expected_entry_price=43200.00,
        expected_holding_minutes=75
    )
    
    await qa_system.consistency_monitor.record_signal(signal)
    
    position = PositionRecord(
        position_id="pos_demo_001",
        signal_id="sig_demo_001",
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.15,
        entry_price=43225.00,
        entry_time=datetime.now(timezone.utc),
        holding_minutes=82
    )
    
    await qa_system.consistency_monitor.record_position(position)
    print("📊 Recorded signal and position")
    
    # 模拟实盘性能数据
    live_performance = [
        {'win_rate': 0.78, 'avg_return': 0.0075, 'sharpe_ratio': 1.72, 'max_drawdown': 0.14},
        {'win_rate': 0.75, 'avg_return': 0.0071, 'sharpe_ratio': 1.68, 'max_drawdown': 0.15},
        {'win_rate': 0.73, 'avg_return': 0.0068, 'sharpe_ratio': 1.63, 'max_drawdown': 0.16},
    ]
    
    for perf in live_performance:
        await qa_system.drift_detector.record_live_performance(perf)
    print("📊 Recorded live performance data")
    
    # 模拟特征数据
    qa_system.feature_monitor.set_feature_baseline('rsi', {
        'mean': 45.0,
        'std': 15.0,
        'min': 10.0,
        'max': 90.0
    })
    
    # 记录一些特征值
    rsi_values = [42.5, 38.2, 35.8, 33.1, 30.5, 28.9]  # 模拟RSI漂移
    for rsi in rsi_values:
        await qa_system.feature_monitor.record_feature_value('rsi', rsi)
    print("📊 Recorded feature data")
    
    # 生成综合质量报告
    quality_report = await qa_system.generate_comprehensive_quality_report()
    
    print(f"\n📋 Quality Assessment Report:")
    print(f"   Overall Level: {quality_report.consistency_level.value.upper()}")
    print(f"   Quality Scores:")
    for metric, score in quality_report.quality_scores.items():
        print(f"     {metric.value}: {score:.3f}")
    
    print(f"   Violations: {len(quality_report.violations)}")
    for violation in quality_report.violations:
        print(f"     [{violation['severity'].upper()}] {violation['message']}")
    
    print(f"   Recommendations: {len(quality_report.recommendations)}")
    for i, rec in enumerate(quality_report.recommendations[:3], 1):
        print(f"     {i}. {rec}")
    
    # 检查质量趋势
    trend_analysis = qa_system.get_quality_trend(days=1)
    if 'error' not in trend_analysis:
        print(f"\n📈 Quality Trend: {trend_analysis.get('overall_trend', 'unknown')}")
    
    print("✅ Demo completed successfully")


if __name__ == "__main__":
    asyncio.run(quality_assurance_demo())