#!/usr/bin/env python3
"""
Strategy Drift Detector for DipMaster Trading System
策略漂移检测器 - 监控策略性能衰减和模型漂移

Features:
- Backtest vs Live performance drift detection
- Statistical significance testing
- Feature distribution drift monitoring
- Model performance degradation alerts
- Regime change detection
- Adaptive threshold management
- Performance attribution analysis
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import numpy as np
from datetime import datetime, timezone, timedelta
import json

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Type of drift detected."""
    PERFORMANCE = "performance"
    FEATURE = "feature"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    REGIME = "regime"


class DriftSeverity(Enum):
    """Drift severity levels."""
    NONE = "none"          # <2% drift
    MINIMAL = "minimal"    # 2-5% drift
    MODERATE = "moderate"  # 5-10% drift
    SIGNIFICANT = "significant"  # 10-20% drift
    CRITICAL = "critical"  # >20% drift


@dataclass
class PerformanceWindow:
    """Performance data for a time window."""
    start_time: float
    end_time: float
    trades_count: int
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    volatility: float
    skewness: float = 0.0
    kurtosis: float = 0.0
    returns: List[float] = field(default_factory=list)


@dataclass
class FeatureStats:
    """Statistical properties of a feature."""
    name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    percentiles: Dict[str, float]  # 25th, 50th, 75th, 95th, 99th
    distribution_type: str = "unknown"
    last_updated: float = 0.0


@dataclass
class DriftAlert:
    """Drift detection alert."""
    alert_id: str
    timestamp: float
    drift_type: DriftType
    severity: DriftSeverity
    metric_name: str
    baseline_value: float
    current_value: float
    drift_percentage: float
    p_value: Optional[float]
    confidence_level: float
    description: str
    recommendation: str
    auto_action: bool = False


class StatisticalTest:
    """Statistical tests for drift detection."""
    
    @staticmethod
    def kolmogorov_smirnov_test(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test for distribution comparison.
        Returns (statistic, p_value).
        """
        try:
            if len(sample1) < 10 or len(sample2) < 10:
                return 0.0, 1.0
            
            # Simple implementation of KS test
            # In production, would use scipy.stats.ks_2samp
            n1, n2 = len(sample1), len(sample2)
            
            # Combine and sort all values
            all_values = sorted(set(sample1 + sample2))
            
            max_diff = 0.0
            for value in all_values:
                # Calculate empirical CDFs
                cdf1 = sum(1 for x in sample1 if x <= value) / n1
                cdf2 = sum(1 for x in sample2 if x <= value) / n2
                diff = abs(cdf1 - cdf2)
                max_diff = max(max_diff, diff)
            
            # Approximate p-value calculation
            critical_value = 1.36 * np.sqrt((n1 + n2) / (n1 * n2))
            p_value = 2 * np.exp(-2 * (max_diff / critical_value) ** 2) if critical_value > 0 else 1.0
            p_value = min(max(p_value, 0.0), 1.0)
            
            return max_diff, p_value
            
        except Exception as e:
            logger.error(f"❌ KS test error: {e}")
            return 0.0, 1.0
    
    @staticmethod
    def welch_t_test(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
        """
        Perform Welch's t-test for mean comparison.
        Returns (t_statistic, p_value).
        """
        try:
            if len(sample1) < 5 or len(sample2) < 5:
                return 0.0, 1.0
            
            mean1, mean2 = statistics.mean(sample1), statistics.mean(sample2)
            var1, var2 = statistics.variance(sample1), statistics.variance(sample2)
            n1, n2 = len(sample1), len(sample2)
            
            if var1 == 0 and var2 == 0:
                return 0.0, 1.0 if mean1 == mean2 else 0.0
            
            # Welch's t-statistic
            pooled_se = np.sqrt(var1/n1 + var2/n2)
            if pooled_se == 0:
                return 0.0, 1.0
            
            t_stat = (mean1 - mean2) / pooled_se
            
            # Degrees of freedom (Welch-Satterthwaite equation)
            if var1 == 0 or var2 == 0:
                return t_stat, 0.05  # Conservative p-value
            
            df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
            
            # Approximate p-value (two-tailed)
            # In production, would use scipy.stats.t.sf
            p_value = 2 * (1 - (1 / (1 + abs(t_stat)**2 / df)**0.5))
            p_value = min(max(p_value, 0.0), 1.0)
            
            return t_stat, p_value
            
        except Exception as e:
            logger.error(f"❌ T-test error: {e}")
            return 0.0, 1.0


class StrategyDriftDetector:
    """
    Advanced strategy drift detection system.
    
    Monitors strategy performance against baseline (backtest) and detects
    significant deviations that may indicate model degradation or regime changes.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 detection_window_hours: int = 24,
                 baseline_window_hours: int = 168):  # 1 week baseline
        """
        Initialize strategy drift detector.
        
        Args:
            config: Configuration parameters
            detection_window_hours: Window for drift detection
            baseline_window_hours: Window for baseline comparison
        """
        self.config = config or {}
        self.detection_window_hours = detection_window_hours
        self.baseline_window_hours = baseline_window_hours
        
        # Baseline data (from backtest or initial live period)
        self.baseline_performance = None
        self.baseline_features = {}
        self.baseline_correlations = {}
        
        # Live performance tracking
        self.performance_history = deque(maxlen=10000)
        self.feature_history = defaultdict(lambda: deque(maxlen=10000))
        
        # Performance windows for comparison
        self.performance_windows = deque(maxlen=100)
        
        # Feature statistics tracking
        self.feature_stats = {}
        self.feature_drift_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Alert management
        self.drift_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
        # Thresholds and configuration
        self.drift_thresholds = self._setup_thresholds()
        self.statistical_significance = 0.05  # 5% significance level
        
        # Regime detection
        self.regime_indicators = deque(maxlen=1000)
        self.current_regime = "normal"
        
        logger.info("📈 StrategyDriftDetector initialized")
    
    def _setup_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Setup drift detection thresholds."""
        return {
            'performance': {
                'minimal': 0.02,      # 2%
                'moderate': 0.05,     # 5%
                'significant': 0.10,  # 10%
                'critical': 0.20      # 20%
            },
            'feature': {
                'minimal': 0.05,      # 5%
                'moderate': 0.10,     # 10%
                'significant': 0.20,  # 20%
                'critical': 0.30      # 30%
            },
            'distribution': {
                'minimal': 0.1,       # KS statistic
                'moderate': 0.2,
                'significant': 0.3,
                'critical': 0.4
            }
        }
    
    def set_baseline(self, baseline_data: Dict[str, Any]):
        """Set baseline performance and feature data."""
        try:
            # Set performance baseline
            if 'performance' in baseline_data:
                perf_data = baseline_data['performance']
                self.baseline_performance = PerformanceWindow(
                    start_time=perf_data.get('start_time', 0),
                    end_time=perf_data.get('end_time', 0),
                    trades_count=perf_data.get('trades_count', 0),
                    win_rate=perf_data.get('win_rate', 0),
                    avg_return=perf_data.get('avg_return', 0),
                    sharpe_ratio=perf_data.get('sharpe_ratio', 0),
                    max_drawdown=perf_data.get('max_drawdown', 0),
                    profit_factor=perf_data.get('profit_factor', 0),
                    volatility=perf_data.get('volatility', 0),
                    returns=perf_data.get('returns', [])
                )
            
            # Set feature baselines
            if 'features' in baseline_data:
                for feature_name, stats in baseline_data['features'].items():
                    self.baseline_features[feature_name] = FeatureStats(
                        name=feature_name,
                        mean=stats.get('mean', 0),
                        std=stats.get('std', 0),
                        min_val=stats.get('min', 0),
                        max_val=stats.get('max', 0),
                        percentiles=stats.get('percentiles', {}),
                        distribution_type=stats.get('distribution_type', 'unknown')
                    )
            
            # Set correlation baselines
            if 'correlations' in baseline_data:
                self.baseline_correlations = baseline_data['correlations']
            
            logger.info(f"📊 Baseline set: {len(self.baseline_features)} features, "
                       f"performance from {self.baseline_performance.trades_count if self.baseline_performance else 0} trades")
            
        except Exception as e:
            logger.error(f"❌ Failed to set baseline: {e}")
    
    def record_trade_performance(self, trade_data: Dict[str, Any]):
        """Record individual trade performance."""
        try:
            trade_record = {
                'timestamp': trade_data.get('timestamp', time.time()),
                'symbol': trade_data.get('symbol'),
                'return': trade_data.get('return', 0),
                'duration': trade_data.get('duration', 0),
                'win': trade_data.get('return', 0) > 0,
                'entry_confidence': trade_data.get('entry_confidence', 0),
                'exit_reason': trade_data.get('exit_reason', 'unknown'),
                'slippage': trade_data.get('slippage', 0),
                'fees': trade_data.get('fees', 0)
            }
            
            self.performance_history.append(trade_record)
            
            # Update performance windows
            self._update_performance_windows()
            
            logger.debug(f"📈 Recorded trade: {trade_record['symbol']} return={trade_record['return']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record trade performance: {e}")
    
    def record_feature_values(self, feature_values: Dict[str, float]):
        """Record current feature values."""
        try:
            current_time = time.time()
            
            for feature_name, value in feature_values.items():
                self.feature_history[feature_name].append({
                    'timestamp': current_time,
                    'value': value
                })
            
            # Update feature statistics
            self._update_feature_statistics(feature_values)
            
            logger.debug(f"📊 Recorded {len(feature_values)} feature values")
            
        except Exception as e:
            logger.error(f"❌ Failed to record features: {e}")
    
    def _update_performance_windows(self):
        """Update rolling performance windows."""
        try:
            current_time = time.time()
            window_seconds = self.detection_window_hours * 3600
            
            # Get trades in the current window
            window_trades = [
                trade for trade in self.performance_history
                if current_time - trade['timestamp'] <= window_seconds
            ]
            
            if len(window_trades) < 5:  # Need minimum trades
                return
            
            # Calculate window performance metrics
            returns = [trade['return'] for trade in window_trades]
            wins = [trade for trade in window_trades if trade['win']]
            
            window_perf = PerformanceWindow(
                start_time=min(trade['timestamp'] for trade in window_trades),
                end_time=max(trade['timestamp'] for trade in window_trades),
                trades_count=len(window_trades),
                win_rate=len(wins) / len(window_trades),
                avg_return=statistics.mean(returns),
                sharpe_ratio=self._calculate_sharpe_ratio(returns),
                max_drawdown=self._calculate_max_drawdown(returns),
                profit_factor=self._calculate_profit_factor(returns),
                volatility=statistics.stdev(returns) if len(returns) > 1 else 0,
                returns=returns
            )
            
            self.performance_windows.append(window_perf)
            
        except Exception as e:
            logger.error(f"❌ Error updating performance windows: {e}")
    
    def _update_feature_statistics(self, feature_values: Dict[str, float]):
        """Update feature statistics."""
        try:
            current_time = time.time()
            
            for feature_name, value in feature_values.items():
                # Get recent values for this feature
                recent_values = [
                    entry['value'] for entry in self.feature_history[feature_name]
                    if current_time - entry['timestamp'] <= self.detection_window_hours * 3600
                ]
                
                if len(recent_values) < 10:  # Need minimum data
                    continue
                
                # Calculate statistics
                percentiles = {
                    '25': np.percentile(recent_values, 25),
                    '50': np.percentile(recent_values, 50),
                    '75': np.percentile(recent_values, 75),
                    '95': np.percentile(recent_values, 95),
                    '99': np.percentile(recent_values, 99)
                }
                
                self.feature_stats[feature_name] = FeatureStats(
                    name=feature_name,
                    mean=statistics.mean(recent_values),
                    std=statistics.stdev(recent_values) if len(recent_values) > 1 else 0,
                    min_val=min(recent_values),
                    max_val=max(recent_values),
                    percentiles=percentiles,
                    last_updated=current_time
                )
            
        except Exception as e:
            logger.error(f"❌ Error updating feature statistics: {e}")
    
    def detect_performance_drift(self) -> List[DriftAlert]:
        """Detect performance drift from baseline."""
        alerts = []
        
        try:
            if not self.baseline_performance or not self.performance_windows:
                return alerts
            
            current_window = self.performance_windows[-1]
            baseline = self.baseline_performance
            
            # Compare key performance metrics
            performance_comparisons = [
                ('win_rate', current_window.win_rate, baseline.win_rate),
                ('avg_return', current_window.avg_return, baseline.avg_return),
                ('sharpe_ratio', current_window.sharpe_ratio, baseline.sharpe_ratio),
                ('volatility', current_window.volatility, baseline.volatility),
                ('max_drawdown', current_window.max_drawdown, baseline.max_drawdown)
            ]
            
            for metric_name, current_value, baseline_value in performance_comparisons:
                if baseline_value == 0:
                    continue
                
                # Calculate drift percentage
                drift_pct = abs(current_value - baseline_value) / abs(baseline_value)
                
                # Determine severity
                severity = self._classify_drift_severity(drift_pct, 'performance')
                
                if severity != DriftSeverity.NONE:
                    # Perform statistical test
                    if metric_name in ['win_rate', 'avg_return'] and len(current_window.returns) > 10:
                        if len(baseline.returns) > 10:
                            _, p_value = StatisticalTest.welch_t_test(
                                current_window.returns, baseline.returns
                            )
                        else:
                            p_value = None
                    else:
                        p_value = None
                    
                    # Create alert
                    alert = DriftAlert(
                        alert_id=f"perf_drift_{metric_name}_{int(time.time())}",
                        timestamp=time.time(),
                        drift_type=DriftType.PERFORMANCE,
                        severity=severity,
                        metric_name=metric_name,
                        baseline_value=baseline_value,
                        current_value=current_value,
                        drift_percentage=drift_pct * 100,
                        p_value=p_value,
                        confidence_level=95.0,
                        description=f"Performance drift detected in {metric_name}: "
                                  f"{current_value:.4f} vs baseline {baseline_value:.4f} "
                                  f"({drift_pct:.1%} change)",
                        recommendation=self._get_drift_recommendation(metric_name, severity),
                        auto_action=severity in [DriftSeverity.SIGNIFICANT, DriftSeverity.CRITICAL]
                    )
                    
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Error detecting performance drift: {e}")
            return []
    
    def detect_feature_drift(self) -> List[DriftAlert]:
        """Detect feature distribution drift."""
        alerts = []
        
        try:
            for feature_name, baseline_stats in self.baseline_features.items():
                if feature_name not in self.feature_stats:
                    continue
                
                current_stats = self.feature_stats[feature_name]
                
                # Compare mean values
                if baseline_stats.std > 0:
                    mean_drift = abs(current_stats.mean - baseline_stats.mean) / baseline_stats.std
                    
                    if mean_drift > 2.0:  # 2 standard deviations
                        severity = self._classify_drift_severity(mean_drift / 2.0, 'feature')
                        
                        # Get recent values for statistical test
                        recent_values = [
                            entry['value'] for entry in self.feature_history[feature_name]
                            if time.time() - entry['timestamp'] <= self.detection_window_hours * 3600
                        ]
                        
                        # Simulate baseline values (in production, would have actual baseline data)
                        baseline_values = np.random.normal(
                            baseline_stats.mean, baseline_stats.std, len(recent_values)
                        ).tolist()
                        
                        # Perform KS test
                        ks_stat, p_value = StatisticalTest.kolmogorov_smirnov_test(
                            recent_values, baseline_values
                        )
                        
                        alert = DriftAlert(
                            alert_id=f"feature_drift_{feature_name}_{int(time.time())}",
                            timestamp=time.time(),
                            drift_type=DriftType.FEATURE,
                            severity=severity,
                            metric_name=feature_name,
                            baseline_value=baseline_stats.mean,
                            current_value=current_stats.mean,
                            drift_percentage=mean_drift * 100,
                            p_value=p_value,
                            confidence_level=95.0,
                            description=f"Feature drift detected in {feature_name}: "
                                      f"mean shifted from {baseline_stats.mean:.4f} to {current_stats.mean:.4f}",
                            recommendation=f"Review {feature_name} calculation and data sources",
                            auto_action=False
                        )
                        
                        alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Error detecting feature drift: {e}")
            return []
    
    def detect_regime_change(self) -> Optional[DriftAlert]:
        """Detect market regime changes."""
        try:
            if len(self.performance_windows) < 10:
                return None
            
            # Calculate regime indicators
            recent_windows = list(self.performance_windows)[-10:]
            
            # Volatility regime indicator
            recent_volatility = [w.volatility for w in recent_windows]
            avg_recent_vol = statistics.mean(recent_volatility)
            
            # Win rate stability
            recent_win_rates = [w.win_rate for w in recent_windows]
            win_rate_std = statistics.stdev(recent_win_rates) if len(recent_win_rates) > 1 else 0
            
            # Sharpe ratio trend
            recent_sharpe = [w.sharpe_ratio for w in recent_windows]
            
            # Simple regime detection logic
            regime_score = 0
            
            # High volatility regime
            if self.baseline_performance and avg_recent_vol > self.baseline_performance.volatility * 1.5:
                regime_score += 1
            
            # Unstable win rate
            if win_rate_std > 0.1:  # 10% standard deviation
                regime_score += 1
            
            # Declining Sharpe ratio
            if len(recent_sharpe) >= 5:
                recent_trend = recent_sharpe[-3:]
                early_trend = recent_sharpe[:3]
                if statistics.mean(recent_trend) < statistics.mean(early_trend) * 0.8:
                    regime_score += 1
            
            # Determine if regime change occurred
            if regime_score >= 2:
                new_regime = "high_volatility" if avg_recent_vol > 0.02 else "low_performance"
                
                if new_regime != self.current_regime:
                    self.current_regime = new_regime
                    
                    alert = DriftAlert(
                        alert_id=f"regime_change_{int(time.time())}",
                        timestamp=time.time(),
                        drift_type=DriftType.REGIME,
                        severity=DriftSeverity.SIGNIFICANT,
                        metric_name="market_regime",
                        baseline_value=0,
                        current_value=regime_score,
                        drift_percentage=0,
                        p_value=None,
                        confidence_level=80.0,
                        description=f"Market regime change detected: {self.current_regime}",
                        recommendation="Consider adjusting strategy parameters or reducing position sizes",
                        auto_action=True
                    )
                    
                    return alert
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error detecting regime change: {e}")
            return None
    
    def run_drift_detection(self) -> List[DriftAlert]:
        """Run comprehensive drift detection."""
        all_alerts = []
        
        try:
            # Performance drift detection
            perf_alerts = self.detect_performance_drift()
            all_alerts.extend(perf_alerts)
            
            # Feature drift detection
            feature_alerts = self.detect_feature_drift()
            all_alerts.extend(feature_alerts)
            
            # Regime change detection
            regime_alert = self.detect_regime_change()
            if regime_alert:
                all_alerts.append(regime_alert)
            
            # Store alerts
            for alert in all_alerts:
                self.drift_alerts[alert.alert_id] = alert
                self.alert_history.append(alert)
            
            if all_alerts:
                logger.warning(f"⚠️ Detected {len(all_alerts)} drift alerts")
                for alert in all_alerts:
                    if alert.severity in [DriftSeverity.SIGNIFICANT, DriftSeverity.CRITICAL]:
                        logger.critical(f"🚨 {alert.severity.value.upper()} drift: {alert.description}")
            
            return all_alerts
            
        except Exception as e:
            logger.error(f"❌ Failed to run drift detection: {e}")
            return []
    
    def _classify_drift_severity(self, drift_value: float, drift_category: str) -> DriftSeverity:
        """Classify drift severity based on value and category."""
        thresholds = self.drift_thresholds.get(drift_category, self.drift_thresholds['performance'])
        
        if drift_value >= thresholds['critical']:
            return DriftSeverity.CRITICAL
        elif drift_value >= thresholds['significant']:
            return DriftSeverity.SIGNIFICANT
        elif drift_value >= thresholds['moderate']:
            return DriftSeverity.MODERATE
        elif drift_value >= thresholds['minimal']:
            return DriftSeverity.MINIMAL
        else:
            return DriftSeverity.NONE
    
    def _get_drift_recommendation(self, metric_name: str, severity: DriftSeverity) -> str:
        """Get recommendation based on drift type and severity."""
        recommendations = {
            'win_rate': {
                DriftSeverity.MINIMAL: "Monitor win rate closely",
                DriftSeverity.MODERATE: "Review signal quality and market conditions",
                DriftSeverity.SIGNIFICANT: "Consider reducing position sizes",
                DriftSeverity.CRITICAL: "Halt trading and investigate strategy"
            },
            'sharpe_ratio': {
                DriftSeverity.MINIMAL: "Monitor risk-adjusted returns",
                DriftSeverity.MODERATE: "Review risk management parameters",
                DriftSeverity.SIGNIFICANT: "Reduce leverage and position sizes",
                DriftSeverity.CRITICAL: "Stop trading and recalibrate strategy"
            },
            'volatility': {
                DriftSeverity.MINIMAL: "Monitor market volatility",
                DriftSeverity.MODERATE: "Adjust position sizing for volatility",
                DriftSeverity.SIGNIFICANT: "Implement volatility filters",
                DriftSeverity.CRITICAL: "Reduce exposure significantly"
            }
        }
        
        return recommendations.get(metric_name, {}).get(
            severity, "Review strategy parameters and market conditions"
        )
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        try:
            if len(returns) < 2:
                return 0.0
            
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            
            if std_return == 0:
                return 0.0
            
            # Assuming risk-free rate of 0 for simplicity
            return mean_return / std_return * np.sqrt(252)  # Annualized
            
        except Exception:
            return 0.0
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown."""
        try:
            if not returns:
                return 0.0
            
            cumulative = [sum(returns[:i+1]) for i in range(len(returns))]
            running_max = [max(cumulative[:i+1]) for i in range(len(cumulative))]
            drawdowns = [running_max[i] - cumulative[i] for i in range(len(cumulative))]
            
            return max(drawdowns) if drawdowns else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor."""
        try:
            profits = sum(r for r in returns if r > 0)
            losses = abs(sum(r for r in returns if r < 0))
            
            if losses == 0:
                return float('inf') if profits > 0 else 1.0
            
            return profits / losses if losses > 0 else 1.0
            
        except Exception:
            return 1.0
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get comprehensive drift detection summary."""
        try:
            active_alerts = [alert for alert in self.drift_alerts.values() 
                           if time.time() - alert.timestamp < 24 * 3600]  # Last 24 hours
            
            # Count alerts by severity
            severity_counts = defaultdict(int)
            for alert in active_alerts:
                severity_counts[alert.severity.value] += 1
            
            # Count alerts by type
            type_counts = defaultdict(int)
            for alert in active_alerts:
                type_counts[alert.drift_type.value] += 1
            
            # Current performance vs baseline
            current_performance = {}
            if self.performance_windows and self.baseline_performance:
                current_window = self.performance_windows[-1]
                current_performance = {
                    'win_rate': {
                        'current': current_window.win_rate,
                        'baseline': self.baseline_performance.win_rate,
                        'drift_pct': ((current_window.win_rate - self.baseline_performance.win_rate) / 
                                     self.baseline_performance.win_rate * 100) if self.baseline_performance.win_rate != 0 else 0
                    },
                    'sharpe_ratio': {
                        'current': current_window.sharpe_ratio,
                        'baseline': self.baseline_performance.sharpe_ratio,
                        'drift_pct': ((current_window.sharpe_ratio - self.baseline_performance.sharpe_ratio) / 
                                     self.baseline_performance.sharpe_ratio * 100) if self.baseline_performance.sharpe_ratio != 0 else 0
                    }
                }
            
            return {
                'timestamp': time.time(),
                'current_regime': self.current_regime,
                'active_alerts': {
                    'total': len(active_alerts),
                    'by_severity': dict(severity_counts),
                    'by_type': dict(type_counts)
                },
                'performance_comparison': current_performance,
                'feature_status': {
                    'total_features': len(self.feature_stats),
                    'baseline_features': len(self.baseline_features),
                    'last_updated': max([fs.last_updated for fs in self.feature_stats.values()]) 
                                  if self.feature_stats else 0
                },
                'data_status': {
                    'performance_windows': len(self.performance_windows),
                    'trade_history': len(self.performance_history),
                    'alert_history': len(self.alert_history)
                },
                'detection_config': {
                    'detection_window_hours': self.detection_window_hours,
                    'baseline_window_hours': self.baseline_window_hours,
                    'significance_level': self.statistical_significance
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get drift summary: {e}")
            return {'error': str(e)}
    
    def export_drift_report(self) -> Dict[str, Any]:
        """Export comprehensive drift analysis report."""
        try:
            return {
                'timestamp': time.time(),
                'report_type': 'strategy_drift_analysis',
                'summary': self.get_drift_summary(),
                'baseline_data': {
                    'performance': self.baseline_performance.__dict__ if self.baseline_performance else None,
                    'features': {name: stats.__dict__ for name, stats in self.baseline_features.items()},
                    'correlations': self.baseline_correlations
                },
                'current_data': {
                    'performance_windows': [w.__dict__ for w in list(self.performance_windows)[-5:]],  # Last 5 windows
                    'feature_stats': {name: stats.__dict__ for name, stats in self.feature_stats.items()}
                },
                'active_alerts': [alert.__dict__ for alert in self.drift_alerts.values() 
                                if time.time() - alert.timestamp < 24 * 3600],
                'statistical_tests': {
                    'significance_level': self.statistical_significance,
                    'methods_used': ['kolmogorov_smirnov', 'welch_t_test']
                },
                'recommendations': self._generate_drift_recommendations()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to export drift report: {e}")
            return {'error': str(e)}
    
    def _generate_drift_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on drift analysis."""
        recommendations = []
        
        try:
            # Check for critical alerts
            critical_alerts = [alert for alert in self.drift_alerts.values() 
                             if alert.severity == DriftSeverity.CRITICAL and 
                             time.time() - alert.timestamp < 24 * 3600]
            
            if critical_alerts:
                recommendations.append("CRITICAL: Consider halting trading until drift issues are resolved")
            
            # Check regime changes
            if self.current_regime != "normal":
                recommendations.append(f"Market regime change detected ({self.current_regime}): adjust strategy parameters")
            
            # Performance-specific recommendations
            if self.performance_windows:
                recent_performance = self.performance_windows[-1]
                if recent_performance.win_rate < 0.5:
                    recommendations.append("Win rate below 50%: review signal quality and entry criteria")
                if recent_performance.sharpe_ratio < 1.0:
                    recommendations.append("Low Sharpe ratio: consider improving risk management")
            
            # Feature-specific recommendations
            feature_alerts = [alert for alert in self.drift_alerts.values() 
                            if alert.drift_type == DriftType.FEATURE and 
                            time.time() - alert.timestamp < 24 * 3600]
            
            if len(feature_alerts) > len(self.baseline_features) * 0.2:  # >20% of features drifting
                recommendations.append("High feature drift detected: review data pipeline and feature engineering")
            
            if not recommendations:
                recommendations.append("No significant drift detected: strategy performing within expected parameters")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            return ["Error generating recommendations"]


# Factory function
def create_strategy_drift_detector(config: Dict[str, Any]) -> StrategyDriftDetector:
    """Create and configure strategy drift detector."""
    return StrategyDriftDetector(
        config=config.get('drift_config', {}),
        detection_window_hours=config.get('detection_window_hours', 24),
        baseline_window_hours=config.get('baseline_window_hours', 168)
    )