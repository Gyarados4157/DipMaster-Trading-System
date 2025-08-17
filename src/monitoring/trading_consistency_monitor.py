#!/usr/bin/env python3
"""
Trading Consistency Monitor for DipMaster Trading System
交易一致性监控器 - 专业级信号-持仓-执行一致性检查

Features:
- Signal-Position-Execution consistency validation
- Backtest vs Production drift detection with statistical tests
- 15-minute boundary compliance monitoring
- Strategy parameter execution verification
- Real-time anomaly detection and alerting
- Comprehensive trade lifecycle tracking
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import numpy as np
from scipy import stats
import pandas as pd

from .kafka_event_producer import KafkaEventProducer, AlertEvent, EventType

logger = logging.getLogger(__name__)


class ConsistencyLevel(Enum):
    """Consistency check levels."""
    PERFECT = "perfect"      # 100% match
    ACCEPTABLE = "acceptable"  # Within tolerance
    WARNING = "warning"      # Outside tolerance but not critical
    CRITICAL = "critical"    # Severe inconsistency requiring action


class DriftType(Enum):
    """Drift detection types."""
    PERFORMANCE = "performance"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    EXECUTION_QUALITY = "execution_quality"


@dataclass
class SignalData:
    """Trading signal data structure."""
    signal_id: str
    timestamp: datetime
    symbol: str
    signal_type: str  # "BUY", "SELL", "HOLD"
    confidence: float
    price: float
    rsi: float
    ma20_distance: float
    volume_ratio: float
    expected_entry_price: float
    expected_holding_minutes: int
    strategy_params: Dict[str, Any]


@dataclass
class PositionData:
    """Position data structure."""
    position_id: str
    signal_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    holding_minutes: Optional[int] = None
    pnl: Optional[float] = None
    realized: bool = False


@dataclass
class ExecutionData:
    """Execution data structure."""
    execution_id: str
    position_id: str
    order_type: str
    symbol: str
    side: str
    quantity: float
    requested_price: float
    executed_price: float
    execution_time: datetime
    latency_ms: float
    slippage_bps: float
    fees: float
    venue: str


@dataclass
class ConsistencyReport:
    """Consistency check result."""
    check_type: str
    level: ConsistencyLevel
    score: float  # 0-100
    details: Dict[str, Any]
    violations: List[str]
    recommendations: List[str]
    timestamp: datetime


class TradingConsistencyMonitor:
    """
    Advanced trading consistency monitor for DipMaster Trading System.
    
    Provides comprehensive monitoring of signal-position-execution consistency,
    backtest vs production drift detection, and strategy compliance validation.
    """
    
    def __init__(self,
                 kafka_producer: Optional[KafkaEventProducer] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize trading consistency monitor.
        
        Args:
            kafka_producer: Kafka event producer for publishing alerts
            config: Configuration parameters
        """
        self.kafka_producer = kafka_producer
        self.config = config or {}
        
        # Data storage
        self.signals: Dict[str, SignalData] = {}
        self.positions: Dict[str, PositionData] = {}
        self.executions: Dict[str, ExecutionData] = {}
        
        # Historical data for drift detection
        self.backtest_performance = {}
        self.production_performance = {}
        self.performance_history = []
        
        # Monitoring thresholds
        self.consistency_thresholds = {
            'signal_position_match': 95.0,  # %
            'position_execution_match': 98.0,  # %
            'price_deviation_bps': 20.0,  # basis points
            'timing_deviation_minutes': 2.0,  # minutes
            'boundary_compliance': 100.0,  # %
            'drift_warning_threshold': 5.0,  # %
            'drift_critical_threshold': 10.0  # %
        }
        self.consistency_thresholds.update(self.config.get('thresholds', {}))
        
        # DipMaster specific parameters
        self.dipmaster_params = {
            'rsi_range': [30, 50],
            'max_holding_minutes': 180,
            'boundary_minutes': [15, 30, 45, 60],
            'target_profit_pct': 0.8,
            'dip_threshold_pct': 0.2,
            'volume_multiplier': 1.5
        }
        self.dipmaster_params.update(self.config.get('dipmaster_params', {}))
        
        # Statistics tracking
        self.stats = {
            'signals_processed': 0,
            'positions_tracked': 0,
            'executions_monitored': 0,
            'consistency_checks': 0,
            'violations_detected': 0,
            'alerts_generated': 0
        }
        
        logger.info("🔍 TradingConsistencyMonitor initialized")
    
    async def record_signal(self, signal_data: SignalData):
        """Record a trading signal for consistency tracking."""
        try:
            self.signals[signal_data.signal_id] = signal_data
            self.stats['signals_processed'] += 1
            
            # Validate signal parameters
            await self._validate_signal_parameters(signal_data)
            
            logger.debug(f"📊 Recorded signal {signal_data.signal_id} for {signal_data.symbol}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record signal: {e}")
    
    async def record_position(self, position_data: PositionData):
        """Record a position for consistency tracking."""
        try:
            self.positions[position_data.position_id] = position_data
            self.stats['positions_tracked'] += 1
            
            # Check signal-position consistency
            await self._check_signal_position_consistency(position_data)
            
            # Check 15-minute boundary compliance if position is closed
            if position_data.realized:
                await self._check_boundary_compliance(position_data)
            
            logger.debug(f"📊 Recorded position {position_data.position_id} for {position_data.symbol}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record position: {e}")
    
    async def record_execution(self, execution_data: ExecutionData):
        """Record an execution for consistency tracking."""
        try:
            self.executions[execution_data.execution_id] = execution_data
            self.stats['executions_monitored'] += 1
            
            # Check position-execution consistency
            await self._check_position_execution_consistency(execution_data)
            
            # Update production performance metrics
            await self._update_production_metrics(execution_data)
            
            logger.debug(f"📊 Recorded execution {execution_data.execution_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record execution: {e}")
    
    async def _validate_signal_parameters(self, signal: SignalData):
        """Validate signal parameters against DipMaster strategy rules."""
        violations = []
        
        # RSI range check
        rsi_min, rsi_max = self.dipmaster_params['rsi_range']
        if not (rsi_min <= signal.rsi <= rsi_max):
            violations.append(f"RSI {signal.rsi} outside range [{rsi_min}, {rsi_max}]")
        
        # Dip confirmation check
        if signal.signal_type == "BUY":
            # Check if price is below MA20 (87% probability condition)
            if signal.ma20_distance >= 0:
                violations.append(f"Buy signal when price above MA20 (distance: {signal.ma20_distance:.3f})")
            
            # Check volume confirmation
            min_volume = self.dipmaster_params['volume_multiplier']
            if signal.volume_ratio < min_volume:
                violations.append(f"Volume ratio {signal.volume_ratio:.2f} below threshold {min_volume}")
        
        # Expected holding time check
        max_holding = self.dipmaster_params['max_holding_minutes']
        if signal.expected_holding_minutes > max_holding:
            violations.append(f"Expected holding {signal.expected_holding_minutes}min exceeds max {max_holding}min")
        
        if violations:
            await self._generate_consistency_alert(
                alert_type="SIGNAL_PARAMETER_VIOLATION",
                severity="WARNING",
                message=f"Signal parameter violations detected for {signal.signal_id}",
                details={
                    'signal_id': signal.signal_id,
                    'symbol': signal.symbol,
                    'violations': violations,
                    'signal_data': asdict(signal)
                }
            )
    
    async def _check_signal_position_consistency(self, position: PositionData):
        """Check consistency between signal and resulting position."""
        signal = self.signals.get(position.signal_id)
        if not signal:
            await self._generate_consistency_alert(
                alert_type="ORPHANED_POSITION",
                severity="WARNING",
                message=f"Position {position.position_id} has no corresponding signal",
                details={'position_id': position.position_id, 'signal_id': position.signal_id}
            )
            return
        
        violations = []
        score = 100.0
        
        # Symbol consistency
        if signal.symbol != position.symbol:
            violations.append(f"Symbol mismatch: signal={signal.symbol}, position={position.symbol}")
            score -= 20
        
        # Direction consistency
        if signal.signal_type != position.side:
            violations.append(f"Direction mismatch: signal={signal.signal_type}, position={position.side}")
            score -= 30
        
        # Price deviation check
        if position.entry_price:
            price_diff_bps = abs(signal.expected_entry_price - position.entry_price) / signal.expected_entry_price * 10000
            max_deviation = self.consistency_thresholds['price_deviation_bps']
            
            if price_diff_bps > max_deviation:
                violations.append(f"Price deviation {price_diff_bps:.1f}bps exceeds threshold {max_deviation}bps")
                score -= min(20, price_diff_bps - max_deviation)
        
        # Timing consistency
        if signal.timestamp and position.entry_time:
            time_diff_minutes = abs((position.entry_time - signal.timestamp).total_seconds() / 60)
            max_timing_diff = self.consistency_thresholds['timing_deviation_minutes']
            
            if time_diff_minutes > max_timing_diff:
                violations.append(f"Timing deviation {time_diff_minutes:.1f}min exceeds threshold {max_timing_diff}min")
                score -= min(15, time_diff_minutes - max_timing_diff)
        
        # Determine consistency level
        level = self._determine_consistency_level(score)
        
        # Generate alert if necessary
        if level in [ConsistencyLevel.WARNING, ConsistencyLevel.CRITICAL]:
            await self._generate_consistency_alert(
                alert_type="SIGNAL_POSITION_INCONSISTENCY",
                severity="WARNING" if level == ConsistencyLevel.WARNING else "CRITICAL",
                message=f"Signal-position consistency issues detected",
                details={
                    'signal_id': signal.signal_id,
                    'position_id': position.position_id,
                    'consistency_score': score,
                    'violations': violations
                }
            )
        
        self.stats['consistency_checks'] += 1
        if violations:
            self.stats['violations_detected'] += len(violations)
    
    async def _check_position_execution_consistency(self, execution: ExecutionData):
        """Check consistency between position and execution."""
        # Find corresponding position
        position = None
        for pos in self.positions.values():
            if pos.position_id == execution.position_id:
                position = pos
                break
        
        if not position:
            await self._generate_consistency_alert(
                alert_type="ORPHANED_EXECUTION",
                severity="WARNING",
                message=f"Execution {execution.execution_id} has no corresponding position",
                details={'execution_id': execution.execution_id, 'position_id': execution.position_id}
            )
            return
        
        violations = []
        score = 100.0
        
        # Symbol consistency
        if position.symbol != execution.symbol:
            violations.append(f"Symbol mismatch: position={position.symbol}, execution={execution.symbol}")
            score -= 20
        
        # Side consistency
        if position.side != execution.side:
            violations.append(f"Side mismatch: position={position.side}, execution={execution.side}")
            score -= 30
        
        # Quantity consistency
        if abs(position.quantity - execution.quantity) > 0.0001:
            violations.append(f"Quantity mismatch: position={position.quantity}, execution={execution.quantity}")
            score -= 10
        
        # Price slippage check
        if execution.slippage_bps > 50:  # 50 bps threshold
            violations.append(f"High slippage: {execution.slippage_bps:.1f}bps")
            score -= min(20, execution.slippage_bps - 50)
        
        # Execution latency check
        if execution.latency_ms > 1000:  # 1 second threshold
            violations.append(f"High latency: {execution.latency_ms:.1f}ms")
            score -= min(15, (execution.latency_ms - 1000) / 100)
        
        # Determine consistency level
        level = self._determine_consistency_level(score)
        
        # Generate alert if necessary
        if level in [ConsistencyLevel.WARNING, ConsistencyLevel.CRITICAL]:
            await self._generate_consistency_alert(
                alert_type="POSITION_EXECUTION_INCONSISTENCY",
                severity="WARNING" if level == ConsistencyLevel.WARNING else "CRITICAL",
                message=f"Position-execution consistency issues detected",
                details={
                    'position_id': position.position_id,
                    'execution_id': execution.execution_id,
                    'consistency_score': score,
                    'violations': violations
                }
            )
        
        self.stats['consistency_checks'] += 1
        if violations:
            self.stats['violations_detected'] += len(violations)
    
    async def _check_boundary_compliance(self, position: PositionData):
        """Check 15-minute boundary compliance for DipMaster strategy."""
        if not position.exit_time or not position.entry_time:
            return
        
        # Calculate holding time
        holding_time = position.exit_time - position.entry_time
        holding_minutes = holding_time.total_seconds() / 60
        
        # Get exit minute within the hour
        exit_minute = position.exit_time.minute
        
        # DipMaster boundary rules: exit at 15, 30, 45, 60 minute marks
        boundary_minutes = self.dipmaster_params['boundary_minutes']
        
        # Check if exit was at boundary
        is_boundary_exit = any(abs(exit_minute - boundary) <= 1 for boundary in boundary_minutes)
        
        violations = []
        if not is_boundary_exit:
            violations.append(f"Exit at minute {exit_minute} not at 15-minute boundary")
        
        # Check maximum holding time
        max_holding = self.dipmaster_params['max_holding_minutes']
        if holding_minutes > max_holding:
            violations.append(f"Holding time {holding_minutes:.1f}min exceeds maximum {max_holding}min")
        
        # Generate alert if violations found
        if violations:
            await self._generate_consistency_alert(
                alert_type="BOUNDARY_COMPLIANCE_VIOLATION",
                severity="WARNING",
                message=f"15-minute boundary compliance violations for position {position.position_id}",
                details={
                    'position_id': position.position_id,
                    'symbol': position.symbol,
                    'exit_minute': exit_minute,
                    'holding_minutes': holding_minutes,
                    'violations': violations
                }
            )
    
    async def detect_backtest_production_drift(self,
                                             backtest_metrics: Dict[str, float],
                                             production_metrics: Dict[str, float],
                                             time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Detect drift between backtest and production performance.
        
        Uses statistical tests to identify significant performance divergence.
        """
        try:
            drift_results = {
                'timestamp': datetime.now(timezone.utc),
                'time_window_hours': time_window_hours,
                'drift_detected': False,
                'drift_types': [],
                'statistical_tests': {},
                'recommendations': []
            }
            
            # Store current metrics
            self.backtest_performance = backtest_metrics
            self.production_performance = production_metrics
            
            # Performance drift detection
            performance_drift = await self._detect_performance_drift(backtest_metrics, production_metrics)
            if performance_drift['drift_detected']:
                drift_results['drift_detected'] = True
                drift_results['drift_types'].append(DriftType.PERFORMANCE.value)
                drift_results['statistical_tests']['performance'] = performance_drift
            
            # Distribution drift detection (if historical data available)
            if len(self.performance_history) > 30:  # Need sufficient history
                distribution_drift = await self._detect_distribution_drift()
                if distribution_drift['drift_detected']:
                    drift_results['drift_detected'] = True
                    drift_results['drift_types'].append(DriftType.DISTRIBUTION.value)
                    drift_results['statistical_tests']['distribution'] = distribution_drift
            
            # Add current metrics to history
            self.performance_history.append({
                'timestamp': time.time(),
                'backtest': backtest_metrics,
                'production': production_metrics
            })
            
            # Limit history size
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            # Generate recommendations based on drift
            if drift_results['drift_detected']:
                drift_results['recommendations'] = await self._generate_drift_recommendations(drift_results)
                
                # Generate alert
                severity = "CRITICAL" if any("critical" in drift_type for drift_type in drift_results['drift_types']) else "WARNING"
                await self._generate_consistency_alert(
                    alert_type="BACKTEST_PRODUCTION_DRIFT",
                    severity=severity,
                    message=f"Drift detected between backtest and production performance",
                    details=drift_results
                )
            
            return drift_results
            
        except Exception as e:
            logger.error(f"❌ Failed to detect drift: {e}")
            return {'error': str(e), 'timestamp': datetime.now(timezone.utc)}
    
    async def _detect_performance_drift(self,
                                      backtest_metrics: Dict[str, float],
                                      production_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect performance metric drift using statistical comparison."""
        
        key_metrics = ['win_rate', 'sharpe_ratio', 'profit_factor', 'max_drawdown', 'avg_trade_pnl']
        drift_detected = False
        metric_drifts = {}
        
        warning_threshold = self.consistency_thresholds['drift_warning_threshold']
        critical_threshold = self.consistency_thresholds['drift_critical_threshold']
        
        for metric in key_metrics:
            if metric in backtest_metrics and metric in production_metrics:
                backtest_val = backtest_metrics[metric]
                production_val = production_metrics[metric]
                
                # Calculate percentage difference
                if backtest_val != 0:
                    pct_diff = abs(production_val - backtest_val) / abs(backtest_val) * 100
                else:
                    pct_diff = abs(production_val) * 100
                
                # Determine drift level
                if pct_diff > critical_threshold:
                    drift_level = "critical"
                    drift_detected = True
                elif pct_diff > warning_threshold:
                    drift_level = "warning"
                    drift_detected = True
                else:
                    drift_level = "normal"
                
                metric_drifts[metric] = {
                    'backtest_value': backtest_val,
                    'production_value': production_val,
                    'percentage_diff': pct_diff,
                    'drift_level': drift_level
                }
        
        return {
            'drift_detected': drift_detected,
            'metric_drifts': metric_drifts,
            'test_type': 'percentage_comparison'
        }
    
    async def _detect_distribution_drift(self) -> Dict[str, Any]:
        """Detect distribution drift using Kolmogorov-Smirnov test."""
        
        if len(self.performance_history) < 50:
            return {'drift_detected': False, 'reason': 'insufficient_data'}
        
        # Extract recent production metrics
        recent_data = self.performance_history[-30:]  # Last 30 periods
        historical_data = self.performance_history[-100:-30]  # Previous 70 periods
        
        drift_detected = False
        test_results = {}
        
        for metric in ['win_rate', 'profit_factor', 'sharpe_ratio']:
            try:
                recent_values = [d['production'].get(metric, 0) for d in recent_data if metric in d['production']]
                historical_values = [d['production'].get(metric, 0) for d in historical_data if metric in d['production']]
                
                if len(recent_values) > 10 and len(historical_values) > 10:
                    # Perform Kolmogorov-Smirnov test
                    ks_stat, p_value = stats.ks_2samp(recent_values, historical_values)
                    
                    # Check for significant difference (p < 0.05)
                    if p_value < 0.05:
                        drift_detected = True
                    
                    test_results[metric] = {
                        'ks_statistic': ks_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'recent_mean': np.mean(recent_values),
                        'historical_mean': np.mean(historical_values)
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to perform KS test for {metric}: {e}")
        
        return {
            'drift_detected': drift_detected,
            'test_results': test_results,
            'test_type': 'kolmogorov_smirnov'
        }
    
    async def _generate_drift_recommendations(self, drift_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on drift detection results."""
        recommendations = []
        
        if DriftType.PERFORMANCE.value in drift_results['drift_types']:
            recommendations.extend([
                "Review and recalibrate strategy parameters",
                "Analyze market regime changes that may affect performance",
                "Consider retraining models with recent data",
                "Implement additional risk controls if performance degradation continues"
            ])
        
        if DriftType.DISTRIBUTION.value in drift_results['drift_types']:
            recommendations.extend([
                "Investigate changes in market microstructure",
                "Review feature engineering pipeline for potential issues",
                "Consider walk-forward analysis to validate model stability",
                "Monitor correlation changes with other strategies"
            ])
        
        return recommendations
    
    def _determine_consistency_level(self, score: float) -> ConsistencyLevel:
        """Determine consistency level based on score."""
        if score >= 95:
            return ConsistencyLevel.PERFECT
        elif score >= 85:
            return ConsistencyLevel.ACCEPTABLE
        elif score >= 70:
            return ConsistencyLevel.WARNING
        else:
            return ConsistencyLevel.CRITICAL
    
    async def _generate_consistency_alert(self,
                                        alert_type: str,
                                        severity: str,
                                        message: str,
                                        details: Dict[str, Any]):
        """Generate and publish consistency alert."""
        try:
            alert = AlertEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                alert_id=f"consistency_{int(time.time())}_{alert_type.lower()}",
                severity=severity,
                category="TRADING_CONSISTENCY",
                message=message,
                affected_systems=["trading_engine", "position_manager", "order_executor"],
                recommended_action=self._get_recommended_action(alert_type, severity),
                auto_remediation=False,
                source="trading_consistency_monitor",
                tags={
                    "alert_type": alert_type,
                    "component": "consistency_monitor",
                    "strategy": "dipmaster"
                }
            )
            
            if self.kafka_producer:
                await self.kafka_producer.publish_alert(alert)
            
            self.stats['alerts_generated'] += 1
            logger.warning(f"🚨 Generated consistency alert: {alert_type} - {message}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate consistency alert: {e}")
    
    def _get_recommended_action(self, alert_type: str, severity: str) -> str:
        """Get recommended action for alert type."""
        actions = {
            "SIGNAL_PARAMETER_VIOLATION": "Review signal generation parameters and strategy configuration",
            "ORPHANED_POSITION": "Investigate signal-position mapping and ensure proper tracking",
            "SIGNAL_POSITION_INCONSISTENCY": "Verify signal processing and position creation logic",
            "POSITION_EXECUTION_INCONSISTENCY": "Check order execution system and venue connectivity",
            "BOUNDARY_COMPLIANCE_VIOLATION": "Review timing manager and 15-minute boundary enforcement",
            "BACKTEST_PRODUCTION_DRIFT": "Analyze market conditions and consider strategy recalibration",
            "ORPHANED_EXECUTION": "Investigate execution tracking and position management system"
        }
        
        base_action = actions.get(alert_type, "Investigate the reported issue and take corrective action")
        
        if severity == "CRITICAL":
            return f"URGENT: {base_action}. Consider halting trading until resolved."
        else:
            return base_action
    
    async def update_production_metrics(self, metrics: Dict[str, float]):
        """Update production metrics for drift detection."""
        self.production_performance.update(metrics)
        
        # Check for drift if backtest metrics are available
        if self.backtest_performance:
            await self.detect_backtest_production_drift(
                self.backtest_performance,
                self.production_performance
            )
    
    async def _update_production_metrics(self, execution: ExecutionData):
        """Update production metrics based on execution data."""
        # This would integrate with the business KPI tracker to get current metrics
        # For now, we'll use placeholder logic
        pass
    
    def get_consistency_report(self) -> Dict[str, Any]:
        """Generate comprehensive consistency report."""
        current_time = datetime.now(timezone.utc)
        
        # Calculate consistency scores
        total_checks = self.stats['consistency_checks']
        violations = self.stats['violations_detected']
        consistency_rate = (total_checks - violations) / max(total_checks, 1) * 100
        
        # Get recent alerts
        recent_violations = []  # Would be populated from alert history
        
        report = {
            'timestamp': current_time.isoformat(),
            'summary': {
                'overall_consistency_rate': consistency_rate,
                'total_checks_performed': total_checks,
                'violations_detected': violations,
                'alerts_generated': self.stats['alerts_generated']
            },
            'statistics': self.stats,
            'data_counts': {
                'signals_tracked': len(self.signals),
                'positions_tracked': len(self.positions),
                'executions_tracked': len(self.executions)
            },
            'thresholds': self.consistency_thresholds,
            'recent_violations': recent_violations[-10:],  # Last 10
            'recommendations': self._get_general_recommendations(consistency_rate)
        }
        
        return report
    
    def _get_general_recommendations(self, consistency_rate: float) -> List[str]:
        """Get general recommendations based on consistency rate."""
        if consistency_rate >= 95:
            return ["System operating at optimal consistency levels", "Continue current monitoring procedures"]
        elif consistency_rate >= 85:
            return [
                "Good consistency levels with minor issues",
                "Review recent violations for improvement opportunities",
                "Consider tightening monitoring thresholds"
            ]
        elif consistency_rate >= 70:
            return [
                "Consistency issues detected - investigation required",
                "Review signal generation and execution systems",
                "Implement additional validation checks"
            ]
        else:
            return [
                "CRITICAL: Severe consistency issues detected",
                "Immediate investigation and corrective action required",
                "Consider halting trading until issues are resolved",
                "Review entire trade processing pipeline"
            ]


# Factory function
def create_consistency_monitor(kafka_producer: Optional[KafkaEventProducer] = None,
                             config: Optional[Dict[str, Any]] = None) -> TradingConsistencyMonitor:
    """Create and configure trading consistency monitor."""
    return TradingConsistencyMonitor(kafka_producer, config)