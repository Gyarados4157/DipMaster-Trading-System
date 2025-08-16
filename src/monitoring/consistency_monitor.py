#!/usr/bin/env python3
"""
Consistency Monitor for DipMaster Trading System
一致性监控系统 - 监控信号-持仓-执行一致性

Features:
- Signal-Position-Execution consistency tracking
- Backtest vs Live drift detection
- Statistical validation and alerts
- Reconciliation and audit trails
- Performance consistency monitoring
- Model degradation detection
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import numpy as np
from datetime import datetime, timezone
import json

logger = logging.getLogger(__name__)


class ConsistencyLevel(Enum):
    """Consistency level enumeration."""
    PERFECT = "perfect"  # 100% consistency
    GOOD = "good"       # 95-99% consistency
    DEGRADED = "degraded"  # 90-95% consistency
    POOR = "poor"       # 80-90% consistency
    CRITICAL = "critical"  # <80% consistency


class DriftSeverity(Enum):
    """Drift severity levels."""
    NONE = "none"       # <2% drift
    LOW = "low"         # 2-5% drift
    MEDIUM = "medium"   # 5-10% drift
    HIGH = "high"       # 10-20% drift
    CRITICAL = "critical"  # >20% drift


@dataclass
class SignalRecord:
    """Signal generation record."""
    signal_id: str
    timestamp: float
    symbol: str
    side: str
    confidence: float
    parameters: Dict[str, Any]
    strategy_version: str
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None


@dataclass
class PositionRecord:
    """Position management record."""
    position_id: str
    signal_id: str
    timestamp: float
    symbol: str
    side: str
    entry_price: float
    quantity: float
    status: str  # 'opened', 'modified', 'closed'
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None


@dataclass
class ExecutionRecord:
    """Execution record."""
    execution_id: str
    signal_id: str
    position_id: str
    timestamp: float
    symbol: str
    side: str
    quantity: float
    price: float
    slippage_bps: float
    latency_ms: float
    status: str
    venue: str
    fees: Optional[float] = None


@dataclass
class ConsistencyMetrics:
    """Consistency tracking metrics."""
    signal_position_consistency: float = 0.0
    position_execution_consistency: float = 0.0
    signal_execution_consistency: float = 0.0
    timing_consistency: float = 0.0
    price_consistency: float = 0.0
    quantity_consistency: float = 0.0
    overall_consistency: float = 0.0
    
    # Detailed breakdown
    signals_generated: int = 0
    positions_opened: int = 0
    executions_completed: int = 0
    signals_without_positions: int = 0
    positions_without_executions: int = 0
    orphaned_executions: int = 0


@dataclass
class DriftMetrics:
    """Model/strategy drift metrics."""
    performance_drift: float = 0.0
    signal_distribution_drift: float = 0.0
    execution_quality_drift: float = 0.0
    risk_profile_drift: float = 0.0
    correlation_drift: float = 0.0
    volatility_drift: float = 0.0
    overall_drift: float = 0.0
    
    # Statistical tests
    ks_test_p_value: Optional[float] = None
    chi_square_p_value: Optional[float] = None
    variance_test_p_value: Optional[float] = None


class ConsistencyMonitor:
    """
    Comprehensive consistency monitoring system.
    
    Tracks and validates consistency across the entire trading pipeline
    from signal generation through position management to execution.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 retention_hours: int = 168):  # 1 week default
        """
        Initialize consistency monitor.
        
        Args:
            config: Configuration parameters
            retention_hours: How long to retain detailed records
        """
        self.config = config or {}
        self.retention_hours = retention_hours
        
        # Data storage
        self.signals = {}  # signal_id -> SignalRecord
        self.positions = {}  # position_id -> PositionRecord
        self.executions = {}  # execution_id -> ExecutionRecord
        
        # Time-series data for trend analysis
        self.signal_history = deque(maxlen=10000)
        self.position_history = deque(maxlen=10000)
        self.execution_history = deque(maxlen=10000)
        
        # Consistency tracking
        self.consistency_metrics = ConsistencyMetrics()
        self.consistency_history = deque(maxlen=1000)
        
        # Drift detection
        self.drift_metrics = DriftMetrics()
        self.drift_history = deque(maxlen=1000)
        
        # Backtest baseline (for drift comparison)
        self.backtest_baseline = {}
        
        # Reconciliation tracking
        self.reconciliation_issues = deque(maxlen=1000)
        self.last_reconciliation = None
        
        logger.info("📊 ConsistencyMonitor initialized")
    
    def record_signal(self, signal: SignalRecord):
        """Record a trading signal."""
        try:
            self.signals[signal.signal_id] = signal
            self.signal_history.append({
                'timestamp': signal.timestamp,
                'signal_id': signal.signal_id,
                'symbol': signal.symbol,
                'side': signal.side,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price
            })
            
            self.consistency_metrics.signals_generated += 1
            logger.debug(f"📡 Recorded signal: {signal.signal_id} for {signal.symbol}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record signal: {e}")
    
    def record_position(self, position: PositionRecord):
        """Record a position update."""
        try:
            self.positions[position.position_id] = position
            self.position_history.append({
                'timestamp': position.timestamp,
                'position_id': position.position_id,
                'signal_id': position.signal_id,
                'symbol': position.symbol,
                'side': position.side,
                'entry_price': position.entry_price,
                'quantity': position.quantity,
                'status': position.status
            })
            
            if position.status == 'opened':
                self.consistency_metrics.positions_opened += 1
            
            logger.debug(f"📍 Recorded position: {position.position_id} for {position.symbol}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record position: {e}")
    
    def record_execution(self, execution: ExecutionRecord):
        """Record an execution."""
        try:
            self.executions[execution.execution_id] = execution
            self.execution_history.append({
                'timestamp': execution.timestamp,
                'execution_id': execution.execution_id,
                'signal_id': execution.signal_id,
                'position_id': execution.position_id,
                'symbol': execution.symbol,
                'side': execution.side,
                'quantity': execution.quantity,
                'price': execution.price,
                'slippage_bps': execution.slippage_bps,
                'latency_ms': execution.latency_ms,
                'status': execution.status,
                'venue': execution.venue
            })
            
            if execution.status == 'filled':
                self.consistency_metrics.executions_completed += 1
            
            logger.debug(f"⚡ Recorded execution: {execution.execution_id} for {execution.symbol}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record execution: {e}")
    
    def calculate_consistency_metrics(self) -> ConsistencyMetrics:
        """Calculate comprehensive consistency metrics."""
        try:
            # Calculate signal-position consistency
            signal_position_pairs = self._match_signals_to_positions()
            self.consistency_metrics.signal_position_consistency = len(signal_position_pairs) / max(len(self.signals), 1)
            
            # Calculate position-execution consistency
            position_execution_pairs = self._match_positions_to_executions()
            self.consistency_metrics.position_execution_consistency = len(position_execution_pairs) / max(len(self.positions), 1)
            
            # Calculate signal-execution consistency (end-to-end)
            signal_execution_pairs = self._match_signals_to_executions()
            self.consistency_metrics.signal_execution_consistency = len(signal_execution_pairs) / max(len(self.signals), 1)
            
            # Calculate timing consistency
            self.consistency_metrics.timing_consistency = self._calculate_timing_consistency()
            
            # Calculate price consistency
            self.consistency_metrics.price_consistency = self._calculate_price_consistency()
            
            # Calculate quantity consistency
            self.consistency_metrics.quantity_consistency = self._calculate_quantity_consistency()
            
            # Calculate overall consistency score
            consistency_scores = [
                self.consistency_metrics.signal_position_consistency,
                self.consistency_metrics.position_execution_consistency,
                self.consistency_metrics.timing_consistency,
                self.consistency_metrics.price_consistency,
                self.consistency_metrics.quantity_consistency
            ]
            self.consistency_metrics.overall_consistency = statistics.mean(consistency_scores)
            
            # Count orphaned records
            self._count_orphaned_records()
            
            # Store in history
            self.consistency_history.append({
                'timestamp': time.time(),
                'metrics': self.consistency_metrics
            })
            
            logger.debug(f"📊 Calculated consistency: {self.consistency_metrics.overall_consistency:.3f}")
            return self.consistency_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate consistency metrics: {e}")
            return self.consistency_metrics
    
    def _match_signals_to_positions(self) -> List[Tuple[str, str]]:
        """Match signals to their corresponding positions."""
        matches = []
        for position in self.positions.values():
            if position.signal_id in self.signals:
                matches.append((position.signal_id, position.position_id))
        return matches
    
    def _match_positions_to_executions(self) -> List[Tuple[str, str]]:
        """Match positions to their corresponding executions."""
        matches = []
        for execution in self.executions.values():
            if execution.position_id in self.positions:
                matches.append((execution.position_id, execution.execution_id))
        return matches
    
    def _match_signals_to_executions(self) -> List[Tuple[str, str]]:
        """Match signals to their corresponding executions (end-to-end)."""
        matches = []
        for execution in self.executions.values():
            if execution.signal_id in self.signals:
                matches.append((execution.signal_id, execution.execution_id))
        return matches
    
    def _calculate_timing_consistency(self) -> float:
        """Calculate timing consistency between pipeline stages."""
        try:
            timing_scores = []
            
            # Check signal-to-position timing
            for position in self.positions.values():
                if position.signal_id in self.signals:
                    signal = self.signals[position.signal_id]
                    time_diff = position.timestamp - signal.timestamp
                    # Good timing: < 60 seconds
                    score = max(0, 1 - (time_diff / 60))
                    timing_scores.append(score)
            
            return statistics.mean(timing_scores) if timing_scores else 0.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating timing consistency: {e}")
            return 0.0
    
    def _calculate_price_consistency(self) -> float:
        """Calculate price consistency between stages."""
        try:
            price_scores = []
            
            # Check signal-to-execution price consistency
            for execution in self.executions.values():
                if execution.signal_id in self.signals:
                    signal = self.signals[execution.signal_id]
                    price_diff_pct = abs(execution.price - signal.entry_price) / signal.entry_price
                    # Good consistency: < 1% price difference
                    score = max(0, 1 - (price_diff_pct / 0.01))
                    price_scores.append(score)
            
            return statistics.mean(price_scores) if price_scores else 0.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating price consistency: {e}")
            return 0.0
    
    def _calculate_quantity_consistency(self) -> float:
        """Calculate quantity consistency between stages."""
        try:
            quantity_scores = []
            
            # Check signal-to-position quantity consistency
            for position in self.positions.values():
                if position.signal_id in self.signals:
                    signal = self.signals[position.signal_id]
                    if signal.position_size:
                        qty_diff_pct = abs(position.quantity - signal.position_size) / signal.position_size
                        # Good consistency: < 5% quantity difference
                        score = max(0, 1 - (qty_diff_pct / 0.05))
                        quantity_scores.append(score)
            
            return statistics.mean(quantity_scores) if quantity_scores else 1.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating quantity consistency: {e}")
            return 1.0
    
    def _count_orphaned_records(self):
        """Count orphaned records (signals without positions, etc.)."""
        # Signals without positions
        signals_with_positions = {pos.signal_id for pos in self.positions.values()}
        self.consistency_metrics.signals_without_positions = len(self.signals) - len(signals_with_positions)
        
        # Positions without executions
        positions_with_executions = {exec.position_id for exec in self.executions.values()}
        self.consistency_metrics.positions_without_executions = len(self.positions) - len(positions_with_executions)
        
        # Orphaned executions (no matching signal or position)
        orphaned_count = 0
        for execution in self.executions.values():
            if execution.signal_id not in self.signals or execution.position_id not in self.positions:
                orphaned_count += 1
        self.consistency_metrics.orphaned_executions = orphaned_count
    
    def set_backtest_baseline(self, baseline: Dict[str, Any]):
        """Set backtest baseline for drift detection."""
        self.backtest_baseline = baseline
        logger.info(f"📊 Set backtest baseline with {len(baseline)} metrics")
    
    def calculate_drift_metrics(self, live_metrics: Dict[str, Any]) -> DriftMetrics:
        """Calculate drift between live and backtest performance."""
        try:
            if not self.backtest_baseline:
                logger.warning("⚠️ No backtest baseline set for drift calculation")
                return self.drift_metrics
            
            # Performance drift (win rate, sharpe ratio, etc.)
            self.drift_metrics.performance_drift = self._calculate_performance_drift(live_metrics)
            
            # Signal distribution drift
            self.drift_metrics.signal_distribution_drift = self._calculate_signal_drift()
            
            # Execution quality drift
            self.drift_metrics.execution_quality_drift = self._calculate_execution_drift()
            
            # Risk profile drift
            self.drift_metrics.risk_profile_drift = self._calculate_risk_drift(live_metrics)
            
            # Overall drift score
            drift_scores = [
                self.drift_metrics.performance_drift,
                self.drift_metrics.signal_distribution_drift,
                self.drift_metrics.execution_quality_drift,
                self.drift_metrics.risk_profile_drift
            ]
            self.drift_metrics.overall_drift = statistics.mean(drift_scores)
            
            # Statistical tests
            self._run_statistical_tests(live_metrics)
            
            # Store in history
            self.drift_history.append({
                'timestamp': time.time(),
                'metrics': self.drift_metrics
            })
            
            logger.debug(f"📈 Calculated drift: {self.drift_metrics.overall_drift:.3f}")
            return self.drift_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate drift metrics: {e}")
            return self.drift_metrics
    
    def _calculate_performance_drift(self, live_metrics: Dict[str, Any]) -> float:
        """Calculate performance drift from backtest baseline."""
        try:
            drift = 0.0
            comparisons = 0
            
            # Compare key performance metrics
            key_metrics = ['win_rate', 'sharpe_ratio', 'avg_trade_pnl', 'max_drawdown']
            
            for metric in key_metrics:
                if metric in self.backtest_baseline and metric in live_metrics:
                    baseline_val = self.backtest_baseline[metric]
                    live_val = live_metrics[metric]
                    
                    if baseline_val != 0:
                        metric_drift = abs(live_val - baseline_val) / abs(baseline_val)
                        drift += metric_drift
                        comparisons += 1
            
            return drift / max(comparisons, 1)
            
        except Exception as e:
            logger.error(f"❌ Error calculating performance drift: {e}")
            return 0.0
    
    def _calculate_signal_drift(self) -> float:
        """Calculate signal distribution drift."""
        try:
            if len(self.signal_history) < 100:
                return 0.0
            
            # Analyze recent vs historical signal patterns
            recent_signals = list(self.signal_history)[-100:]
            historical_signals = list(self.signal_history)[:-100] if len(self.signal_history) > 100 else []
            
            if not historical_signals:
                return 0.0
            
            # Compare confidence score distributions
            recent_confidence = [s['confidence'] for s in recent_signals]
            historical_confidence = [s['confidence'] for s in historical_signals]
            
            # Simple drift calculation based on mean difference
            recent_mean = statistics.mean(recent_confidence)
            historical_mean = statistics.mean(historical_confidence)
            
            drift = abs(recent_mean - historical_mean) / max(historical_mean, 0.01)
            return min(drift, 1.0)  # Cap at 100% drift
            
        except Exception as e:
            logger.error(f"❌ Error calculating signal drift: {e}")
            return 0.0
    
    def _calculate_execution_drift(self) -> float:
        """Calculate execution quality drift."""
        try:
            if len(self.execution_history) < 50:
                return 0.0
            
            # Analyze recent vs historical execution quality
            recent_executions = list(self.execution_history)[-50:]
            historical_executions = list(self.execution_history)[:-50] if len(self.execution_history) > 50 else []
            
            if not historical_executions:
                return 0.0
            
            # Compare slippage and latency
            recent_slippage = [e['slippage_bps'] for e in recent_executions if e['slippage_bps'] is not None]
            historical_slippage = [e['slippage_bps'] for e in historical_executions if e['slippage_bps'] is not None]
            
            if not recent_slippage or not historical_slippage:
                return 0.0
            
            recent_mean_slippage = statistics.mean(recent_slippage)
            historical_mean_slippage = statistics.mean(historical_slippage)
            
            drift = abs(recent_mean_slippage - historical_mean_slippage) / max(historical_mean_slippage, 1.0)
            return min(drift, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Error calculating execution drift: {e}")
            return 0.0
    
    def _calculate_risk_drift(self, live_metrics: Dict[str, Any]) -> float:
        """Calculate risk profile drift."""
        try:
            if 'volatility' not in self.backtest_baseline or 'volatility' not in live_metrics:
                return 0.0
            
            baseline_vol = self.backtest_baseline['volatility']
            live_vol = live_metrics['volatility']
            
            if baseline_vol == 0:
                return 0.0
            
            vol_drift = abs(live_vol - baseline_vol) / baseline_vol
            return min(vol_drift, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Error calculating risk drift: {e}")
            return 0.0
    
    def _run_statistical_tests(self, live_metrics: Dict[str, Any]):
        """Run statistical tests for drift detection."""
        try:
            # Placeholder for statistical tests
            # In production, would use scipy.stats for KS test, chi-square, etc.
            self.drift_metrics.ks_test_p_value = None
            self.drift_metrics.chi_square_p_value = None
            self.drift_metrics.variance_test_p_value = None
            
        except Exception as e:
            logger.error(f"❌ Error running statistical tests: {e}")
    
    def perform_reconciliation(self) -> Dict[str, Any]:
        """Perform comprehensive reconciliation check."""
        try:
            reconciliation_report = {
                'timestamp': time.time(),
                'signals_count': len(self.signals),
                'positions_count': len(self.positions),
                'executions_count': len(self.executions),
                'issues': [],
                'consistency_score': self.consistency_metrics.overall_consistency,
                'recommendations': []
            }
            
            # Check for missing position links
            signals_without_positions = []
            for signal_id in self.signals:
                if not any(pos.signal_id == signal_id for pos in self.positions.values()):
                    signals_without_positions.append(signal_id)
            
            if signals_without_positions:
                issue = {
                    'type': 'missing_positions',
                    'count': len(signals_without_positions),
                    'signal_ids': signals_without_positions[:10],  # Limit for brevity
                    'severity': 'medium'
                }
                reconciliation_report['issues'].append(issue)
                reconciliation_report['recommendations'].append(
                    "Investigate signal-to-position conversion process"
                )
            
            # Check for missing execution links
            positions_without_executions = []
            for position_id in self.positions:
                if not any(exec.position_id == position_id for exec in self.executions.values()):
                    positions_without_executions.append(position_id)
            
            if positions_without_executions:
                issue = {
                    'type': 'missing_executions',
                    'count': len(positions_without_executions),
                    'position_ids': positions_without_executions[:10],
                    'severity': 'high'
                }
                reconciliation_report['issues'].append(issue)
                reconciliation_report['recommendations'].append(
                    "Review position-to-execution conversion and order management"
                )
            
            # Check for timing issues
            timing_issues = self._check_timing_issues()
            if timing_issues:
                reconciliation_report['issues'].extend(timing_issues)
            
            # Determine overall reconciliation status
            if not reconciliation_report['issues']:
                reconciliation_report['status'] = 'clean'
            elif len([i for i in reconciliation_report['issues'] if i['severity'] == 'high']) > 0:
                reconciliation_report['status'] = 'issues_found'
            else:
                reconciliation_report['status'] = 'minor_issues'
            
            # Store reconciliation result
            self.reconciliation_issues.append(reconciliation_report)
            self.last_reconciliation = time.time()
            
            logger.info(f"🔍 Reconciliation complete: {reconciliation_report['status']} "
                       f"({len(reconciliation_report['issues'])} issues)")
            
            return reconciliation_report
            
        except Exception as e:
            logger.error(f"❌ Failed to perform reconciliation: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def _check_timing_issues(self) -> List[Dict[str, Any]]:
        """Check for timing-related issues."""
        issues = []
        
        try:
            # Check for stale signals (signals without recent activity)
            current_time = time.time()
            stale_threshold = 3600  # 1 hour
            
            stale_signals = []
            for signal in self.signals.values():
                if current_time - signal.timestamp > stale_threshold:
                    # Check if there's any related activity
                    has_position = any(pos.signal_id == signal.signal_id for pos in self.positions.values())
                    has_execution = any(exec.signal_id == signal.signal_id for exec in self.executions.values())
                    
                    if not has_position and not has_execution:
                        stale_signals.append(signal.signal_id)
            
            if stale_signals:
                issues.append({
                    'type': 'stale_signals',
                    'count': len(stale_signals),
                    'signal_ids': stale_signals[:10],
                    'severity': 'low',
                    'description': f'Signals older than {stale_threshold/3600} hours without activity'
                })
            
        except Exception as e:
            logger.error(f"❌ Error checking timing issues: {e}")
        
        return issues
    
    def get_consistency_level(self) -> ConsistencyLevel:
        """Get current consistency level."""
        score = self.consistency_metrics.overall_consistency
        
        if score >= 0.99:
            return ConsistencyLevel.PERFECT
        elif score >= 0.95:
            return ConsistencyLevel.GOOD
        elif score >= 0.90:
            return ConsistencyLevel.DEGRADED
        elif score >= 0.80:
            return ConsistencyLevel.POOR
        else:
            return ConsistencyLevel.CRITICAL
    
    def get_drift_severity(self) -> DriftSeverity:
        """Get current drift severity."""
        drift = self.drift_metrics.overall_drift
        
        if drift < 0.02:
            return DriftSeverity.NONE
        elif drift < 0.05:
            return DriftSeverity.LOW
        elif drift < 0.10:
            return DriftSeverity.MEDIUM
        elif drift < 0.20:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        try:
            return {
                'timestamp': time.time(),
                'consistency': {
                    'level': self.get_consistency_level().value,
                    'score': self.consistency_metrics.overall_consistency,
                    'metrics': {
                        'signal_position': self.consistency_metrics.signal_position_consistency,
                        'position_execution': self.consistency_metrics.position_execution_consistency,
                        'timing': self.consistency_metrics.timing_consistency,
                        'price': self.consistency_metrics.price_consistency,
                        'quantity': self.consistency_metrics.quantity_consistency
                    },
                    'counts': {
                        'signals': self.consistency_metrics.signals_generated,
                        'positions': self.consistency_metrics.positions_opened,
                        'executions': self.consistency_metrics.executions_completed,
                        'orphaned_signals': self.consistency_metrics.signals_without_positions,
                        'orphaned_positions': self.consistency_metrics.positions_without_executions,
                        'orphaned_executions': self.consistency_metrics.orphaned_executions
                    }
                },
                'drift': {
                    'severity': self.get_drift_severity().value,
                    'score': self.drift_metrics.overall_drift,
                    'components': {
                        'performance': self.drift_metrics.performance_drift,
                        'signals': self.drift_metrics.signal_distribution_drift,
                        'execution': self.drift_metrics.execution_quality_drift,
                        'risk': self.drift_metrics.risk_profile_drift
                    }
                },
                'reconciliation': {
                    'last_check': self.last_reconciliation,
                    'issues_found': len(self.reconciliation_issues),
                    'status': 'current' if self.last_reconciliation and (time.time() - self.last_reconciliation < 3600) else 'overdue'
                },
                'data_counts': {
                    'signals': len(self.signals),
                    'positions': len(self.positions),
                    'executions': len(self.executions),
                    'signal_history': len(self.signal_history),
                    'position_history': len(self.position_history),
                    'execution_history': len(self.execution_history)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get monitoring summary: {e}")
            return {'error': str(e)}
    
    def cleanup_old_data(self):
        """Clean up old data based on retention policy."""
        try:
            current_time = time.time()
            retention_seconds = self.retention_hours * 3600
            
            # Clean up signals
            old_signals = [sid for sid, signal in self.signals.items() 
                          if current_time - signal.timestamp > retention_seconds]
            for sid in old_signals:
                del self.signals[sid]
            
            # Clean up positions
            old_positions = [pid for pid, position in self.positions.items() 
                            if current_time - position.timestamp > retention_seconds]
            for pid in old_positions:
                del self.positions[pid]
            
            # Clean up executions
            old_executions = [eid for eid, execution in self.executions.items() 
                             if current_time - execution.timestamp > retention_seconds]
            for eid in old_executions:
                del self.executions[eid]
            
            if old_signals or old_positions or old_executions:
                logger.info(f"🧹 Cleaned up old data: {len(old_signals)} signals, "
                           f"{len(old_positions)} positions, {len(old_executions)} executions")
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old data: {e}")


# Factory function
def create_consistency_monitor(config: Dict[str, Any]) -> ConsistencyMonitor:
    """Create and configure consistency monitor."""
    return ConsistencyMonitor(
        config=config.get('consistency_config', {}),
        retention_hours=config.get('retention_hours', 168)
    )