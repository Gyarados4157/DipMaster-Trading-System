#!/usr/bin/env python3
"""
DipMaster Strategy Monitor for DipMaster Trading System
DipMaster策略专用监控器 - 策略特定规则和合规检查

Features:
- 15-minute boundary compliance monitoring
- DipMaster strategy parameter validation
- RSI range and volume confirmation checks
- Exit timing optimization monitoring
- Strategy performance drift detection
- Real-time strategy health scoring
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
from collections import defaultdict, deque

from .trading_consistency_monitor import TradingConsistencyMonitor, SignalData, PositionData
from .enhanced_event_producer import EnhancedEventProducer
from .structured_logging_system import StructuredLogger, LogCategory, LogContext

logger = logging.getLogger(__name__)


class StrategyHealthLevel(Enum):
    """Strategy health levels."""
    EXCELLENT = "excellent"      # 90-100
    GOOD = "good"               # 75-89
    FAIR = "fair"               # 60-74
    POOR = "poor"               # 40-59
    CRITICAL = "critical"       # 0-39


class DipMasterRule(Enum):
    """DipMaster strategy rules."""
    RSI_RANGE_CHECK = "rsi_range_check"
    DIP_CONFIRMATION = "dip_confirmation"  
    VOLUME_CONFIRMATION = "volume_confirmation"
    BOUNDARY_EXIT_TIMING = "boundary_exit_timing"
    MAX_HOLDING_TIME = "max_holding_time"
    MA20_POSITION_CHECK = "ma20_position_check"
    TARGET_PROFIT_CHECK = "target_profit_check"
    SIGNAL_CONFIDENCE_CHECK = "signal_confidence_check"


@dataclass
class DipMasterParameters:
    """DipMaster strategy parameters."""
    rsi_min: float = 30.0
    rsi_max: float = 50.0
    dip_threshold_pct: float = 0.2
    volume_multiplier: float = 1.5
    max_holding_minutes: int = 180
    target_profit_pct: float = 0.8
    boundary_minutes: List[int] = None
    min_confidence: float = 0.6
    ma20_deviation_max: float = -0.001  # Must be below MA20
    
    def __post_init__(self):
        if self.boundary_minutes is None:
            self.boundary_minutes = [15, 30, 45, 60]


@dataclass
class StrategyViolation:
    """Strategy rule violation."""
    rule: DipMasterRule
    severity: str  # "INFO", "WARNING", "ERROR", "CRITICAL"
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    signal_id: Optional[str] = None
    position_id: Optional[str] = None
    symbol: Optional[str] = None


@dataclass
class StrategyHealthReport:
    """Strategy health assessment report."""
    timestamp: datetime
    overall_score: float
    health_level: StrategyHealthLevel
    compliance_rate: float
    rule_scores: Dict[DipMasterRule, float]
    violations: List[StrategyViolation]
    recommendations: List[str]
    performance_metrics: Dict[str, float]


class DipMasterStrategyMonitor:
    """
    DipMaster strategy-specific monitoring system.
    
    Monitors compliance with DipMaster strategy rules, validates
    parameter execution, and provides strategy health assessment.
    """
    
    def __init__(self,
                 event_producer: Optional[EnhancedEventProducer] = None,
                 structured_logger: Optional[StructuredLogger] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize DipMaster strategy monitor.
        
        Args:
            event_producer: Event producer for alerts
            structured_logger: Structured logger for detailed logging
            config: Configuration parameters
        """
        self.event_producer = event_producer
        self.structured_logger = structured_logger
        self.config = config or {}
        
        # DipMaster parameters
        self.params = DipMasterParameters(**self.config.get('dipmaster_params', {}))
        
        # Monitoring data
        self.signals_tracked = {}
        self.positions_tracked = {}
        self.violations = deque(maxlen=1000)
        self.compliance_history = deque(maxlen=100)
        
        # Rule tracking
        self.rule_checks = defaultdict(int)
        self.rule_violations = defaultdict(int)
        self.rule_scores = {rule: 100.0 for rule in DipMasterRule}
        
        # Performance tracking
        self.boundary_exits = defaultdict(int)
        self.exit_timing_stats = deque(maxlen=500)
        self.holding_time_stats = deque(maxlen=500)
        
        # Health scoring
        self.health_history = deque(maxlen=100)
        self.last_health_check = 0
        self.health_check_interval = 300  # 5 minutes
        
        logger.info("🎯 DipMasterStrategyMonitor initialized")
    
    async def validate_signal(self, signal: SignalData) -> List[StrategyViolation]:
        """Validate signal against DipMaster strategy rules."""
        violations = []
        current_time = datetime.now(timezone.utc)
        
        # Store signal for tracking
        self.signals_tracked[signal.signal_id] = signal
        
        # Rule 1: RSI Range Check
        rsi_violation = await self._check_rsi_range(signal)
        if rsi_violation:
            violations.append(rsi_violation)
        
        # Rule 2: Dip Confirmation
        dip_violation = await self._check_dip_confirmation(signal)
        if dip_violation:
            violations.append(dip_violation)
        
        # Rule 3: Volume Confirmation
        volume_violation = await self._check_volume_confirmation(signal)
        if volume_violation:
            violations.append(volume_violation)
        
        # Rule 4: MA20 Position Check
        ma20_violation = await self._check_ma20_position(signal)
        if ma20_violation:
            violations.append(ma20_violation)
        
        # Rule 5: Signal Confidence Check
        confidence_violation = await self._check_signal_confidence(signal)
        if confidence_violation:
            violations.append(confidence_violation)
        
        # Update rule tracking
        for rule in DipMasterRule:
            self.rule_checks[rule] += 1
        
        # Add violations to tracking
        for violation in violations:
            self.violations.append(violation)
            self.rule_violations[violation.rule] += 1
            await self._log_violation(violation)
        
        # Update rule scores
        self._update_rule_scores()
        
        # Log signal validation
        if self.structured_logger:
            self.structured_logger.log_signal_generated(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                confidence=signal.confidence,
                price=signal.price,
                strategy="dipmaster",
                technical_indicators={
                    'rsi': signal.rsi,
                    'ma20_distance': signal.ma20_distance,
                    'volume_ratio': signal.volume_ratio
                },
                context=LogContext(
                    session_id=str(int(time.time())),
                    strategy="dipmaster",
                    symbol=signal.symbol,
                    correlation_id=signal.signal_id
                ),
                data={
                    'violations_count': len(violations),
                    'violations': [v.rule.value for v in violations]
                }
            )
        
        return violations
    
    async def validate_position_exit(self, position: PositionData) -> List[StrategyViolation]:
        """Validate position exit against DipMaster strategy rules."""
        violations = []
        current_time = datetime.now(timezone.utc)
        
        if not position.exit_time or not position.entry_time:
            return violations
        
        # Store position for tracking
        self.positions_tracked[position.position_id] = position
        
        # Rule 6: Boundary Exit Timing
        boundary_violation = await self._check_boundary_exit_timing(position)
        if boundary_violation:
            violations.append(boundary_violation)
        
        # Rule 7: Max Holding Time
        holding_violation = await self._check_max_holding_time(position)
        if holding_violation:
            violations.append(holding_violation)
        
        # Rule 8: Target Profit Check (if applicable)
        if position.pnl and position.pnl > 0:
            profit_violation = await self._check_target_profit(position)
            if profit_violation:
                violations.append(profit_violation)
        
        # Update statistics
        holding_minutes = (position.exit_time - position.entry_time).total_seconds() / 60
        self.holding_time_stats.append(holding_minutes)
        
        exit_minute = position.exit_time.minute
        self.boundary_exits[exit_minute] += 1
        
        # Add violations to tracking
        for violation in violations:
            self.violations.append(violation)
            self.rule_violations[violation.rule] += 1
            await self._log_violation(violation)
        
        # Update rule scores
        self._update_rule_scores()
        
        # Log position exit
        if self.structured_logger:
            self.structured_logger.log_trade_exit(
                symbol=position.symbol,
                side=position.side,
                quantity=position.quantity,
                exit_price=position.exit_price or 0,
                entry_price=position.entry_price,
                pnl=position.pnl or 0,
                holding_minutes=int(holding_minutes),
                trade_id=position.position_id,
                strategy="dipmaster",
                context=LogContext(
                    session_id=str(int(time.time())),
                    strategy="dipmaster",
                    symbol=position.symbol,
                    trade_id=position.position_id
                ),
                data={
                    'violations_count': len(violations),
                    'exit_minute': exit_minute,
                    'boundary_compliant': exit_minute in self.params.boundary_minutes
                }
            )
        
        return violations
    
    async def _check_rsi_range(self, signal: SignalData) -> Optional[StrategyViolation]:
        """Check RSI range compliance."""
        if signal.signal_type != "BUY":
            return None
        
        if not (self.params.rsi_min <= signal.rsi <= self.params.rsi_max):
            return StrategyViolation(
                rule=DipMasterRule.RSI_RANGE_CHECK,
                severity="WARNING",
                message=f"RSI {signal.rsi:.1f} outside optimal range [{self.params.rsi_min}, {self.params.rsi_max}]",
                details={
                    'rsi_value': signal.rsi,
                    'expected_min': self.params.rsi_min,
                    'expected_max': self.params.rsi_max,
                    'deviation': min(abs(signal.rsi - self.params.rsi_min), 
                                   abs(signal.rsi - self.params.rsi_max))
                },
                timestamp=datetime.now(timezone.utc),
                signal_id=signal.signal_id,
                symbol=signal.symbol
            )
        return None
    
    async def _check_dip_confirmation(self, signal: SignalData) -> Optional[StrategyViolation]:
        """Check dip confirmation (price below expected entry)."""
        if signal.signal_type != "BUY":
            return None
        
        # Check if current price indicates a dip
        if signal.price >= signal.expected_entry_price:
            return StrategyViolation(
                rule=DipMasterRule.DIP_CONFIRMATION,
                severity="WARNING",
                message=f"Price {signal.price:.4f} not confirming dip vs expected {signal.expected_entry_price:.4f}",
                details={
                    'current_price': signal.price,
                    'expected_entry': signal.expected_entry_price,
                    'price_difference': signal.price - signal.expected_entry_price
                },
                timestamp=datetime.now(timezone.utc),
                signal_id=signal.signal_id,
                symbol=signal.symbol
            )
        return None
    
    async def _check_volume_confirmation(self, signal: SignalData) -> Optional[StrategyViolation]:
        """Check volume confirmation."""
        if signal.volume_ratio < self.params.volume_multiplier:
            return StrategyViolation(
                rule=DipMasterRule.VOLUME_CONFIRMATION,
                severity="WARNING",
                message=f"Volume ratio {signal.volume_ratio:.2f} below threshold {self.params.volume_multiplier}",
                details={
                    'volume_ratio': signal.volume_ratio,
                    'required_ratio': self.params.volume_multiplier,
                    'volume_deficit': self.params.volume_multiplier - signal.volume_ratio
                },
                timestamp=datetime.now(timezone.utc),
                signal_id=signal.signal_id,
                symbol=signal.symbol
            )
        return None
    
    async def _check_ma20_position(self, signal: SignalData) -> Optional[StrategyViolation]:
        """Check MA20 position (price should be below MA20 for dip buying)."""
        if signal.signal_type != "BUY":
            return None
        
        if signal.ma20_distance > self.params.ma20_deviation_max:
            return StrategyViolation(
                rule=DipMasterRule.MA20_POSITION_CHECK,
                severity="ERROR",
                message=f"Price above MA20 (distance: {signal.ma20_distance:.4f}), violating dip buying logic",
                details={
                    'ma20_distance': signal.ma20_distance,
                    'max_allowed': self.params.ma20_deviation_max,
                    'position_vs_ma20': 'above' if signal.ma20_distance > 0 else 'below'
                },
                timestamp=datetime.now(timezone.utc),
                signal_id=signal.signal_id,
                symbol=signal.symbol
            )
        return None
    
    async def _check_signal_confidence(self, signal: SignalData) -> Optional[StrategyViolation]:
        """Check signal confidence level."""
        if signal.confidence < self.params.min_confidence:
            return StrategyViolation(
                rule=DipMasterRule.SIGNAL_CONFIDENCE_CHECK,
                severity="WARNING",
                message=f"Signal confidence {signal.confidence:.2f} below minimum {self.params.min_confidence}",
                details={
                    'confidence': signal.confidence,
                    'min_required': self.params.min_confidence,
                    'confidence_deficit': self.params.min_confidence - signal.confidence
                },
                timestamp=datetime.now(timezone.utc),
                signal_id=signal.signal_id,
                symbol=signal.symbol
            )
        return None
    
    async def _check_boundary_exit_timing(self, position: PositionData) -> Optional[StrategyViolation]:
        """Check 15-minute boundary exit timing."""
        if not position.exit_time:
            return None
        
        exit_minute = position.exit_time.minute
        
        # Check if exit was at proper boundary
        if exit_minute not in self.params.boundary_minutes:
            # Find closest boundary
            closest_boundary = min(self.params.boundary_minutes, 
                                 key=lambda x: abs(x - exit_minute))
            
            return StrategyViolation(
                rule=DipMasterRule.BOUNDARY_EXIT_TIMING,
                severity="ERROR",
                message=f"Exit at minute {exit_minute} not at 15-minute boundary. Closest: {closest_boundary}",
                details={
                    'exit_minute': exit_minute,
                    'valid_boundaries': self.params.boundary_minutes,
                    'closest_boundary': closest_boundary,
                    'deviation_minutes': abs(exit_minute - closest_boundary)
                },
                timestamp=datetime.now(timezone.utc),
                position_id=position.position_id,
                symbol=position.symbol
            )
        return None
    
    async def _check_max_holding_time(self, position: PositionData) -> Optional[StrategyViolation]:
        """Check maximum holding time compliance."""
        if not position.entry_time or not position.exit_time:
            return None
        
        holding_minutes = (position.exit_time - position.entry_time).total_seconds() / 60
        
        if holding_minutes > self.params.max_holding_minutes:
            return StrategyViolation(
                rule=DipMasterRule.MAX_HOLDING_TIME,
                severity="ERROR",
                message=f"Holding time {holding_minutes:.1f}min exceeds maximum {self.params.max_holding_minutes}min",
                details={
                    'holding_minutes': holding_minutes,
                    'max_allowed': self.params.max_holding_minutes,
                    'excess_time': holding_minutes - self.params.max_holding_minutes
                },
                timestamp=datetime.now(timezone.utc),
                position_id=position.position_id,
                symbol=position.symbol
            )
        return None
    
    async def _check_target_profit(self, position: PositionData) -> Optional[StrategyViolation]:
        """Check if target profit was achieved efficiently."""
        if not position.pnl or not position.entry_time or not position.exit_time:
            return None
        
        # Calculate profit percentage
        if position.entry_price > 0:
            profit_pct = (position.pnl / (position.entry_price * position.quantity)) * 100
            
            # If profit was achieved, check if it met target
            if profit_pct > 0 and profit_pct < self.params.target_profit_pct:
                return StrategyViolation(
                    rule=DipMasterRule.TARGET_PROFIT_CHECK,
                    severity="INFO",
                    message=f"Profit {profit_pct:.2f}% below target {self.params.target_profit_pct}%",
                    details={
                        'profit_pct': profit_pct,
                        'target_pct': self.params.target_profit_pct,
                        'profit_deficit': self.params.target_profit_pct - profit_pct
                    },
                    timestamp=datetime.now(timezone.utc),
                    position_id=position.position_id,
                    symbol=position.symbol
                )
        return None
    
    async def _log_violation(self, violation: StrategyViolation):
        """Log strategy violation."""
        # Publish alert for significant violations
        if violation.severity in ["ERROR", "CRITICAL"] and self.event_producer:
            await self.event_producer.publish_alert(
                alert_id=f"dipmaster_violation_{int(time.time())}",
                severity=violation.severity,
                category="STRATEGY_COMPLIANCE",
                message=violation.message,
                affected_systems=["strategy_engine", "dipmaster"],
                recommended_action=self._get_violation_recommendation(violation),
                source="dipmaster_strategy_monitor",
                tags={
                    'rule': violation.rule.value,
                    'strategy': 'dipmaster',
                    'symbol': violation.symbol or 'unknown'
                }
            )
    
    def _get_violation_recommendation(self, violation: StrategyViolation) -> str:
        """Get recommendation for violation type."""
        recommendations = {
            DipMasterRule.RSI_RANGE_CHECK: "Review RSI thresholds and market conditions",
            DipMasterRule.DIP_CONFIRMATION: "Verify dip detection logic and price feeds",
            DipMasterRule.VOLUME_CONFIRMATION: "Check volume calculation and thresholds",
            DipMasterRule.MA20_POSITION_CHECK: "Review MA20 calculation and position logic",
            DipMasterRule.BOUNDARY_EXIT_TIMING: "Check timing manager and exit logic",
            DipMasterRule.MAX_HOLDING_TIME: "Review position management and exit triggers",
            DipMasterRule.SIGNAL_CONFIDENCE_CHECK: "Calibrate signal confidence thresholds",
            DipMasterRule.TARGET_PROFIT_CHECK: "Review profit targets and exit strategy"
        }
        return recommendations.get(violation.rule, "Review strategy parameters and implementation")
    
    def _update_rule_scores(self):
        """Update rule compliance scores."""
        for rule in DipMasterRule:
            checks = self.rule_checks[rule]
            violations = self.rule_violations[rule]
            
            if checks > 0:
                compliance_rate = (checks - violations) / checks
                self.rule_scores[rule] = compliance_rate * 100
            else:
                self.rule_scores[rule] = 100.0
    
    async def generate_health_report(self) -> StrategyHealthReport:
        """Generate comprehensive strategy health report."""
        current_time = datetime.now(timezone.utc)
        
        # Calculate overall compliance rate
        total_checks = sum(self.rule_checks.values())
        total_violations = sum(self.rule_violations.values())
        compliance_rate = (total_checks - total_violations) / max(total_checks, 1) * 100
        
        # Calculate weighted health score
        rule_weights = {
            DipMasterRule.RSI_RANGE_CHECK: 0.15,
            DipMasterRule.DIP_CONFIRMATION: 0.15,
            DipMasterRule.VOLUME_CONFIRMATION: 0.15,
            DipMasterRule.MA20_POSITION_CHECK: 0.20,
            DipMasterRule.BOUNDARY_EXIT_TIMING: 0.20,
            DipMasterRule.MAX_HOLDING_TIME: 0.10,
            DipMasterRule.SIGNAL_CONFIDENCE_CHECK: 0.05
        }
        
        weighted_score = sum(
            self.rule_scores[rule] * weight 
            for rule, weight in rule_weights.items()
        )
        
        # Determine health level
        if weighted_score >= 90:
            health_level = StrategyHealthLevel.EXCELLENT
        elif weighted_score >= 75:
            health_level = StrategyHealthLevel.GOOD
        elif weighted_score >= 60:
            health_level = StrategyHealthLevel.FAIR
        elif weighted_score >= 40:
            health_level = StrategyHealthLevel.POOR
        else:
            health_level = StrategyHealthLevel.CRITICAL
        
        # Get recent violations
        recent_violations = [
            v for v in list(self.violations)[-20:]  # Last 20 violations
            if (current_time - v.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(weighted_score, recent_violations)
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        report = StrategyHealthReport(
            timestamp=current_time,
            overall_score=weighted_score,
            health_level=health_level,
            compliance_rate=compliance_rate,
            rule_scores=dict(self.rule_scores),
            violations=recent_violations,
            recommendations=recommendations,
            performance_metrics=performance_metrics
        )
        
        # Store in history
        self.health_history.append({
            'timestamp': current_time.timestamp(),
            'score': weighted_score,
            'compliance_rate': compliance_rate
        })
        
        return report
    
    def _generate_recommendations(self,
                                health_score: float,
                                recent_violations: List[StrategyViolation]) -> List[str]:
        """Generate recommendations based on health score and violations."""
        recommendations = []
        
        if health_score < 60:
            recommendations.append("URGENT: Strategy health is poor - comprehensive review required")
        
        # Rule-specific recommendations
        violation_counts = defaultdict(int)
        for violation in recent_violations:
            violation_counts[violation.rule] += 1
        
        for rule, count in violation_counts.items():
            if count >= 3:  # 3+ violations of same rule
                recommendations.append(f"Review {rule.value} - {count} recent violations detected")
        
        # Boundary compliance specific
        if DipMasterRule.BOUNDARY_EXIT_TIMING in violation_counts:
            recommendations.append("Check timing manager - boundary exit compliance issues detected")
        
        # RSI and MA20 issues
        if (DipMasterRule.RSI_RANGE_CHECK in violation_counts or 
            DipMasterRule.MA20_POSITION_CHECK in violation_counts):
            recommendations.append("Review market conditions - signal quality may be degraded")
        
        # General recommendations
        if health_score >= 90:
            recommendations.append("Strategy operating optimally - maintain current parameters")
        elif health_score >= 75:
            recommendations.append("Strategy performing well - minor optimizations possible")
        else:
            recommendations.append("Consider parameter recalibration and strategy review")
        
        return recommendations
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate strategy performance metrics."""
        metrics = {}
        
        # Boundary exit distribution
        total_exits = sum(self.boundary_exits.values())
        if total_exits > 0:
            for minute in self.params.boundary_minutes:
                pct = (self.boundary_exits[minute] / total_exits) * 100
                metrics[f'boundary_exit_{minute}min_pct'] = pct
        
        # Holding time statistics
        if self.holding_time_stats:
            holding_times = list(self.holding_time_stats)
            metrics['avg_holding_time'] = np.mean(holding_times)
            metrics['median_holding_time'] = np.median(holding_times)
            metrics['max_holding_time'] = np.max(holding_times)
            
            # Compliance rate for holding time
            compliant_count = sum(1 for ht in holding_times if ht <= self.params.max_holding_minutes)
            metrics['holding_time_compliance_rate'] = (compliant_count / len(holding_times)) * 100
        
        # Rule compliance rates
        for rule in DipMasterRule:
            checks = self.rule_checks[rule]
            violations = self.rule_violations[rule]
            if checks > 0:
                metrics[f'{rule.value}_compliance_rate'] = ((checks - violations) / checks) * 100
        
        return metrics
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            'signals_tracked': len(self.signals_tracked),
            'positions_tracked': len(self.positions_tracked),
            'total_violations': len(self.violations),
            'rule_checks': dict(self.rule_checks),
            'rule_violations': dict(self.rule_violations),
            'rule_scores': dict(self.rule_scores),
            'boundary_exits': dict(self.boundary_exits),
            'health_history_size': len(self.health_history)
        }


# Factory function
def create_dipmaster_monitor(event_producer: Optional[EnhancedEventProducer] = None,
                           structured_logger: Optional[StructuredLogger] = None,
                           config: Optional[Dict[str, Any]] = None) -> DipMasterStrategyMonitor:
    """Create and configure DipMaster strategy monitor."""
    return DipMasterStrategyMonitor(event_producer, structured_logger, config)