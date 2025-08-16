#!/usr/bin/env python3
"""
Performance Monitor for DipMaster Trading System
性能监控器 - 实时交易性能跟踪和分析

Features:
- Real-time win rate tracking
- Sharpe ratio monitoring
- Maximum drawdown tracking
- P&L ratio monitoring
- Execution quality assessment
- Performance attribution analysis
- Benchmark comparison
- Performance alerts and reporting
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


class PerformanceLevel(Enum):
    """Performance level classification."""
    EXCELLENT = "excellent"    # Top 10% performance
    GOOD = "good"             # Above average
    AVERAGE = "average"       # Market performance
    POOR = "poor"            # Below average
    CRITICAL = "critical"     # Significant underperformance


class TradeOutcome(Enum):
    """Trade outcome classification."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


@dataclass
class TradeRecord:
    """Individual trade record."""
    trade_id: str
    timestamp: float
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: float
    exit_time: float
    duration_seconds: float
    return_pct: float
    return_usd: float
    fees: float
    slippage_bps: float
    outcome: TradeOutcome
    strategy: str
    confidence: float = 0.0
    exit_reason: str = "unknown"
    market_condition: str = "normal"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    timestamp: float
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    
    # Return metrics
    total_return_pct: float = 0.0
    total_return_usd: float = 0.0
    avg_return_per_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # Risk-adjusted metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Drawdown metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    recovery_factor: float = 0.0
    
    # Trade timing
    avg_trade_duration: float = 0.0
    avg_time_between_trades: float = 0.0
    
    # Execution quality
    avg_slippage_bps: float = 0.0
    avg_fees_per_trade: float = 0.0
    execution_cost_pct: float = 0.0
    
    # Volatility and consistency
    return_volatility: float = 0.0
    return_skewness: float = 0.0
    return_kurtosis: float = 0.0
    consistency_ratio: float = 0.0


@dataclass
class PerformanceAlert:
    """Performance-related alert."""
    alert_id: str
    timestamp: float
    alert_type: str
    severity: str
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    recommendation: str


class PerformanceBenchmark:
    """Performance benchmark for comparison."""
    
    def __init__(self, name: str, target_metrics: Dict[str, float]):
        self.name = name
        self.target_metrics = target_metrics
        self.historical_performance = deque(maxlen=1000)
    
    def update_performance(self, metrics: PerformanceMetrics):
        """Update benchmark with current performance."""
        self.historical_performance.append({
            'timestamp': metrics.timestamp,
            'win_rate': metrics.win_rate,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'return_pct': metrics.total_return_pct
        })
    
    def compare_performance(self, current_metrics: PerformanceMetrics) -> Dict[str, float]:
        """Compare current performance to benchmark targets."""
        comparisons = {}
        
        for metric_name, target_value in self.target_metrics.items():
            current_value = getattr(current_metrics, metric_name, 0)
            
            if target_value != 0:
                performance_ratio = current_value / target_value
                comparisons[metric_name] = {
                    'current': current_value,
                    'target': target_value,
                    'ratio': performance_ratio,
                    'outperformance_pct': (performance_ratio - 1) * 100
                }
        
        return comparisons


class DrawdownAnalyzer:
    """Analyze drawdown patterns and recovery."""
    
    def __init__(self):
        self.equity_curve = deque(maxlen=10000)
        self.drawdown_periods = []
        self.current_drawdown_start = None
        self.peak_equity = 0.0
    
    def update_equity(self, equity_value: float, timestamp: float):
        """Update equity curve and analyze drawdowns."""
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity_value
        })
        
        # Update peak equity
        if equity_value > self.peak_equity:
            # New peak - end any current drawdown
            if self.current_drawdown_start is not None:
                drawdown_period = {
                    'start_time': self.current_drawdown_start['timestamp'],
                    'end_time': timestamp,
                    'start_equity': self.current_drawdown_start['equity'],
                    'trough_equity': min(entry['equity'] for entry in self.equity_curve 
                                       if entry['timestamp'] >= self.current_drawdown_start['timestamp']),
                    'recovery_equity': equity_value,
                    'duration_hours': (timestamp - self.current_drawdown_start['timestamp']) / 3600,
                    'drawdown_pct': (self.peak_equity - min(entry['equity'] for entry in self.equity_curve 
                                                           if entry['timestamp'] >= self.current_drawdown_start['timestamp'])) / self.peak_equity
                }
                self.drawdown_periods.append(drawdown_period)
                self.current_drawdown_start = None
            
            self.peak_equity = equity_value
        
        elif equity_value < self.peak_equity:
            # In drawdown
            if self.current_drawdown_start is None:
                # Start of new drawdown period
                self.current_drawdown_start = {
                    'timestamp': timestamp,
                    'equity': self.peak_equity
                }
    
    def get_current_drawdown(self) -> float:
        """Get current drawdown percentage."""
        if not self.equity_curve:
            return 0.0
        
        current_equity = self.equity_curve[-1]['equity']
        return (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
    
    def get_drawdown_statistics(self) -> Dict[str, Any]:
        """Get comprehensive drawdown statistics."""
        if not self.drawdown_periods:
            return {
                'avg_drawdown': 0.0,
                'max_drawdown': self.get_current_drawdown(),
                'avg_recovery_time': 0.0,
                'drawdown_frequency': 0.0,
                'recovery_factor': 0.0
            }
        
        drawdowns = [dd['drawdown_pct'] for dd in self.drawdown_periods]
        recovery_times = [dd['duration_hours'] for dd in self.drawdown_periods]
        
        return {
            'avg_drawdown': statistics.mean(drawdowns),
            'max_drawdown': max(drawdowns + [self.get_current_drawdown()]),
            'avg_recovery_time': statistics.mean(recovery_times),
            'drawdown_frequency': len(self.drawdown_periods) / max(len(self.equity_curve) / 100, 1),  # Per 100 data points
            'recovery_factor': self.peak_equity / max(drawdowns) if drawdowns and max(drawdowns) > 0 else 0.0,
            'total_drawdown_periods': len(self.drawdown_periods),
            'current_drawdown': self.get_current_drawdown()
        }


class PerformanceMonitor:
    """
    Comprehensive trading performance monitoring system.
    
    Tracks and analyzes all aspects of trading performance with real-time
    alerts, benchmarking, and detailed attribution analysis.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 benchmark_targets: Optional[Dict[str, float]] = None):
        """
        Initialize performance monitor.
        
        Args:
            config: Configuration parameters
            benchmark_targets: Target performance metrics for benchmarking
        """
        self.config = config or {}
        
        # Trade tracking
        self.trades = {}  # trade_id -> TradeRecord
        self.trade_history = deque(maxlen=10000)
        
        # Performance metrics
        self.current_metrics = PerformanceMetrics(timestamp=time.time())
        self.metrics_history = deque(maxlen=1000)
        
        # Drawdown analysis
        self.drawdown_analyzer = DrawdownAnalyzer()
        
        # Performance alerts
        self.performance_alerts = {}
        self.alert_thresholds = self._setup_alert_thresholds()
        
        # Benchmarking
        default_targets = {
            'win_rate': 0.55,      # 55% win rate target
            'sharpe_ratio': 1.5,   # 1.5 Sharpe ratio target
            'max_drawdown': 0.15,  # 15% max drawdown limit
            'profit_factor': 1.5,  # 1.5 profit factor target
            'return_volatility': 0.20  # 20% volatility target
        }
        
        benchmark_targets = benchmark_targets or default_targets
        self.benchmark = PerformanceBenchmark("DipMaster_Target", benchmark_targets)
        
        # Performance attribution
        self.symbol_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0.0, 'wins': 0})
        self.strategy_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0.0, 'wins': 0})
        self.hourly_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0.0, 'wins': 0})
        
        # Equity tracking
        self.initial_equity = self.config.get('initial_equity', 100000)
        self.current_equity = self.initial_equity
        
        logger.info("📊 PerformanceMonitor initialized")
    
    def _setup_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Setup performance alert thresholds."""
        return {
            'win_rate': {
                'critical': 0.30,  # Below 30%
                'warning': 0.45,   # Below 45%
                'target': 0.55     # Target 55%
            },
            'sharpe_ratio': {
                'critical': 0.5,   # Below 0.5
                'warning': 1.0,    # Below 1.0
                'target': 1.5      # Target 1.5
            },
            'max_drawdown': {
                'warning': 0.10,   # Above 10%
                'critical': 0.20,  # Above 20%
                'emergency': 0.30  # Above 30%
            },
            'profit_factor': {
                'critical': 0.8,   # Below 0.8
                'warning': 1.0,    # Below 1.0
                'target': 1.5      # Target 1.5
            }
        }
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a completed trade."""
        try:
            # Create trade record
            return_pct = ((trade_data['exit_price'] - trade_data['entry_price']) / trade_data['entry_price']) * 100
            if trade_data.get('side', 'buy').lower() == 'sell':
                return_pct = -return_pct
            
            return_usd = return_pct / 100 * trade_data['entry_price'] * trade_data['quantity']
            return_usd -= trade_data.get('fees', 0)
            
            # Determine outcome
            if return_pct > 0.1:  # > 0.1%
                outcome = TradeOutcome.WIN
            elif return_pct < -0.1:  # < -0.1%
                outcome = TradeOutcome.LOSS
            else:
                outcome = TradeOutcome.BREAKEVEN
            
            trade_record = TradeRecord(
                trade_id=trade_data['trade_id'],
                timestamp=trade_data.get('timestamp', time.time()),
                symbol=trade_data['symbol'],
                side=trade_data.get('side', 'buy'),
                entry_price=trade_data['entry_price'],
                exit_price=trade_data['exit_price'],
                quantity=trade_data['quantity'],
                entry_time=trade_data.get('entry_time', trade_data.get('timestamp', time.time())),
                exit_time=trade_data.get('exit_time', time.time()),
                duration_seconds=trade_data.get('duration_seconds', 0),
                return_pct=return_pct,
                return_usd=return_usd,
                fees=trade_data.get('fees', 0),
                slippage_bps=trade_data.get('slippage_bps', 0),
                outcome=outcome,
                strategy=trade_data.get('strategy', 'dipmaster'),
                confidence=trade_data.get('confidence', 0.0),
                exit_reason=trade_data.get('exit_reason', 'unknown'),
                market_condition=trade_data.get('market_condition', 'normal')
            )
            
            # Store trade
            self.trades[trade_record.trade_id] = trade_record
            self.trade_history.append(trade_record)
            
            # Update equity
            self.current_equity += return_usd
            self.drawdown_analyzer.update_equity(self.current_equity, trade_record.timestamp)
            
            # Update performance attribution
            self._update_attribution(trade_record)
            
            # Recalculate metrics
            self.calculate_performance_metrics()
            
            logger.debug(f"📈 Recorded trade: {trade_record.symbol} {return_pct:.2f}% "
                        f"({trade_record.outcome.value})")
            
        except Exception as e:
            logger.error(f"❌ Failed to record trade: {e}")
    
    def _update_attribution(self, trade: TradeRecord):
        """Update performance attribution by symbol, strategy, and time."""
        # Symbol attribution
        self.symbol_performance[trade.symbol]['trades'] += 1
        self.symbol_performance[trade.symbol]['pnl'] += trade.return_usd
        if trade.outcome == TradeOutcome.WIN:
            self.symbol_performance[trade.symbol]['wins'] += 1
        
        # Strategy attribution
        self.strategy_performance[trade.strategy]['trades'] += 1
        self.strategy_performance[trade.strategy]['pnl'] += trade.return_usd
        if trade.outcome == TradeOutcome.WIN:
            self.strategy_performance[trade.strategy]['wins'] += 1
        
        # Hourly attribution
        hour = int(trade.timestamp % 86400 // 3600)  # Hour of day
        self.hourly_performance[hour]['trades'] += 1
        self.hourly_performance[hour]['pnl'] += trade.return_usd
        if trade.outcome == TradeOutcome.WIN:
            self.hourly_performance[hour]['wins'] += 1
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        try:
            current_time = time.time()
            
            if not self.trade_history:
                self.current_metrics = PerformanceMetrics(timestamp=current_time)
                return self.current_metrics
            
            trades = list(self.trade_history)
            
            # Basic trade statistics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.outcome == TradeOutcome.WIN])
            losing_trades = len([t for t in trades if t.outcome == TradeOutcome.LOSS])
            breakeven_trades = len([t for t in trades if t.outcome == TradeOutcome.BREAKEVEN])
            
            # Return calculations
            returns = [t.return_pct for t in trades]
            usd_returns = [t.return_usd for t in trades]
            
            total_return_usd = sum(usd_returns)
            total_return_pct = (self.current_equity - self.initial_equity) / self.initial_equity * 100
            avg_return_per_trade = statistics.mean(returns) if returns else 0.0
            
            # Win/Loss analysis
            wins = [t.return_pct for t in trades if t.outcome == TradeOutcome.WIN]
            losses = [t.return_pct for t in trades if t.outcome == TradeOutcome.LOSS]
            
            avg_win = statistics.mean(wins) if wins else 0.0
            avg_loss = statistics.mean(losses) if losses else 0.0
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # Profit factor
            gross_profit = sum(wins) if wins else 0.0
            gross_loss = abs(sum(losses)) if losses else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
            
            # Risk-adjusted metrics
            return_volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0
            sharpe_ratio = avg_return_per_trade / return_volatility if return_volatility > 0 else 0.0
            
            # Sortino ratio (downside deviation)
            downside_returns = [r for r in returns if r < 0]
            downside_deviation = statistics.stdev(downside_returns) if len(downside_returns) > 1 else 0.0
            sortino_ratio = avg_return_per_trade / downside_deviation if downside_deviation > 0 else 0.0
            
            # Drawdown metrics
            drawdown_stats = self.drawdown_analyzer.get_drawdown_statistics()
            current_drawdown = drawdown_stats['current_drawdown']
            max_drawdown = drawdown_stats['max_drawdown']
            
            # Calmar ratio
            calmar_ratio = total_return_pct / (max_drawdown * 100) if max_drawdown > 0 else 0.0
            
            # Trade timing
            durations = [t.duration_seconds for t in trades if t.duration_seconds > 0]
            avg_trade_duration = statistics.mean(durations) if durations else 0.0
            
            # Time between trades
            if len(trades) > 1:
                time_diffs = [trades[i].timestamp - trades[i-1].timestamp for i in range(1, len(trades))]
                avg_time_between_trades = statistics.mean(time_diffs)
            else:
                avg_time_between_trades = 0.0
            
            # Execution quality
            slippages = [t.slippage_bps for t in trades if t.slippage_bps > 0]
            fees = [t.fees for t in trades if t.fees > 0]
            
            avg_slippage_bps = statistics.mean(slippages) if slippages else 0.0
            avg_fees_per_trade = statistics.mean(fees) if fees else 0.0
            execution_cost_pct = (sum(fees) / abs(total_return_usd)) * 100 if total_return_usd != 0 else 0.0
            
            # Advanced statistics
            return_skewness = self._calculate_skewness(returns) if len(returns) > 2 else 0.0
            return_kurtosis = self._calculate_kurtosis(returns) if len(returns) > 3 else 0.0
            
            # Consistency ratio (percentage of profitable periods)
            if len(returns) >= 10:
                periods = [returns[i:i+10] for i in range(0, len(returns)-9, 10)]
                profitable_periods = sum(1 for period in periods if sum(period) > 0)
                consistency_ratio = profitable_periods / len(periods) if periods else 0.0
            else:
                consistency_ratio = 0.0
            
            # Update metrics
            self.current_metrics = PerformanceMetrics(
                timestamp=current_time,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                breakeven_trades=breakeven_trades,
                total_return_pct=total_return_pct,
                total_return_usd=total_return_usd,
                avg_return_per_trade=avg_return_per_trade,
                avg_win=avg_win,
                avg_loss=avg_loss,
                win_rate=win_rate,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                avg_drawdown=drawdown_stats['avg_drawdown'],
                recovery_factor=drawdown_stats['recovery_factor'],
                avg_trade_duration=avg_trade_duration,
                avg_time_between_trades=avg_time_between_trades,
                avg_slippage_bps=avg_slippage_bps,
                avg_fees_per_trade=avg_fees_per_trade,
                execution_cost_pct=execution_cost_pct,
                return_volatility=return_volatility,
                return_skewness=return_skewness,
                return_kurtosis=return_kurtosis,
                consistency_ratio=consistency_ratio
            )
            
            # Store in history
            self.metrics_history.append(self.current_metrics)
            
            # Update benchmark
            self.benchmark.update_performance(self.current_metrics)
            
            # Check for alerts
            self._check_performance_alerts()
            
            logger.debug(f"📊 Updated metrics: WR={win_rate:.1%}, SR={sharpe_ratio:.2f}, "
                        f"DD={current_drawdown:.1%}")
            
            return self.current_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate performance metrics: {e}")
            return self.current_metrics
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of returns."""
        try:
            if len(data) < 3:
                return 0.0
            
            mean = statistics.mean(data)
            std = statistics.stdev(data)
            
            if std == 0:
                return 0.0
            
            skewness = sum(((x - mean) / std) ** 3 for x in data) / len(data)
            return skewness
            
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of returns."""
        try:
            if len(data) < 4:
                return 0.0
            
            mean = statistics.mean(data)
            std = statistics.stdev(data)
            
            if std == 0:
                return 0.0
            
            kurtosis = sum(((x - mean) / std) ** 4 for x in data) / len(data) - 3
            return kurtosis
            
        except Exception:
            return 0.0
    
    def _check_performance_alerts(self):
        """Check for performance-based alerts."""
        try:
            alerts_generated = []
            
            # Win rate alerts
            if self.current_metrics.win_rate < self.alert_thresholds['win_rate']['critical']:
                alert = self._create_performance_alert(
                    'win_rate_critical',
                    'critical',
                    'win_rate',
                    self.current_metrics.win_rate,
                    self.alert_thresholds['win_rate']['critical'],
                    f"Critical win rate: {self.current_metrics.win_rate:.1%}",
                    "Review strategy signals and market conditions"
                )
                alerts_generated.append(alert)
            
            elif self.current_metrics.win_rate < self.alert_thresholds['win_rate']['warning']:
                alert = self._create_performance_alert(
                    'win_rate_warning',
                    'warning',
                    'win_rate',
                    self.current_metrics.win_rate,
                    self.alert_thresholds['win_rate']['warning'],
                    f"Low win rate: {self.current_metrics.win_rate:.1%}",
                    "Monitor signal quality and consider parameter adjustments"
                )
                alerts_generated.append(alert)
            
            # Sharpe ratio alerts
            if self.current_metrics.sharpe_ratio < self.alert_thresholds['sharpe_ratio']['critical']:
                alert = self._create_performance_alert(
                    'sharpe_critical',
                    'critical',
                    'sharpe_ratio',
                    self.current_metrics.sharpe_ratio,
                    self.alert_thresholds['sharpe_ratio']['critical'],
                    f"Critical Sharpe ratio: {self.current_metrics.sharpe_ratio:.2f}",
                    "Review risk management and volatility control"
                )
                alerts_generated.append(alert)
            
            # Drawdown alerts
            if self.current_metrics.current_drawdown > self.alert_thresholds['max_drawdown']['emergency']:
                alert = self._create_performance_alert(
                    'drawdown_emergency',
                    'emergency',
                    'current_drawdown',
                    self.current_metrics.current_drawdown,
                    self.alert_thresholds['max_drawdown']['emergency'],
                    f"Emergency drawdown: {self.current_metrics.current_drawdown:.1%}",
                    "Consider halting trading immediately"
                )
                alerts_generated.append(alert)
            
            elif self.current_metrics.current_drawdown > self.alert_thresholds['max_drawdown']['critical']:
                alert = self._create_performance_alert(
                    'drawdown_critical',
                    'critical',
                    'current_drawdown',
                    self.current_metrics.current_drawdown,
                    self.alert_thresholds['max_drawdown']['critical'],
                    f"Critical drawdown: {self.current_metrics.current_drawdown:.1%}",
                    "Reduce position sizes and review strategy"
                )
                alerts_generated.append(alert)
            
            # Profit factor alerts
            if self.current_metrics.profit_factor < self.alert_thresholds['profit_factor']['critical']:
                alert = self._create_performance_alert(
                    'profit_factor_critical',
                    'critical',
                    'profit_factor',
                    self.current_metrics.profit_factor,
                    self.alert_thresholds['profit_factor']['critical'],
                    f"Critical profit factor: {self.current_metrics.profit_factor:.2f}",
                    "Review exit strategies and risk management"
                )
                alerts_generated.append(alert)
            
            # Log alerts
            for alert in alerts_generated:
                if alert.severity == 'critical':
                    logger.critical(f"🚨 {alert.message}: {alert.recommendation}")
                elif alert.severity == 'emergency':
                    logger.critical(f"🚨🚨 EMERGENCY: {alert.message}: {alert.recommendation}")
                else:
                    logger.warning(f"⚠️ {alert.message}: {alert.recommendation}")
            
        except Exception as e:
            logger.error(f"❌ Error checking performance alerts: {e}")
    
    def _create_performance_alert(self,
                                alert_type: str,
                                severity: str,
                                metric_name: str,
                                current_value: float,
                                threshold_value: float,
                                message: str,
                                recommendation: str) -> PerformanceAlert:
        """Create a performance alert."""
        alert_id = f"{alert_type}_{int(time.time())}"
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            timestamp=time.time(),
            alert_type=alert_type,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            recommendation=recommendation
        )
        
        self.performance_alerts[alert_id] = alert
        return alert
    
    def get_performance_level(self) -> PerformanceLevel:
        """Classify current performance level."""
        try:
            # Score based on multiple metrics
            score = 0
            
            # Win rate scoring
            if self.current_metrics.win_rate >= 0.65:
                score += 25
            elif self.current_metrics.win_rate >= 0.55:
                score += 20
            elif self.current_metrics.win_rate >= 0.45:
                score += 10
            elif self.current_metrics.win_rate >= 0.35:
                score += 5
            
            # Sharpe ratio scoring
            if self.current_metrics.sharpe_ratio >= 2.0:
                score += 25
            elif self.current_metrics.sharpe_ratio >= 1.5:
                score += 20
            elif self.current_metrics.sharpe_ratio >= 1.0:
                score += 10
            elif self.current_metrics.sharpe_ratio >= 0.5:
                score += 5
            
            # Drawdown scoring (inverted - lower is better)
            if self.current_metrics.current_drawdown <= 0.05:
                score += 25
            elif self.current_metrics.current_drawdown <= 0.10:
                score += 20
            elif self.current_metrics.current_drawdown <= 0.15:
                score += 10
            elif self.current_metrics.current_drawdown <= 0.25:
                score += 5
            
            # Profit factor scoring
            if self.current_metrics.profit_factor >= 2.0:
                score += 25
            elif self.current_metrics.profit_factor >= 1.5:
                score += 20
            elif self.current_metrics.profit_factor >= 1.2:
                score += 10
            elif self.current_metrics.profit_factor >= 1.0:
                score += 5
            
            # Classify based on total score
            if score >= 80:
                return PerformanceLevel.EXCELLENT
            elif score >= 60:
                return PerformanceLevel.GOOD
            elif score >= 40:
                return PerformanceLevel.AVERAGE
            elif score >= 20:
                return PerformanceLevel.POOR
            else:
                return PerformanceLevel.CRITICAL
                
        except Exception as e:
            logger.error(f"❌ Error determining performance level: {e}")
            return PerformanceLevel.AVERAGE
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            benchmark_comparison = self.benchmark.compare_performance(self.current_metrics)
            performance_level = self.get_performance_level()
            active_alerts = [alert for alert in self.performance_alerts.values() 
                           if time.time() - alert.timestamp < 24 * 3600]  # Last 24 hours
            
            return {
                'timestamp': time.time(),
                'performance_level': performance_level.value,
                'current_metrics': {
                    'total_trades': self.current_metrics.total_trades,
                    'win_rate': self.current_metrics.win_rate,
                    'profit_factor': self.current_metrics.profit_factor,
                    'sharpe_ratio': self.current_metrics.sharpe_ratio,
                    'total_return_pct': self.current_metrics.total_return_pct,
                    'total_return_usd': self.current_metrics.total_return_usd,
                    'current_drawdown': self.current_metrics.current_drawdown,
                    'max_drawdown': self.current_metrics.max_drawdown,
                    'avg_return_per_trade': self.current_metrics.avg_return_per_trade,
                    'return_volatility': self.current_metrics.return_volatility,
                    'consistency_ratio': self.current_metrics.consistency_ratio
                },
                'benchmark_comparison': benchmark_comparison,
                'attribution': {
                    'by_symbol': dict(self.symbol_performance),
                    'by_strategy': dict(self.strategy_performance),
                    'by_hour': dict(self.hourly_performance)
                },
                'execution_quality': {
                    'avg_slippage_bps': self.current_metrics.avg_slippage_bps,
                    'avg_fees_per_trade': self.current_metrics.avg_fees_per_trade,
                    'execution_cost_pct': self.current_metrics.execution_cost_pct
                },
                'risk_metrics': {
                    'current_drawdown': self.current_metrics.current_drawdown,
                    'max_drawdown': self.current_metrics.max_drawdown,
                    'volatility': self.current_metrics.return_volatility,
                    'skewness': self.current_metrics.return_skewness,
                    'kurtosis': self.current_metrics.return_kurtosis
                },
                'alerts': {
                    'active_count': len(active_alerts),
                    'by_severity': {
                        'critical': len([a for a in active_alerts if a.severity == 'critical']),
                        'warning': len([a for a in active_alerts if a.severity == 'warning']),
                        'emergency': len([a for a in active_alerts if a.severity == 'emergency'])
                    }
                },
                'equity': {
                    'current': self.current_equity,
                    'initial': self.initial_equity,
                    'peak': self.drawdown_analyzer.peak_equity
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get performance summary: {e}")
            return {'error': str(e)}
    
    def export_performance_report(self) -> Dict[str, Any]:
        """Export comprehensive performance report."""
        try:
            return {
                'timestamp': time.time(),
                'report_type': 'trading_performance',
                'summary': self.get_performance_summary(),
                'detailed_metrics': self.current_metrics.__dict__,
                'trade_statistics': {
                    'total_trades': len(self.trade_history),
                    'recent_trades': [t.__dict__ for t in list(self.trade_history)[-10:]],  # Last 10 trades
                    'win_loss_breakdown': {
                        'wins': self.current_metrics.winning_trades,
                        'losses': self.current_metrics.losing_trades,
                        'breakevens': self.current_metrics.breakeven_trades
                    }
                },
                'drawdown_analysis': self.drawdown_analyzer.get_drawdown_statistics(),
                'performance_attribution': {
                    'symbol_performance': dict(self.symbol_performance),
                    'strategy_performance': dict(self.strategy_performance),
                    'hourly_performance': dict(self.hourly_performance)
                },
                'benchmark_analysis': self.benchmark.compare_performance(self.current_metrics),
                'active_alerts': [alert.__dict__ for alert in self.performance_alerts.values() 
                                if time.time() - alert.timestamp < 24 * 3600],
                'recommendations': self._generate_performance_recommendations()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to export performance report: {e}")
            return {'error': str(e)}
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        try:
            # Win rate recommendations
            if self.current_metrics.win_rate < 0.50:
                recommendations.append("Win rate below 50%: Review signal quality and entry criteria")
            
            # Sharpe ratio recommendations
            if self.current_metrics.sharpe_ratio < 1.0:
                recommendations.append("Low Sharpe ratio: Improve risk-adjusted returns through better risk management")
            
            # Drawdown recommendations
            if self.current_metrics.current_drawdown > 0.15:
                recommendations.append("High drawdown: Consider reducing position sizes and implementing stricter stop losses")
            
            # Profit factor recommendations
            if self.current_metrics.profit_factor < 1.2:
                recommendations.append("Low profit factor: Review exit strategies and profit-taking mechanisms")
            
            # Execution cost recommendations
            if self.current_metrics.execution_cost_pct > 5.0:
                recommendations.append("High execution costs: Optimize order routing and reduce trading frequency")
            
            # Consistency recommendations
            if self.current_metrics.consistency_ratio < 0.6:
                recommendations.append("Low consistency: Focus on more reliable trading patterns and risk control")
            
            # Volatility recommendations
            if self.current_metrics.return_volatility > 0.30:
                recommendations.append("High volatility: Implement better position sizing and volatility filters")
            
            if not recommendations:
                recommendations.append("Performance within acceptable parameters: Continue current strategy")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            return ["Error generating recommendations"]


# Factory function
def create_performance_monitor(config: Dict[str, Any]) -> PerformanceMonitor:
    """Create and configure performance monitor."""
    benchmark_targets = config.get('benchmark_targets', {
        'win_rate': 0.55,
        'sharpe_ratio': 1.5,
        'max_drawdown': 0.15,
        'profit_factor': 1.5
    })
    
    return PerformanceMonitor(
        config=config.get('performance_config', {}),
        benchmark_targets=benchmark_targets
    )