#!/usr/bin/env python3
"""
Business KPI Tracker for DipMaster Trading System
业务KPI跟踪器 - 专门监控交易业务关键指标

Features:
- Trading performance metrics (win rate, PnL, drawdown)
- Strategy effectiveness tracking
- Risk management metrics
- Position and order analytics
- Real-time KPI calculation and alerting
- Historical performance analysis
"""

import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics
import logging

from .metrics_collector import MetricsCollector, get_metrics_collector

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Individual trade record for KPI calculation."""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float = 0.0
    entry_time: float = field(default_factory=time.time)
    exit_time: Optional[float] = None
    pnl: Optional[float] = None
    commission: float = 0.0
    duration_seconds: Optional[int] = None
    strategy: str = "dipmaster"
    tags: Dict[str, str] = field(default_factory=dict)
    
    def is_completed(self) -> bool:
        """Check if trade is completed."""
        return self.exit_price is not None and self.exit_time is not None
    
    def calculate_pnl(self) -> float:
        """Calculate PnL for the trade."""
        if not self.is_completed():
            return 0.0
        
        if self.side.lower() == 'buy':
            pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # sell
            pnl = (self.entry_price - self.exit_price) * self.quantity
        
        return pnl - self.commission
    
    def calculate_return_percentage(self) -> float:
        """Calculate return as percentage."""
        if not self.is_completed():
            return 0.0
        
        investment = self.entry_price * self.quantity
        if investment == 0:
            return 0.0
        
        return (self.calculate_pnl() / investment) * 100


@dataclass  
class PositionInfo:
    """Current position information."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    duration_seconds: int = 0
    strategy: str = "dipmaster"
    
    def update_current_price(self, price: float):
        """Update current price and unrealized PnL."""
        self.current_price = price
        
        if self.side.lower() == 'buy':
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity


class BusinessKPITracker:
    """
    Business KPI tracking system for trading performance.
    
    Tracks key trading metrics including win rate, PnL, drawdown,
    strategy effectiveness, and risk management KPIs.
    """
    
    def __init__(self,
                 metrics_collector: Optional[MetricsCollector] = None,
                 history_retention_hours: int = 720):  # 30 days
        """
        Initialize business KPI tracker.
        
        Args:
            metrics_collector: Metrics collector instance
            history_retention_hours: Hours to retain historical data
        """
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.history_retention_hours = history_retention_hours
        
        # Thread-safe storage
        self._lock = threading.RLock()
        
        # Trading data
        self._completed_trades: deque = deque()
        self._active_positions: Dict[str, PositionInfo] = {}
        self._daily_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Performance tracking
        self._equity_curve: deque = deque()
        self._drawdown_tracker: List[float] = []
        self._peak_equity = 0.0
        
        # Strategy statistics
        self._strategy_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'total_volume': 0.0
        })
        
        # KPI calculation intervals
        self._last_kpi_calculation = 0
        self._kpi_calculation_interval = 60  # seconds
        
        # Register core metrics
        self._register_core_metrics()
        
        # Start background KPI calculation
        self._start_kpi_thread()
        
        logger.info("📈 BusinessKPITracker initialized")
    
    def _register_core_metrics(self):
        """Register core business KPI metrics."""
        from .metrics_collector import MetricType
        
        core_metrics = [
            # Trading Performance
            ('trading.win_rate', MetricType.GAUGE, 'Trading win rate percentage', '%'),
            ('trading.total_pnl', MetricType.GAUGE, 'Total profit and loss', 'USD'),
            ('trading.daily_pnl', MetricType.GAUGE, 'Daily profit and loss', 'USD'),
            ('trading.total_trades', MetricType.COUNTER, 'Total number of trades', 'count'),
            ('trading.winning_trades', MetricType.COUNTER, 'Number of winning trades', 'count'),
            ('trading.losing_trades', MetricType.COUNTER, 'Number of losing trades', 'count'),
            
            # Risk Management
            ('risk.max_drawdown', MetricType.GAUGE, 'Maximum drawdown percentage', '%'),
            ('risk.current_drawdown', MetricType.GAUGE, 'Current drawdown percentage', '%'),
            ('risk.active_positions', MetricType.GAUGE, 'Number of active positions', 'count'),
            ('risk.total_exposure', MetricType.GAUGE, 'Total position exposure', 'USD'),
            ('risk.unrealized_pnl', MetricType.GAUGE, 'Unrealized profit/loss', 'USD'),
            
            # Strategy Performance  
            ('strategy.avg_trade_duration', MetricType.GAUGE, 'Average trade duration', 'minutes'),
            ('strategy.avg_win_amount', MetricType.GAUGE, 'Average winning trade amount', 'USD'),
            ('strategy.avg_loss_amount', MetricType.GAUGE, 'Average losing trade amount', 'USD'),
            ('strategy.profit_factor', MetricType.GAUGE, 'Profit factor ratio', 'ratio'),
            ('strategy.sharpe_ratio', MetricType.GAUGE, 'Risk-adjusted return ratio', 'ratio'),
            
            # System Performance
            ('system.orders_per_minute', MetricType.GAUGE, 'Orders executed per minute', 'count/min'),
            ('system.avg_order_execution_time', MetricType.GAUGE, 'Average order execution time', 'ms'),
            ('system.api_error_rate', MetricType.GAUGE, 'API error rate percentage', '%'),
            ('system.websocket_latency', MetricType.GAUGE, 'WebSocket data latency', 'ms')
        ]
        
        for name, metric_type, description, unit in core_metrics:
            self.metrics_collector.register_metric(name, metric_type, description, unit)
    
    def _start_kpi_thread(self):
        """Start background thread for KPI calculations."""
        def kpi_worker():
            while True:
                try:
                    time.sleep(self._kpi_calculation_interval)
                    self._calculate_all_kpis()
                except Exception as e:
                    logger.error(f"❌ KPI calculation error: {e}")
        
        thread = threading.Thread(target=kpi_worker, daemon=True)
        thread.start()
    
    def record_trade_entry(self,
                          trade_id: str,
                          symbol: str,
                          side: str,
                          price: float,
                          quantity: float,
                          strategy: str = "dipmaster",
                          tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record trade entry.
        
        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            side: 'buy' or 'sell'
            price: Entry price
            quantity: Trade quantity
            strategy: Strategy name
            tags: Additional tags
        """
        with self._lock:
            # Create position info
            position = PositionInfo(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                strategy=strategy
            )
            
            self._active_positions[trade_id] = position
            
            # Update metrics
            self.metrics_collector.increment_counter('trading.total_trades')
            self.metrics_collector.set_gauge('risk.active_positions', len(self._active_positions))
            
            # Calculate total exposure
            total_exposure = sum(pos.entry_price * pos.quantity 
                               for pos in self._active_positions.values())
            self.metrics_collector.set_gauge('risk.total_exposure', total_exposure)
            
            logger.debug(f"📈 Trade entry recorded: {trade_id} {symbol} {side} {quantity}@{price}")
    
    def record_trade_exit(self,
                         trade_id: str,
                         exit_price: float,
                         commission: float = 0.0,
                         tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """
        Record trade exit and calculate PnL.
        
        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            commission: Trading commission
            tags: Additional tags
            
        Returns:
            Trade PnL or None if trade not found
        """
        with self._lock:
            if trade_id not in self._active_positions:
                logger.warning(f"⚠️  Trade not found for exit: {trade_id}")
                return None
            
            position = self._active_positions.pop(trade_id)
            
            # Create completed trade record
            trade = TradeRecord(
                trade_id=trade_id,
                symbol=position.symbol,
                side=position.side,
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                entry_time=time.time() - position.duration_seconds,
                exit_time=time.time(),
                commission=commission,
                duration_seconds=position.duration_seconds,
                strategy=position.strategy,
                tags=tags or {}
            )
            
            # Calculate PnL
            pnl = trade.calculate_pnl()
            trade.pnl = pnl
            
            # Store completed trade
            self._completed_trades.append(trade)
            
            # Update strategy statistics
            self._update_strategy_stats(trade)
            
            # Update metrics
            is_winner = pnl > 0
            if is_winner:
                self.metrics_collector.increment_counter('trading.winning_trades')
            else:
                self.metrics_collector.increment_counter('trading.losing_trades')
            
            self.metrics_collector.set_gauge('risk.active_positions', len(self._active_positions))
            
            # Update equity curve
            self._update_equity_curve(pnl)
            
            logger.info(f"💰 Trade completed: {trade_id} PnL: ${pnl:.2f}")
            return pnl
    
    def update_position_price(self, trade_id: str, current_price: float):
        """Update current price for active position."""
        with self._lock:
            if trade_id in self._active_positions:
                position = self._active_positions[trade_id]
                position.update_current_price(current_price)
                position.duration_seconds = int(time.time() - (position.entry_price or time.time()))
    
    def _update_strategy_stats(self, trade: TradeRecord):
        """Update strategy-specific statistics."""
        strategy = trade.strategy
        stats = self._strategy_stats[strategy]
        
        stats['total_trades'] += 1
        if trade.pnl and trade.pnl > 0:
            stats['winning_trades'] += 1
        
        stats['total_pnl'] += trade.pnl or 0
        stats['total_volume'] += trade.entry_price * trade.quantity
    
    def _update_equity_curve(self, pnl: float):
        """Update equity curve and drawdown tracking."""
        current_equity = (self._equity_curve[-1] if self._equity_curve else 0) + pnl
        self._equity_curve.append(current_equity)
        
        # Update peak equity
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        
        # Calculate drawdown
        if self._peak_equity > 0:
            current_drawdown = ((self._peak_equity - current_equity) / self._peak_equity) * 100
            self._drawdown_tracker.append(current_drawdown)
            
            # Keep only recent drawdown data
            if len(self._drawdown_tracker) > 1000:
                self._drawdown_tracker = self._drawdown_tracker[-500:]
    
    def _calculate_all_kpis(self):
        """Calculate and update all KPI metrics."""
        try:
            current_time = time.time()
            
            # Skip if too soon since last calculation
            if current_time - self._last_kpi_calculation < self._kpi_calculation_interval:
                return
            
            with self._lock:
                # Trading performance KPIs
                self._calculate_trading_kpis()
                
                # Risk management KPIs
                self._calculate_risk_kpis()
                
                # Strategy performance KPIs
                self._calculate_strategy_kpis()
                
                # System performance KPIs
                self._calculate_system_kpis()
            
            self._last_kpi_calculation = current_time
            
        except Exception as e:
            logger.error(f"❌ KPI calculation failed: {e}")
    
    def _calculate_trading_kpis(self):
        """Calculate trading performance KPIs."""
        if not self._completed_trades:
            return
        
        # Get recent trades (last 24 hours)
        cutoff_time = time.time() - 86400
        recent_trades = [t for t in self._completed_trades 
                        if t.exit_time and t.exit_time >= cutoff_time]
        
        if not recent_trades:
            return
        
        # Win rate calculation
        winning_trades = [t for t in recent_trades if t.pnl and t.pnl > 0]
        win_rate = (len(winning_trades) / len(recent_trades)) * 100
        self.metrics_collector.set_gauge('trading.win_rate', win_rate)
        
        # PnL calculations
        total_pnl = sum(t.pnl or 0 for t in recent_trades)
        self.metrics_collector.set_gauge('trading.daily_pnl', total_pnl)
        
        # Calculate total PnL across all trades
        all_time_pnl = sum(t.pnl or 0 for t in self._completed_trades)
        self.metrics_collector.set_gauge('trading.total_pnl', all_time_pnl)
    
    def _calculate_risk_kpis(self):
        """Calculate risk management KPIs."""
        # Current drawdown
        if self._drawdown_tracker:
            current_drawdown = self._drawdown_tracker[-1]
            max_drawdown = max(self._drawdown_tracker) if self._drawdown_tracker else 0
            
            self.metrics_collector.set_gauge('risk.current_drawdown', current_drawdown)
            self.metrics_collector.set_gauge('risk.max_drawdown', max_drawdown)
        
        # Active positions and exposure
        self.metrics_collector.set_gauge('risk.active_positions', len(self._active_positions))
        
        # Unrealized PnL
        total_unrealized = sum(pos.unrealized_pnl for pos in self._active_positions.values())
        self.metrics_collector.set_gauge('risk.unrealized_pnl', total_unrealized)
        
        # Total exposure
        total_exposure = sum(pos.entry_price * pos.quantity 
                           for pos in self._active_positions.values())
        self.metrics_collector.set_gauge('risk.total_exposure', total_exposure)
    
    def _calculate_strategy_kpis(self):
        """Calculate strategy performance KPIs."""
        if not self._completed_trades:
            return
        
        # Average trade duration
        durations = [t.duration_seconds for t in self._completed_trades 
                    if t.duration_seconds]
        if durations:
            avg_duration_minutes = statistics.mean(durations) / 60
            self.metrics_collector.set_gauge('strategy.avg_trade_duration', avg_duration_minutes)
        
        # Winning and losing trade amounts
        winning_amounts = [t.pnl for t in self._completed_trades 
                          if t.pnl and t.pnl > 0]
        losing_amounts = [abs(t.pnl) for t in self._completed_trades 
                         if t.pnl and t.pnl < 0]
        
        if winning_amounts:
            avg_win = statistics.mean(winning_amounts)
            self.metrics_collector.set_gauge('strategy.avg_win_amount', avg_win)
        
        if losing_amounts:
            avg_loss = statistics.mean(losing_amounts)
            self.metrics_collector.set_gauge('strategy.avg_loss_amount', avg_loss)
        
        # Profit factor
        if winning_amounts and losing_amounts:
            profit_factor = sum(winning_amounts) / sum(losing_amounts)
            self.metrics_collector.set_gauge('strategy.profit_factor', profit_factor)
        
        # Sharpe ratio (simplified calculation)
        if len(self._completed_trades) > 10:
            returns = [t.calculate_return_percentage() for t in self._completed_trades[-50:]]
            if returns:
                avg_return = statistics.mean(returns)
                return_std = statistics.stdev(returns) if len(returns) > 1 else 1
                sharpe_ratio = avg_return / return_std if return_std > 0 else 0
                self.metrics_collector.set_gauge('strategy.sharpe_ratio', sharpe_ratio)
    
    def _calculate_system_kpis(self):
        """Calculate system performance KPIs."""
        # These would be updated by other system components
        # For now, we just ensure the metrics exist
        pass
    
    def get_trading_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get trading performance summary.
        
        Args:
            hours: Hours to look back
            
        Returns:
            Dictionary with trading summary
        """
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            recent_trades = [t for t in self._completed_trades 
                           if t.exit_time and t.exit_time >= cutoff_time]
            
            if not recent_trades:
                return {'message': 'No trades in specified period'}
            
            winning_trades = [t for t in recent_trades if t.pnl and t.pnl > 0]
            losing_trades = [t for t in recent_trades if t.pnl and t.pnl < 0]
            
            summary = {
                'period_hours': hours,
                'total_trades': len(recent_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': (len(winning_trades) / len(recent_trades) * 100) if recent_trades else 0,
                'total_pnl': sum(t.pnl or 0 for t in recent_trades),
                'avg_pnl_per_trade': statistics.mean([t.pnl or 0 for t in recent_trades]),
                'best_trade': max(recent_trades, key=lambda x: x.pnl or 0).pnl if recent_trades else 0,
                'worst_trade': min(recent_trades, key=lambda x: x.pnl or 0).pnl if recent_trades else 0,
                'active_positions': len(self._active_positions),
                'unrealized_pnl': sum(pos.unrealized_pnl for pos in self._active_positions.values())
            }
            
            return summary
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by strategy."""
        with self._lock:
            performance = {}
            
            for strategy, stats in self._strategy_stats.items():
                if stats['total_trades'] > 0:
                    win_rate = (stats['winning_trades'] / stats['total_trades']) * 100
                    avg_pnl = stats['total_pnl'] / stats['total_trades']
                    
                    performance[strategy] = {
                        'total_trades': stats['total_trades'],
                        'winning_trades': stats['winning_trades'],
                        'win_rate': win_rate,
                        'total_pnl': stats['total_pnl'],
                        'avg_pnl_per_trade': avg_pnl,
                        'total_volume': stats['total_volume']
                    }
            
            return performance
    
    def export_trade_history(self, format_type: str = "json") -> str:
        """Export completed trade history."""
        with self._lock:
            if format_type.lower() == "json":
                trades_data = []
                for trade in self._completed_trades:
                    trades_data.append({
                        'trade_id': trade.trade_id,
                        'symbol': trade.symbol,
                        'side': trade.side,
                        'entry_price': trade.entry_price,
                        'exit_price': trade.exit_price,
                        'quantity': trade.quantity,
                        'pnl': trade.pnl,
                        'return_pct': trade.calculate_return_percentage(),
                        'duration_seconds': trade.duration_seconds,
                        'strategy': trade.strategy,
                        'entry_time': trade.entry_time,
                        'exit_time': trade.exit_time
                    })
                
                import json
                return json.dumps(trades_data, indent=2)
        
        return ""


def create_kpi_tracker(metrics_collector: Optional[MetricsCollector] = None) -> BusinessKPITracker:
    """Factory function to create business KPI tracker."""
    return BusinessKPITracker(metrics_collector)