#!/usr/bin/env python3
"""
Performance Tracker for Adaptive DipMaster Strategy
自适应DipMaster策略性能跟踪器

This module implements comprehensive performance monitoring and analysis for the
adaptive parameter system. It tracks real-time metrics, regime performance,
and provides signal quality analysis to support parameter optimization.

Features:
- Real-time P&L and risk metrics calculation
- Regime-specific performance attribution
- Signal quality and timing analysis
- Drawdown and recovery monitoring
- Portfolio-level risk aggregation

Author: Portfolio Risk Optimizer Agent
Date: 2025-08-17
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
import sqlite3
from pathlib import Path
from collections import defaultdict, deque
import threading
import asyncio
import time

# Statistical and analysis libraries
from scipy import stats
from scipy.stats import norm, t
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Market regime and core components
from .market_regime_detector import MarketRegime, RegimeSignal
from .adaptive_parameter_engine import ParameterSet, PerformanceMetrics
from ..types.common_types import *

warnings.filterwarnings('ignore')

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MetricType(Enum):
    """Types of performance metrics"""
    RETURN = "return"
    RISK = "risk"
    EFFICIENCY = "efficiency"
    REGIME = "regime"
    TIMING = "timing"

@dataclass
class TradeRecord:
    """Individual trade record for tracking"""
    trade_id: str
    symbol: str
    regime: MarketRegime
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    pnl_absolute: float
    pnl_percentage: float
    holding_minutes: int
    parameters_used: Dict
    signal_confidence: float
    exit_reason: str
    slippage: float = 0.0
    commission: float = 0.0

@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time"""
    timestamp: datetime
    total_value: float
    cash_balance: float
    open_positions: int
    total_exposure: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    var_1d: float
    beta: float
    correlation_risk: float
    regime_distribution: Dict[str, float]

@dataclass
class PerformanceAlert:
    """Performance-based alert"""
    alert_id: str
    timestamp: datetime
    level: AlertLevel
    title: str
    message: str
    metrics: Dict
    symbol: Optional[str]
    regime: Optional[MarketRegime]
    action_required: bool

@dataclass
class RegimePerformanceReport:
    """Performance report by regime"""
    regime: MarketRegime
    start_date: datetime
    end_date: datetime
    total_trades: int
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    best_symbols: List[str]
    worst_symbols: List[str]
    parameter_stability: float

class PerformanceTracker:
    """
    Comprehensive Performance Tracking System
    综合性能跟踪系统
    
    Monitors strategy performance across multiple dimensions:
    1. Real-time P&L and risk metrics
    2. Regime-specific performance attribution
    3. Signal quality and prediction accuracy
    4. Parameter effectiveness tracking
    5. Portfolio-level risk aggregation
    """
    
    def __init__(self, config: Optional[Dict] = None, db_path: Optional[str] = None):
        """Initialize performance tracker"""
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Database setup
        self.db_path = db_path or "data/performance_tracker.db"
        self._initialize_database()
        
        # In-memory tracking
        self.trade_records = deque(maxlen=10000)
        self.portfolio_snapshots = deque(maxlen=1000)
        self.alerts = deque(maxlen=500)
        
        # Performance caches
        self.metrics_cache = {}
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        self.symbol_performance = defaultdict(lambda: defaultdict(list))
        
        # Real-time tracking
        self.current_positions = {}
        self.daily_stats = defaultdict(float)
        self.running_pnl = 0.0
        self.starting_capital = 10000.0  # Default starting capital
        
        # Threading
        self._lock = threading.Lock()
        
        # Monitoring flags
        self._is_monitoring = False
        self._last_snapshot = datetime.now()
        
        self.logger.info("PerformanceTracker initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for performance tracking"""
        return {
            'tracking': {
                'snapshot_frequency': 300,  # 5 minutes
                'metrics_cache_ttl': 60,    # 1 minute
                'alert_cooldown': 300,      # 5 minutes
                'max_trade_records': 10000,
                'real_time_updates': True
            },
            'alerts': {
                'drawdown_warning': 0.03,    # 3%
                'drawdown_critical': 0.05,   # 5%
                'win_rate_warning': 0.4,     # 40%
                'sharpe_warning': 1.0,       # Sharpe < 1.0
                'var_warning': 0.02,         # 2% daily VaR
                'correlation_warning': 0.8    # 80% correlation
            },
            'targets': {
                'daily_return': 0.001,       # 0.1% daily
                'win_rate': 0.65,            # 65%
                'sharpe_ratio': 2.0,         # Target Sharpe
                'max_drawdown': 0.05,        # 5%
                'profit_factor': 1.5,        # 1.5x profit factor
                'var_limit': 0.02            # 2% daily VaR
            },
            'regime_weights': {
                'RANGE_BOUND': 0.4,
                'STRONG_UPTREND': 0.25,
                'STRONG_DOWNTREND': 0.15,
                'HIGH_VOLATILITY': 0.1,
                'LOW_VOLATILITY': 0.1
            }
        }
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Trade records table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_records (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    position_size REAL NOT NULL,
                    pnl_absolute REAL NOT NULL,
                    pnl_percentage REAL NOT NULL,
                    holding_minutes INTEGER NOT NULL,
                    parameters_used TEXT,
                    signal_confidence REAL,
                    exit_reason TEXT,
                    slippage REAL DEFAULT 0.0,
                    commission REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Portfolio snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    total_value REAL NOT NULL,
                    cash_balance REAL NOT NULL,
                    open_positions INTEGER NOT NULL,
                    total_exposure REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    var_1d REAL NOT NULL,
                    beta REAL NOT NULL,
                    correlation_risk REAL NOT NULL,
                    regime_distribution TEXT
                )
            ''')
            
            # Performance alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    level TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metrics TEXT,
                    symbol TEXT,
                    regime TEXT,
                    action_required BOOLEAN NOT NULL,
                    acknowledged BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
        
        self.logger.info(f"Database initialized at {self.db_path}")
    
    def record_trade(self, trade: TradeRecord):
        """Record a completed trade"""
        with self._lock:
            # Add to in-memory storage
            self.trade_records.append(trade)
            
            # Update regime and symbol performance
            self.regime_performance[trade.regime][trade.symbol].append(trade)
            self.symbol_performance[trade.symbol][trade.regime].append(trade)
            
            # Update running P&L
            self.running_pnl += trade.pnl_absolute
            self.daily_stats['realized_pnl'] += trade.pnl_absolute
            
            # Store in database
            self._store_trade_in_db(trade)
            
            # Check for alerts
            self._check_trade_alerts(trade)
            
            # Clear metrics cache
            self.metrics_cache.clear()
        
        self.logger.debug(f"Trade recorded: {trade.symbol} {trade.regime.value} "
                         f"PnL: ${trade.pnl_absolute:.2f} ({trade.pnl_percentage:.2%})")
    
    def _store_trade_in_db(self, trade: TradeRecord):
        """Store trade record in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO trade_records (
                        trade_id, symbol, regime, entry_time, exit_time, entry_price,
                        exit_price, position_size, pnl_absolute, pnl_percentage,
                        holding_minutes, parameters_used, signal_confidence,
                        exit_reason, slippage, commission
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.trade_id, trade.symbol, trade.regime.value,
                    trade.entry_time, trade.exit_time, trade.entry_price,
                    trade.exit_price, trade.position_size, trade.pnl_absolute,
                    trade.pnl_percentage, trade.holding_minutes,
                    json.dumps(trade.parameters_used), trade.signal_confidence,
                    trade.exit_reason, trade.slippage, trade.commission
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store trade in database: {e}")
    
    def update_portfolio_snapshot(self, positions: Dict, market_values: Dict):
        """Update current portfolio state"""
        snapshot_time = datetime.now()
        
        # Calculate portfolio metrics
        total_value = sum(market_values.values())
        cash_balance = self.starting_capital + self.running_pnl - sum(
            pos.get('market_value', 0) for pos in positions.values()
        )
        
        open_positions = len([p for p in positions.values() if p.get('quantity', 0) != 0])
        total_exposure = sum(abs(pos.get('market_value', 0)) for pos in positions.values())
        
        # Calculate unrealized P&L
        unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions.values())
        
        # Calculate daily P&L
        daily_pnl = self.daily_stats.get('realized_pnl', 0) + unrealized_pnl
        
        # Risk metrics (simplified calculations)
        var_1d = self._calculate_portfolio_var(positions, market_values)
        beta = self._calculate_portfolio_beta(positions, market_values)
        correlation_risk = self._calculate_correlation_risk(positions)
        
        # Regime distribution
        regime_distribution = self._calculate_regime_distribution(positions)
        
        snapshot = PortfolioSnapshot(
            timestamp=snapshot_time,
            total_value=total_value,
            cash_balance=cash_balance,
            open_positions=open_positions,
            total_exposure=total_exposure,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.daily_stats.get('realized_pnl', 0),
            daily_pnl=daily_pnl,
            var_1d=var_1d,
            beta=beta,
            correlation_risk=correlation_risk,
            regime_distribution=regime_distribution
        )
        
        with self._lock:
            self.portfolio_snapshots.append(snapshot)
            self.current_positions = positions.copy()
            
            # Store in database
            self._store_snapshot_in_db(snapshot)
            
            # Check for portfolio-level alerts
            self._check_portfolio_alerts(snapshot)
        
        self._last_snapshot = snapshot_time
    
    def _calculate_portfolio_var(self, positions: Dict, market_values: Dict) -> float:
        """Calculate portfolio Value at Risk (simplified)"""
        if not positions:
            return 0.0
        
        # Simple VaR calculation based on position values and assumed volatilities
        total_variance = 0.0
        for symbol, position in positions.items():
            market_value = market_values.get(symbol, 0)
            # Assume 2% daily volatility for crypto
            daily_vol = 0.02
            position_var = abs(market_value) * daily_vol
            total_variance += position_var ** 2
        
        portfolio_vol = np.sqrt(total_variance)
        # 95% VaR
        var_95 = portfolio_vol * 1.645
        
        return var_95 / max(sum(market_values.values()), 1)  # As percentage of portfolio
    
    def _calculate_portfolio_beta(self, positions: Dict, market_values: Dict) -> float:
        """Calculate portfolio beta (simplified)"""
        if not positions:
            return 0.0
        
        # Simplified beta calculation - assume all crypto has beta 1.0 to market
        total_value = sum(market_values.values())
        if total_value == 0:
            return 0.0
        
        weighted_beta = 0.0
        for symbol, market_value in market_values.items():
            weight = market_value / total_value
            # Assume different betas for different assets
            if 'BTC' in symbol:
                beta = 1.0
            elif 'ETH' in symbol:
                beta = 1.2
            else:
                beta = 1.5  # Alt coins typically higher beta
            
            weighted_beta += weight * beta
        
        return weighted_beta
    
    def _calculate_correlation_risk(self, positions: Dict) -> float:
        """Calculate portfolio correlation risk"""
        if len(positions) <= 1:
            return 0.0
        
        # Simplified correlation risk - assume high correlation for crypto
        return 0.7  # Placeholder for now
    
    def _calculate_regime_distribution(self, positions: Dict) -> Dict[str, float]:
        """Calculate distribution of positions across regimes"""
        if not positions:
            return {}
        
        # Simplified regime distribution based on current holdings
        # In practice, this would use actual regime classifications
        regime_dist = {
            'RANGE_BOUND': 0.4,
            'STRONG_UPTREND': 0.3,
            'STRONG_DOWNTREND': 0.1,
            'HIGH_VOLATILITY': 0.1,
            'LOW_VOLATILITY': 0.1
        }
        
        return regime_dist
    
    def _store_snapshot_in_db(self, snapshot: PortfolioSnapshot):
        """Store portfolio snapshot in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO portfolio_snapshots (
                        timestamp, total_value, cash_balance, open_positions,
                        total_exposure, unrealized_pnl, realized_pnl, daily_pnl,
                        var_1d, beta, correlation_risk, regime_distribution
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot.timestamp, snapshot.total_value, snapshot.cash_balance,
                    snapshot.open_positions, snapshot.total_exposure,
                    snapshot.unrealized_pnl, snapshot.realized_pnl, snapshot.daily_pnl,
                    snapshot.var_1d, snapshot.beta, snapshot.correlation_risk,
                    json.dumps(snapshot.regime_distribution)
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store portfolio snapshot: {e}")
    
    def get_performance_metrics(self, symbol: Optional[str] = None,
                              regime: Optional[MarketRegime] = None,
                              timeframe: str = '1d') -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        cache_key = f"{symbol}_{regime}_{timeframe}"
        
        # Check cache
        if cache_key in self.metrics_cache:
            cached_time, metrics = self.metrics_cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=self.config['tracking']['metrics_cache_ttl']):
                return metrics
        
        # Filter trades
        trades = self._filter_trades(symbol, regime, timeframe)
        
        if not trades:
            return PerformanceMetrics(
                win_rate=0.0, avg_return=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                max_drawdown=0.0, var_95=0.0, expected_shortfall=0.0,
                profit_factor=0.0, avg_holding_time=0.0, regime_consistency=0.0
            )
        
        # Calculate metrics
        returns = [trade.pnl_percentage for trade in trades]
        holding_times = [trade.holding_minutes for trade in trades]
        
        returns_array = np.array(returns)
        
        # Basic metrics
        win_rate = np.mean(returns_array > 0)
        avg_return = np.mean(returns_array)
        
        # Risk-adjusted metrics
        if np.std(returns_array) > 0:
            sharpe_ratio = avg_return / np.std(returns_array) * np.sqrt(365 * 24 * 12)
        else:
            sharpe_ratio = 0.0
        
        # Sortino ratio
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino_ratio = avg_return / np.std(downside_returns) * np.sqrt(365 * 24 * 12)
        else:
            sortino_ratio = sharpe_ratio
        
        # Drawdown
        cumulative_returns = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        
        # VaR and ES
        var_95 = abs(np.percentile(returns_array, 5)) if len(returns_array) > 0 else 0.0
        tail_losses = returns_array[returns_array <= np.percentile(returns_array, 5)]
        expected_shortfall = abs(np.mean(tail_losses)) if len(tail_losses) > 0 else 0.0
        
        # Profit factor
        winning_trades = returns_array[returns_array > 0]
        losing_trades = returns_array[returns_array < 0]
        
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            profit_factor = np.sum(winning_trades) / abs(np.sum(losing_trades))
        else:
            profit_factor = float('inf') if len(winning_trades) > 0 else 0.0
        
        # Average holding time
        avg_holding_time = np.mean(holding_times) if holding_times else 0.0
        
        # Regime consistency
        regime_consistency = self._calculate_regime_consistency(trades)
        
        metrics = PerformanceMetrics(
            win_rate=win_rate,
            avg_return=avg_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            profit_factor=profit_factor,
            avg_holding_time=avg_holding_time,
            regime_consistency=regime_consistency
        )
        
        # Cache result
        self.metrics_cache[cache_key] = (datetime.now(), metrics)
        
        return metrics
    
    def _filter_trades(self, symbol: Optional[str], regime: Optional[MarketRegime],
                      timeframe: str) -> List[TradeRecord]:
        """Filter trades based on criteria"""
        trades = list(self.trade_records)
        
        # Filter by symbol
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        # Filter by regime
        if regime:
            trades = [t for t in trades if t.regime == regime]
        
        # Filter by timeframe
        now = datetime.now()
        if timeframe == '1h':
            cutoff = now - timedelta(hours=1)
        elif timeframe == '1d':
            cutoff = now - timedelta(days=1)
        elif timeframe == '1w':
            cutoff = now - timedelta(weeks=1)
        elif timeframe == '1m':
            cutoff = now - timedelta(days=30)
        else:
            cutoff = datetime.min
        
        trades = [t for t in trades if t.exit_time and t.exit_time >= cutoff]
        
        return trades
    
    def _calculate_regime_consistency(self, trades: List[TradeRecord]) -> float:
        """Calculate regime consistency score"""
        if not trades:
            return 0.0
        
        # Group trades by regime
        regime_performance = defaultdict(list)
        for trade in trades:
            regime_performance[trade.regime].append(trade.pnl_percentage)
        
        # Calculate consistency across regimes
        regime_sharpes = {}
        for regime, returns in regime_performance.items():
            if len(returns) > 1 and np.std(returns) > 0:
                regime_sharpes[regime] = np.mean(returns) / np.std(returns)
        
        if not regime_sharpes:
            return 0.0
        
        # Consistency is inverse of variance in regime Sharpes
        sharpe_values = list(regime_sharpes.values())
        if len(sharpe_values) > 1:
            consistency = 1.0 / (1.0 + np.var(sharpe_values))
        else:
            consistency = 1.0
        
        return consistency
    
    def _check_trade_alerts(self, trade: TradeRecord):
        """Check for trade-specific alerts"""
        alerts_config = self.config['alerts']
        
        # Large loss alert
        if trade.pnl_percentage < -0.02:  # 2% loss
            self._create_alert(
                level=AlertLevel.WARNING,
                title=f"Large Loss - {trade.symbol}",
                message=f"Trade loss of {trade.pnl_percentage:.2%} in {trade.regime.value}",
                symbol=trade.symbol,
                regime=trade.regime,
                metrics={'pnl_percentage': trade.pnl_percentage}
            )
        
        # Long holding time alert
        if trade.holding_minutes > 240:  # 4 hours
            self._create_alert(
                level=AlertLevel.INFO,
                title=f"Long Holding Time - {trade.symbol}",
                message=f"Trade held for {trade.holding_minutes} minutes",
                symbol=trade.symbol,
                regime=trade.regime,
                metrics={'holding_minutes': trade.holding_minutes}
            )
    
    def _check_portfolio_alerts(self, snapshot: PortfolioSnapshot):
        """Check for portfolio-level alerts"""
        alerts_config = self.config['alerts']
        
        # Drawdown alerts
        if snapshot.total_value > 0:
            drawdown = (self.starting_capital - snapshot.total_value) / self.starting_capital
            
            if drawdown > alerts_config['drawdown_critical']:
                self._create_alert(
                    level=AlertLevel.CRITICAL,
                    title="Critical Drawdown Alert",
                    message=f"Portfolio drawdown: {drawdown:.2%}",
                    metrics={'drawdown': drawdown},
                    action_required=True
                )
            elif drawdown > alerts_config['drawdown_warning']:
                self._create_alert(
                    level=AlertLevel.WARNING,
                    title="Drawdown Warning",
                    message=f"Portfolio drawdown: {drawdown:.2%}",
                    metrics={'drawdown': drawdown}
                )
        
        # VaR alert
        if snapshot.var_1d > alerts_config['var_warning']:
            self._create_alert(
                level=AlertLevel.WARNING,
                title="High VaR Alert",
                message=f"Daily VaR: {snapshot.var_1d:.2%}",
                metrics={'var_1d': snapshot.var_1d}
            )
        
        # Correlation alert
        if snapshot.correlation_risk > alerts_config['correlation_warning']:
            self._create_alert(
                level=AlertLevel.WARNING,
                title="High Correlation Risk",
                message=f"Portfolio correlation risk: {snapshot.correlation_risk:.2%}",
                metrics={'correlation_risk': snapshot.correlation_risk}
            )
    
    def _create_alert(self, level: AlertLevel, title: str, message: str,
                     symbol: Optional[str] = None, regime: Optional[MarketRegime] = None,
                     metrics: Optional[Dict] = None, action_required: bool = False):
        """Create and store performance alert"""
        alert = PerformanceAlert(
            alert_id=f"alert_{int(time.time() * 1000)}",
            timestamp=datetime.now(),
            level=level,
            title=title,
            message=message,
            metrics=metrics or {},
            symbol=symbol,
            regime=regime,
            action_required=action_required
        )
        
        with self._lock:
            self.alerts.append(alert)
            
            # Store in database
            self._store_alert_in_db(alert)
        
        self.logger.log(
            logging.ERROR if level == AlertLevel.CRITICAL else logging.WARNING,
            f"{level.value.upper()}: {title} - {message}"
        )
    
    def _store_alert_in_db(self, alert: PerformanceAlert):
        """Store alert in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_alerts (
                        alert_id, timestamp, level, title, message, metrics,
                        symbol, regime, action_required
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.alert_id, alert.timestamp, alert.level.value,
                    alert.title, alert.message, json.dumps(alert.metrics),
                    alert.symbol, alert.regime.value if alert.regime else None,
                    alert.action_required
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store alert: {e}")
    
    def get_regime_performance_report(self, regime: MarketRegime,
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None) -> RegimePerformanceReport:
        """Generate comprehensive performance report for a regime"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # Filter trades for regime and date range
        regime_trades = [
            trade for trade in self.trade_records
            if trade.regime == regime and 
            trade.exit_time and 
            start_date <= trade.exit_time <= end_date
        ]
        
        if not regime_trades:
            return RegimePerformanceReport(
                regime=regime,
                start_date=start_date,
                end_date=end_date,
                total_trades=0,
                win_rate=0.0,
                avg_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                profit_factor=0.0,
                best_symbols=[],
                worst_symbols=[],
                parameter_stability=0.0
            )
        
        # Calculate metrics
        returns = [trade.pnl_percentage for trade in regime_trades]
        returns_array = np.array(returns)
        
        total_trades = len(regime_trades)
        win_rate = np.mean(returns_array > 0)
        avg_return = np.mean(returns_array)
        
        if np.std(returns_array) > 0:
            sharpe_ratio = avg_return / np.std(returns_array) * np.sqrt(365 * 24 * 12)
        else:
            sharpe_ratio = 0.0
        
        # Drawdown
        cumulative_returns = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        
        # Profit factor
        winning_trades = returns_array[returns_array > 0]
        losing_trades = returns_array[returns_array < 0]
        
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            profit_factor = np.sum(winning_trades) / abs(np.sum(losing_trades))
        else:
            profit_factor = float('inf') if len(winning_trades) > 0 else 0.0
        
        # Symbol performance analysis
        symbol_performance = defaultdict(list)
        for trade in regime_trades:
            symbol_performance[trade.symbol].append(trade.pnl_percentage)
        
        symbol_avg_returns = {
            symbol: np.mean(returns) 
            for symbol, returns in symbol_performance.items()
        }
        
        best_symbols = sorted(symbol_avg_returns.keys(), 
                            key=lambda x: symbol_avg_returns[x], reverse=True)[:5]
        worst_symbols = sorted(symbol_avg_returns.keys(), 
                             key=lambda x: symbol_avg_returns[x])[:5]
        
        # Parameter stability (simplified)
        parameter_stability = 0.8  # Placeholder
        
        return RegimePerformanceReport(
            regime=regime,
            start_date=start_date,
            end_date=end_date,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_return=avg_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            best_symbols=best_symbols,
            worst_symbols=worst_symbols,
            parameter_stability=parameter_stability
        )
    
    def get_real_time_dashboard_data(self) -> Dict:
        """Get real-time data for dashboard display"""
        current_snapshot = self.portfolio_snapshots[-1] if self.portfolio_snapshots else None
        recent_trades = list(self.trade_records)[-20:]  # Last 20 trades
        recent_alerts = list(self.alerts)[-10:]  # Last 10 alerts
        
        # Calculate current day performance
        today = datetime.now().date()
        today_trades = [
            trade for trade in self.trade_records
            if trade.exit_time and trade.exit_time.date() == today
        ]
        
        today_metrics = self.get_performance_metrics(timeframe='1d')
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio': {
                'total_value': current_snapshot.total_value if current_snapshot else 0.0,
                'daily_pnl': current_snapshot.daily_pnl if current_snapshot else 0.0,
                'unrealized_pnl': current_snapshot.unrealized_pnl if current_snapshot else 0.0,
                'open_positions': current_snapshot.open_positions if current_snapshot else 0,
                'var_1d': current_snapshot.var_1d if current_snapshot else 0.0,
                'beta': current_snapshot.beta if current_snapshot else 0.0
            },
            'performance': {
                'today_trades': len(today_trades),
                'win_rate': today_metrics.win_rate,
                'avg_return': today_metrics.avg_return,
                'sharpe_ratio': today_metrics.sharpe_ratio,
                'max_drawdown': today_metrics.max_drawdown
            },
            'recent_trades': [
                {
                    'symbol': trade.symbol,
                    'regime': trade.regime.value,
                    'pnl_percentage': trade.pnl_percentage,
                    'holding_minutes': trade.holding_minutes,
                    'exit_time': trade.exit_time.isoformat() if trade.exit_time else None
                }
                for trade in recent_trades
            ],
            'alerts': [
                {
                    'level': alert.level.value,
                    'title': alert.title,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in recent_alerts
            ],
            'regime_distribution': current_snapshot.regime_distribution if current_snapshot else {}
        }
    
    def export_performance_report(self, output_path: str, 
                                include_trades: bool = True,
                                include_charts: bool = True):
        """Export comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_trades': len(self.trade_records),
                'total_pnl': self.running_pnl,
                'tracking_period': {
                    'start': min(trade.entry_time for trade in self.trade_records).isoformat() if self.trade_records else None,
                    'end': max(trade.exit_time for trade in self.trade_records if trade.exit_time).isoformat() if self.trade_records else None
                }
            },
            'overall_performance': asdict(self.get_performance_metrics()),
            'regime_performance': {},
            'symbol_performance': {},
            'recent_alerts': [asdict(alert) for alert in list(self.alerts)[-50:]]
        }
        
        # Regime performance
        for regime in MarketRegime:
            regime_report = self.get_regime_performance_report(regime)
            report['regime_performance'][regime.value] = asdict(regime_report)
        
        # Symbol performance (top symbols)
        symbol_trades = defaultdict(list)
        for trade in self.trade_records:
            symbol_trades[trade.symbol].append(trade)
        
        for symbol in sorted(symbol_trades.keys())[:20]:  # Top 20 symbols
            symbol_metrics = self.get_performance_metrics(symbol=symbol)
            report['symbol_performance'][symbol] = asdict(symbol_metrics)
        
        # Include detailed trades if requested
        if include_trades:
            report['detailed_trades'] = [asdict(trade) for trade in list(self.trade_records)[-1000:]]
        
        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Performance report exported to {output_file}")
        
        return report

# Factory function
def create_performance_tracker(config: Optional[Dict] = None, 
                             db_path: Optional[str] = None) -> PerformanceTracker:
    """Factory function to create performance tracker"""
    return PerformanceTracker(config, db_path)