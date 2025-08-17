#!/usr/bin/env python3
"""
Multi-Timeframe Performance Tracker for DipMaster Strategy
多时间框架性能跟踪器 - DipMaster策略专用

This module implements comprehensive performance tracking and analytics for the
multi-timeframe signal integration system. It monitors signal quality, execution
efficiency, and provides feedback for continuous optimization.

Key Features:
- Real-time signal quality assessment
- Multi-timeframe execution analytics
- Performance attribution analysis
- Optimization feedback mechanisms
- Risk-adjusted performance metrics
- Signal confluence effectiveness tracking

Performance Targets:
- Signal Accuracy: >85% confluence score reliability
- Execution Quality: <0.5 bps average slippage
- Risk Metrics: Sharpe >4.0, Max DD <3%
- Timing Efficiency: >90% optimal execution windows

Author: Monitoring Log Collector Agent
Date: 2025-08-17
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
import sqlite3
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
from pathlib import Path
import pickle

# Statistical and ML libraries
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Core components
from .multi_timeframe_signal_engine import (
    MultitimeframeSignal, TimeFrameSignal, TimeFrame, ExecutionRecommendation
)
from .market_regime_detector import MarketRegime
from ..types.common_types import *

warnings.filterwarnings('ignore')

class PerformanceMetric(Enum):
    """Performance metric types"""
    WIN_RATE = "win_rate"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    PROFIT_FACTOR = "profit_factor"
    AVERAGE_RETURN = "average_return"
    SIGNAL_ACCURACY = "signal_accuracy"
    EXECUTION_QUALITY = "execution_quality"
    CONFLUENCE_EFFECTIVENESS = "confluence_effectiveness"

class SignalOutcome(Enum):
    """Signal outcome classifications"""
    TRUE_POSITIVE = "true_positive"    # Predicted bull, was bull
    TRUE_NEGATIVE = "true_negative"    # Predicted bear, was bear
    FALSE_POSITIVE = "false_positive"  # Predicted bull, was bear
    FALSE_NEGATIVE = "false_negative"  # Predicted bear, was bull
    NEUTRAL = "neutral"                # No clear direction

@dataclass
class TradeResult:
    """Complete trade result data"""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    side: str  # 'buy' or 'sell'
    pnl: float
    pnl_pct: float
    holding_period_minutes: int
    execution_method: str
    slippage_bps: float
    
    # Signal information
    entry_signal: MultitimeframeSignal
    exit_signal: Optional[MultitimeframeSignal]
    confluence_score_entry: float
    confluence_score_exit: Optional[float]
    
    # Performance metrics
    signal_accuracy: float
    execution_quality: float
    risk_adjusted_return: float
    
    # Market context
    market_regime: MarketRegime
    volatility_regime: str
    timeframe_consistency: float

@dataclass
class SignalPerformanceMetrics:
    """Signal-specific performance metrics"""
    symbol: str
    timeframe: TimeFrame
    total_signals: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_confluence_score: float
    confidence_correlation: float  # Correlation between confidence and outcome
    false_positive_rate: float
    false_negative_rate: float
    signal_consistency: float

@dataclass
class ExecutionPerformanceMetrics:
    """Execution-specific performance metrics"""
    symbol: str
    execution_method: str
    total_executions: int
    avg_slippage_bps: float
    fill_rate: float
    timing_efficiency: float
    cost_effectiveness: float
    market_impact: float
    latency_ms: float

@dataclass
class RiskMetrics:
    """Risk assessment metrics"""
    var_95: float
    cvar_95: float
    max_drawdown: float
    volatility: float
    skewness: float
    kurtosis: float
    tail_ratio: float
    downside_deviation: float

class MultitimeframePerformanceTracker:
    """
    Comprehensive Multi-Timeframe Performance Tracking System
    综合多时间框架性能跟踪系统
    
    Tracks and analyzes performance across all aspects of the multi-timeframe
    signal integration system, providing actionable insights for optimization.
    """
    
    def __init__(self, config: Optional[Dict] = None, db_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Database for persistent storage
        self.db_path = db_path or "data/multitf_performance.db"
        self._initialize_database()
        
        # In-memory tracking
        self.trade_results = deque(maxlen=10000)
        self.signal_history = defaultdict(lambda: deque(maxlen=5000))
        self.execution_history = defaultdict(lambda: deque(maxlen=5000))
        
        # Performance caches
        self.performance_cache = {}
        self.optimization_suggestions = defaultdict(list)
        
        # Real-time metrics
        self.live_metrics = defaultdict(dict)
        self.alert_thresholds = self.config['alert_thresholds']
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance benchmark tracking
        self.benchmarks = {
            'signal_accuracy': 0.85,
            'execution_slippage': 0.5,  # bps
            'sharpe_ratio': 4.0,
            'max_drawdown': 0.03,
            'win_rate': 0.70
        }
        
        self.logger.info("MultitimeframePerformanceTracker initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Default configuration for performance tracking"""
        return {
            'tracking_window_days': 30,
            'min_sample_size': 50,
            'analysis_intervals': {
                'real_time': 1,      # minutes
                'short_term': 60,    # minutes  
                'medium_term': 1440, # minutes (1 day)
                'long_term': 10080   # minutes (1 week)
            },
            'alert_thresholds': {
                'signal_accuracy_min': 0.75,
                'execution_slippage_max': 1.0,  # bps
                'sharpe_ratio_min': 2.0,
                'max_drawdown_max': 0.05,
                'win_rate_min': 0.60
            },
            'performance_weights': {
                'signal_quality': 0.30,
                'execution_efficiency': 0.25,
                'risk_management': 0.25,
                'profitability': 0.20
            },
            'optimization_triggers': {
                'performance_degradation': 0.15,  # 15% drop triggers optimization
                'signal_accuracy_drop': 0.10,     # 10% accuracy drop
                'execution_quality_drop': 0.20    # 20% execution quality drop
            }
        }
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Trade results table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trade_results (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    entry_price REAL,
                    exit_price REAL,
                    size REAL,
                    side TEXT,
                    pnl REAL,
                    pnl_pct REAL,
                    holding_period_minutes INTEGER,
                    execution_method TEXT,
                    slippage_bps REAL,
                    confluence_score_entry REAL,
                    confluence_score_exit REAL,
                    signal_accuracy REAL,
                    execution_quality REAL,
                    risk_adjusted_return REAL,
                    market_regime TEXT,
                    volatility_regime TEXT,
                    timeframe_consistency REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Signal performance table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signal_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timeframe TEXT,
                    signal_time TIMESTAMP,
                    confluence_score REAL,
                    execution_signal TEXT,
                    actual_outcome TEXT,
                    signal_accuracy REAL,
                    market_regime TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Execution performance table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS execution_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    execution_time TIMESTAMP,
                    execution_method TEXT,
                    target_price REAL,
                    actual_price REAL,
                    slippage_bps REAL,
                    fill_rate REAL,
                    latency_ms REAL,
                    market_impact REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
        
        self.logger.info(f"Database initialized at {self.db_path}")
    
    def record_trade_result(self, trade_result: TradeResult):
        """Record a completed trade result"""
        with self._lock:
            # Add to in-memory storage
            self.trade_results.append(trade_result)
            
            # Add to database
            self._save_trade_to_db(trade_result)
            
            # Update real-time metrics
            self._update_live_metrics(trade_result)
            
            # Check for alerts
            self._check_performance_alerts(trade_result.symbol)
        
        self.logger.info(f"Trade result recorded: {trade_result.trade_id} - "
                        f"PnL: {trade_result.pnl_pct:.2%}, Accuracy: {trade_result.signal_accuracy:.2f}")
    
    def _save_trade_to_db(self, trade_result: TradeResult):
        """Save trade result to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO trade_results (
                        trade_id, symbol, entry_time, exit_time, entry_price, exit_price,
                        size, side, pnl, pnl_pct, holding_period_minutes, execution_method,
                        slippage_bps, confluence_score_entry, confluence_score_exit,
                        signal_accuracy, execution_quality, risk_adjusted_return,
                        market_regime, volatility_regime, timeframe_consistency
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_result.trade_id, trade_result.symbol,
                    trade_result.entry_time, trade_result.exit_time,
                    trade_result.entry_price, trade_result.exit_price,
                    trade_result.size, trade_result.side,
                    trade_result.pnl, trade_result.pnl_pct,
                    trade_result.holding_period_minutes, trade_result.execution_method,
                    trade_result.slippage_bps, trade_result.confluence_score_entry,
                    trade_result.confluence_score_exit, trade_result.signal_accuracy,
                    trade_result.execution_quality, trade_result.risk_adjusted_return,
                    trade_result.market_regime.value, trade_result.volatility_regime,
                    trade_result.timeframe_consistency
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error saving trade to database: {e}")
    
    def record_signal_performance(self, signal: MultitimeframeSignal, 
                                actual_outcome: SignalOutcome,
                                market_regime: MarketRegime):
        """Record signal performance for analysis"""
        
        # Calculate signal accuracy based on outcome
        if signal.confluence_score > 0.6:  # Strong signal
            if actual_outcome in [SignalOutcome.TRUE_POSITIVE, SignalOutcome.TRUE_NEGATIVE]:
                accuracy = 1.0
            else:
                accuracy = 0.0
        elif signal.confluence_score > 0.3:  # Weak signal
            if actual_outcome == SignalOutcome.NEUTRAL:
                accuracy = 0.8  # Correctly identified uncertain market
            elif actual_outcome in [SignalOutcome.TRUE_POSITIVE, SignalOutcome.TRUE_NEGATIVE]:
                accuracy = 0.6
            else:
                accuracy = 0.2
        else:  # Very weak signal
            accuracy = 0.5  # Neutral
        
        # Save to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO signal_performance (
                        symbol, timeframe, signal_time, confluence_score,
                        execution_signal, actual_outcome, signal_accuracy, market_regime
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal.symbol, 'multi_tf', signal.timestamp,
                    signal.confluence_score, signal.execution_signal,
                    actual_outcome.value, accuracy, market_regime.value
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error saving signal performance: {e}")
    
    def record_execution_performance(self, symbol: str, execution_method: str,
                                   target_price: float, actual_price: float,
                                   fill_rate: float, latency_ms: float):
        """Record execution performance metrics"""
        
        # Calculate slippage
        slippage_bps = abs(actual_price - target_price) / target_price * 10000
        
        # Estimate market impact (simplified)
        market_impact = max(0, slippage_bps - 0.5)  # Anything above 0.5 bps is impact
        
        # Save to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO execution_performance (
                        symbol, execution_time, execution_method, target_price,
                        actual_price, slippage_bps, fill_rate, latency_ms, market_impact
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, datetime.now(), execution_method, target_price,
                    actual_price, slippage_bps, fill_rate, latency_ms, market_impact
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error saving execution performance: {e}")
    
    def _update_live_metrics(self, trade_result: TradeResult):
        """Update real-time performance metrics"""
        symbol = trade_result.symbol
        
        # Recent trades for rolling metrics
        recent_trades = [tr for tr in self.trade_results 
                        if tr.symbol == symbol and 
                        tr.exit_time >= datetime.now() - timedelta(hours=24)]
        
        if not recent_trades:
            return
        
        # Calculate rolling metrics
        pnl_values = [tr.pnl_pct for tr in recent_trades]
        accuracy_values = [tr.signal_accuracy for tr in recent_trades]
        slippage_values = [tr.slippage_bps for tr in recent_trades]
        
        self.live_metrics[symbol] = {
            'win_rate': np.mean([pnl > 0 for pnl in pnl_values]),
            'avg_return': np.mean(pnl_values),
            'avg_signal_accuracy': np.mean(accuracy_values),
            'avg_slippage': np.mean(slippage_values),
            'sharpe_ratio': self._calculate_sharpe_ratio(pnl_values),
            'max_drawdown': self._calculate_max_drawdown(pnl_values),
            'total_trades': len(recent_trades),
            'last_update': datetime.now()
        }
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_return = np.mean(returns_array) - risk_free_rate / (252 * 24 * 12)  # 5-minute periods
        
        if np.std(returns_array) == 0:
            return 0.0
        
        sharpe = excess_return / np.std(returns_array) * np.sqrt(252 * 24 * 12)  # Annualized
        return sharpe
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0.0
        
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        
        return abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    def _check_performance_alerts(self, symbol: str):
        """Check for performance alerts and trigger notifications"""
        if symbol not in self.live_metrics:
            return
        
        metrics = self.live_metrics[symbol]
        alerts = []
        
        # Check each threshold
        if metrics['avg_signal_accuracy'] < self.alert_thresholds['signal_accuracy_min']:
            alerts.append(f"Signal accuracy below threshold: {metrics['avg_signal_accuracy']:.2f}")
        
        if metrics['avg_slippage'] > self.alert_thresholds['execution_slippage_max']:
            alerts.append(f"Execution slippage above threshold: {metrics['avg_slippage']:.2f} bps")
        
        if metrics['sharpe_ratio'] < self.alert_thresholds['sharpe_ratio_min']:
            alerts.append(f"Sharpe ratio below threshold: {metrics['sharpe_ratio']:.2f}")
        
        if metrics['max_drawdown'] > self.alert_thresholds['max_drawdown_max']:
            alerts.append(f"Max drawdown above threshold: {metrics['max_drawdown']:.2%}")
        
        if metrics['win_rate'] < self.alert_thresholds['win_rate_min']:
            alerts.append(f"Win rate below threshold: {metrics['win_rate']:.2%}")
        
        # Log alerts
        for alert in alerts:
            self.logger.warning(f"PERFORMANCE ALERT for {symbol}: {alert}")
        
        # Store alerts for reporting
        if alerts:
            self.optimization_suggestions[symbol].extend(alerts)
    
    def calculate_signal_performance_metrics(self, symbol: str, 
                                           timeframe: Optional[TimeFrame] = None,
                                           lookback_days: int = 30) -> SignalPerformanceMetrics:
        """Calculate comprehensive signal performance metrics"""
        
        # Query signal performance data
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM signal_performance 
                    WHERE symbol = ? AND signal_time >= ?
                '''
                params = [symbol, cutoff_date]
                
                if timeframe:
                    query += ' AND timeframe = ?'
                    params.append(timeframe.value)
                
                df = pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            self.logger.error(f"Error querying signal performance: {e}")
            df = pd.DataFrame()
        
        if df.empty:
            return SignalPerformanceMetrics(
                symbol=symbol,
                timeframe=timeframe or TimeFrame.M15,
                total_signals=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                avg_confluence_score=0.0,
                confidence_correlation=0.0,
                false_positive_rate=0.0,
                false_negative_rate=0.0,
                signal_consistency=0.0
            )
        
        # Calculate metrics
        total_signals = len(df)
        avg_accuracy = df['signal_accuracy'].mean()
        avg_confluence = df['confluence_score'].mean()
        
        # Binary classification metrics (simplified)
        strong_signals = df['confluence_score'] > 0.6
        correct_signals = df['signal_accuracy'] > 0.8
        
        if len(strong_signals) > 0:
            precision = np.mean(correct_signals[strong_signals]) if strong_signals.sum() > 0 else 0.0
            recall = np.mean(strong_signals[correct_signals]) if correct_signals.sum() > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            precision = recall = f1 = 0.0
        
        # Confidence correlation
        if len(df) > 10:
            confidence_correlation = np.corrcoef(df['confluence_score'], df['signal_accuracy'])[0, 1]
            if np.isnan(confidence_correlation):
                confidence_correlation = 0.0
        else:
            confidence_correlation = 0.0
        
        # False positive/negative rates (simplified)
        false_positive_rate = np.mean((strong_signals) & (~correct_signals))
        false_negative_rate = np.mean((~strong_signals) & (correct_signals))
        
        # Signal consistency (variance in confluence scores)
        signal_consistency = 1.0 - (df['confluence_score'].std() / (df['confluence_score'].mean() + 1e-6))
        signal_consistency = max(0.0, min(1.0, signal_consistency))
        
        return SignalPerformanceMetrics(
            symbol=symbol,
            timeframe=timeframe or TimeFrame.M15,
            total_signals=total_signals,
            accuracy=avg_accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            avg_confluence_score=avg_confluence,
            confidence_correlation=confidence_correlation,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            signal_consistency=signal_consistency
        )
    
    def calculate_execution_performance_metrics(self, symbol: str,
                                              execution_method: Optional[str] = None,
                                              lookback_days: int = 30) -> ExecutionPerformanceMetrics:
        """Calculate execution performance metrics"""
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM execution_performance 
                    WHERE symbol = ? AND execution_time >= ?
                '''
                params = [symbol, cutoff_date]
                
                if execution_method:
                    query += ' AND execution_method = ?'
                    params.append(execution_method)
                
                df = pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            self.logger.error(f"Error querying execution performance: {e}")
            df = pd.DataFrame()
        
        if df.empty:
            return ExecutionPerformanceMetrics(
                symbol=symbol,
                execution_method=execution_method or 'unknown',
                total_executions=0,
                avg_slippage_bps=0.0,
                fill_rate=0.0,
                timing_efficiency=0.0,
                cost_effectiveness=0.0,
                market_impact=0.0,
                latency_ms=0.0
            )
        
        # Calculate metrics
        total_executions = len(df)
        avg_slippage = df['slippage_bps'].mean()
        avg_fill_rate = df['fill_rate'].mean()
        avg_market_impact = df['market_impact'].mean()
        avg_latency = df['latency_ms'].mean()
        
        # Timing efficiency (inverse of latency, normalized)
        timing_efficiency = max(0.0, 1.0 - (avg_latency / 1000))  # 1000ms = 0 efficiency
        
        # Cost effectiveness (inverse of slippage + impact)
        total_cost = avg_slippage + avg_market_impact
        cost_effectiveness = max(0.0, 1.0 - (total_cost / 10))  # 10 bps = 0 effectiveness
        
        return ExecutionPerformanceMetrics(
            symbol=symbol,
            execution_method=execution_method or 'unknown',
            total_executions=total_executions,
            avg_slippage_bps=avg_slippage,
            fill_rate=avg_fill_rate,
            timing_efficiency=timing_efficiency,
            cost_effectiveness=cost_effectiveness,
            market_impact=avg_market_impact,
            latency_ms=avg_latency
        )
    
    def calculate_risk_metrics(self, symbol: str, lookback_days: int = 30) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        # Get recent trade results
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_trades = [tr for tr in self.trade_results 
                        if tr.symbol == symbol and tr.exit_time >= cutoff_date]
        
        if not recent_trades or len(recent_trades) < 10:
            return RiskMetrics(
                var_95=0.0, cvar_95=0.0, max_drawdown=0.0, volatility=0.0,
                skewness=0.0, kurtosis=0.0, tail_ratio=0.0, downside_deviation=0.0
            )
        
        returns = np.array([tr.pnl_pct for tr in recent_trades])
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        tail_losses = returns[returns <= var_95]
        cvar_95 = np.mean(tail_losses) if len(tail_losses) > 0 else var_95
        
        # Volatility
        volatility = np.std(returns)
        
        # Higher moments
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Tail ratio
        upside_99 = np.percentile(returns, 99)
        downside_1 = np.percentile(returns, 1)
        tail_ratio = abs(upside_99 / downside_1) if downside_1 != 0 else 1.0
        
        # Downside deviation
        negative_returns = returns[returns < 0]
        downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0.0
        
        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(returns.tolist())
        
        return RiskMetrics(
            var_95=abs(var_95),
            cvar_95=abs(cvar_95),
            max_drawdown=max_drawdown,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio,
            downside_deviation=downside_deviation
        )
    
    def generate_optimization_suggestions(self, symbol: str) -> Dict[str, List[str]]:
        """Generate optimization suggestions based on performance analysis"""
        suggestions = {
            'signal_quality': [],
            'execution_efficiency': [],
            'risk_management': [],
            'general': []
        }
        
        # Analyze signal performance
        signal_metrics = self.calculate_signal_performance_metrics(symbol)
        
        if signal_metrics.accuracy < 0.75:
            suggestions['signal_quality'].append(
                f"Signal accuracy ({signal_metrics.accuracy:.2f}) below target. "
                "Consider adjusting confluence thresholds or timeframe weights."
            )
        
        if signal_metrics.confidence_correlation < 0.3:
            suggestions['signal_quality'].append(
                "Low correlation between signal confidence and actual outcomes. "
                "Review confluence calculation methodology."
            )
        
        if signal_metrics.false_positive_rate > 0.3:
            suggestions['signal_quality'].append(
                f"High false positive rate ({signal_metrics.false_positive_rate:.2f}). "
                "Consider stricter entry criteria or better trend filters."
            )
        
        # Analyze execution performance
        exec_metrics = self.calculate_execution_performance_metrics(symbol)
        
        if exec_metrics.avg_slippage_bps > 1.0:
            suggestions['execution_efficiency'].append(
                f"High execution slippage ({exec_metrics.avg_slippage_bps:.2f} bps). "
                "Consider using limit orders or TWAP/VWAP strategies."
            )
        
        if exec_metrics.timing_efficiency < 0.7:
            suggestions['execution_efficiency'].append(
                "Low timing efficiency. Consider reducing order size or "
                "improving market timing algorithms."
            )
        
        # Analyze risk metrics
        risk_metrics = self.calculate_risk_metrics(symbol)
        
        if risk_metrics.max_drawdown > 0.05:
            suggestions['risk_management'].append(
                f"High maximum drawdown ({risk_metrics.max_drawdown:.2%}). "
                "Implement tighter stop losses or position sizing controls."
            )
        
        if risk_metrics.var_95 > 0.02:
            suggestions['risk_management'].append(
                f"High daily VaR ({risk_metrics.var_95:.2%}). "
                "Consider reducing position sizes or improving risk controls."
            )
        
        # Live metrics analysis
        if symbol in self.live_metrics:
            live = self.live_metrics[symbol]
            
            if live['win_rate'] < 0.65:
                suggestions['general'].append(
                    f"Win rate ({live['win_rate']:.2%}) below target. "
                    "Review overall strategy effectiveness."
                )
            
            if live['sharpe_ratio'] < 3.0:
                suggestions['general'].append(
                    f"Sharpe ratio ({live['sharpe_ratio']:.2f}) below target. "
                    "Focus on risk-adjusted returns optimization."
                )
        
        return suggestions
    
    def generate_performance_report(self, symbol: Optional[str] = None,
                                  lookback_days: int = 30) -> Dict:
        """Generate comprehensive performance report"""
        
        if symbol:
            symbols = [symbol]
        else:
            # Get all symbols with recent activity
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            symbols = list(set(tr.symbol for tr in self.trade_results 
                             if tr.exit_time >= cutoff_date))
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'lookback_days': lookback_days,
                'symbols_analyzed': len(symbols),
                'total_trades': len([tr for tr in self.trade_results 
                                   if tr.exit_time >= cutoff_date])
            },
            'summary': {},
            'symbol_analysis': {},
            'optimization_suggestions': {},
            'alerts': {}
        }
        
        # Overall summary
        all_returns = []
        all_accuracies = []
        all_slippages = []
        
        for sym in symbols:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            sym_trades = [tr for tr in self.trade_results 
                         if tr.symbol == sym and tr.exit_time >= cutoff_date]
            
            if sym_trades:
                all_returns.extend([tr.pnl_pct for tr in sym_trades])
                all_accuracies.extend([tr.signal_accuracy for tr in sym_trades])
                all_slippages.extend([tr.slippage_bps for tr in sym_trades])
        
        if all_returns:
            report['summary'] = {
                'total_return': sum(all_returns),
                'avg_return_per_trade': np.mean(all_returns),
                'win_rate': np.mean([r > 0 for r in all_returns]),
                'sharpe_ratio': self._calculate_sharpe_ratio(all_returns),
                'max_drawdown': self._calculate_max_drawdown(all_returns),
                'avg_signal_accuracy': np.mean(all_accuracies),
                'avg_execution_slippage': np.mean(all_slippages),
                'profit_factor': (sum([r for r in all_returns if r > 0]) / 
                                abs(sum([r for r in all_returns if r < 0]))) 
                                if any(r < 0 for r in all_returns) else float('inf')
            }
        
        # Per-symbol analysis
        for sym in symbols:
            signal_metrics = self.calculate_signal_performance_metrics(sym, lookback_days=lookback_days)
            exec_metrics = self.calculate_execution_performance_metrics(sym, lookback_days=lookback_days)
            risk_metrics = self.calculate_risk_metrics(sym, lookback_days=lookback_days)
            
            report['symbol_analysis'][sym] = {
                'signal_performance': asdict(signal_metrics),
                'execution_performance': asdict(exec_metrics),
                'risk_metrics': asdict(risk_metrics),
                'live_metrics': self.live_metrics.get(sym, {})
            }
            
            # Optimization suggestions
            report['optimization_suggestions'][sym] = self.generate_optimization_suggestions(sym)
            
            # Recent alerts
            report['alerts'][sym] = self.optimization_suggestions[sym][-5:]  # Last 5 alerts
        
        return report
    
    def export_performance_data(self, output_path: str, format: str = 'json'):
        """Export performance data to file"""
        report = self.generate_performance_report()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_file.with_suffix('.json'), 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format.lower() == 'csv':
            # Export trade results as CSV
            trades_data = []
            for trade in self.trade_results:
                trades_data.append(asdict(trade))
            
            df = pd.DataFrame(trades_data)
            df.to_csv(output_file.with_suffix('.csv'), index=False)
        
        self.logger.info(f"Performance data exported to {output_file}")

# Factory function
def create_performance_tracker(config: Optional[Dict] = None, 
                             db_path: Optional[str] = None) -> MultitimeframePerformanceTracker:
    """Factory function to create performance tracker"""
    return MultitimeframePerformanceTracker(config, db_path)