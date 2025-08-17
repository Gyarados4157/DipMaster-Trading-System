#!/usr/bin/env python3
"""
Multi-Timeframe Strategy Orchestrator for DipMaster Enhanced System
多时间框架策略编排器 - DipMaster增强系统

This module serves as the master orchestrator that integrates all components of the
enhanced DipMaster strategy system, including:
- Multi-timeframe signal generation and coordination
- Market regime detection and adaptation
- Adaptive parameter optimization
- Risk management and position sizing
- Execution optimization and monitoring

The orchestrator ensures seamless integration between all subsystems while maintaining
the core DipMaster strategy principles and achieving target performance improvements.

Integration Features:
- Unified signal processing across all timeframes
- Regime-aware parameter adaptation
- Real-time performance monitoring and optimization
- Risk-adjusted position sizing and execution
- Comprehensive feedback loops for continuous improvement

Performance Targets:
- BTCUSDT Win Rate: 47.7% → 70%+
- Portfolio Sharpe Ratio: 3.65 → 4.0+
- Max Drawdown: Current → <3%
- Execution Quality: <0.5 bps slippage

Author: Strategy Orchestrator Agent
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
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
from pathlib import Path

# Core strategy components
from .multi_timeframe_signal_engine import (
    MultiTimeframeSignalEngine, MultitimeframeSignal, TimeFrame, ExecutionRecommendation,
    TrendAlignmentAnalyzer, ConfluenceCalculator, ExecutionOptimizer
)
from .market_regime_detector import MarketRegimeDetector, MarketRegime, RegimeSignal
from .adaptive_parameter_engine import AdaptiveParameterEngine, ParameterSet, OptimizationMethod
from .multitf_performance_tracker import (
    MultitimeframePerformanceTracker, TradeResult, SignalOutcome, PerformanceMetric
)
from .signal_detector import TradingSignal, SignalType
from .position_manager import PositionManager
from .risk_manager import RiskManager
from ..types.common_types import *

warnings.filterwarnings('ignore')

class StrategyState(Enum):
    """Overall strategy state"""
    ACTIVE = "active"
    PAUSED = "paused"
    OPTIMIZATION = "optimization"
    EMERGENCY_STOP = "emergency_stop"
    MAINTENANCE = "maintenance"

class DecisionPriority(Enum):
    """Decision priority levels"""
    EMERGENCY = "emergency"    # Immediate risk management
    HIGH = "high"             # Strong signal execution
    MEDIUM = "medium"         # Normal operations
    LOW = "low"               # Optimization and maintenance

@dataclass
class StrategyDecision:
    """Unified strategy decision output"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold', 'close', 'reduce'
    size: float
    confidence: float
    priority: DecisionPriority
    execution_method: str
    timeframe_signals: Dict[TimeFrame, Any]
    regime_context: MarketRegime
    risk_assessment: Dict[str, float]
    expected_holding_minutes: int
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: str
    timestamp: datetime

@dataclass
class PortfolioContext:
    """Current portfolio context for decision making"""
    total_value: float
    available_cash: float
    positions: Dict[str, Dict]
    correlation_matrix: pd.DataFrame
    risk_metrics: Dict[str, float]
    regime_distribution: Dict[MarketRegime, float]
    performance_attribution: Dict[str, float]

class MultitimeframeStrategyOrchestrator:
    """
    Master Strategy Orchestrator
    主策略编排器
    
    Coordinates all components of the enhanced DipMaster strategy system to generate
    unified trading decisions with optimal risk-reward characteristics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Initialize core components
        self.signal_engine = MultiTimeframeSignalEngine(self.config.get('signal_engine', {}))
        self.regime_detector = MarketRegimeDetector(self.config.get('regime_detector', {}))
        self.parameter_engine = AdaptiveParameterEngine(self.config.get('parameter_engine', {}))
        self.performance_tracker = MultitimeframePerformanceTracker(
            self.config.get('performance_tracker', {}),
            self.config.get('db_path', 'data/strategy_performance.db')
        )
        
        # Strategy state management
        self.strategy_state = StrategyState.ACTIVE
        self.active_symbols = set()
        self.symbol_data_cache = {}
        self.decision_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Risk and position management
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.02)  # 2% daily VaR
        self.max_position_size = self.config.get('max_position_size', 0.15)    # 15% per position
        self.max_correlation = self.config.get('max_correlation', 0.7)         # 70% max correlation
        self.max_regime_exposure = self.config.get('max_regime_exposure', 0.6) # 60% per regime
        
        # Performance monitoring
        self.performance_check_interval = timedelta(minutes=15)
        self.last_performance_check = datetime.now()
        self.optimization_in_progress = {}
        
        # Thread safety
        self._lock = threading.Lock()
        self._decision_lock = threading.Lock()
        
        # Integration state
        self.regime_cache = {}
        self.parameter_cache = {}
        self.performance_cache = {}
        
        self.logger.info("MultitimeframeStrategyOrchestrator initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Default configuration for the strategy orchestrator"""
        return {
            'decision_framework': {
                'min_confluence_score': 0.4,
                'regime_confidence_threshold': 0.7,
                'parameter_optimization_interval_trades': 100,
                'performance_review_interval_minutes': 15,
                'emergency_stop_drawdown': 0.10  # 10% portfolio drawdown
            },
            'risk_management': {
                'max_positions': 10,
                'max_position_size_pct': 15.0,
                'max_portfolio_var': 2.0,
                'max_correlation': 0.7,
                'position_sizing_method': 'risk_parity',
                'stop_loss_method': 'atr_based',
                'take_profit_method': 'regime_adaptive'
            },
            'execution_optimization': {
                'default_execution_method': 'limit',
                'urgency_threshold_market': 0.9,
                'urgency_threshold_twap': 0.6,
                'slippage_tolerance_bps': 2.0,
                'partial_fill_threshold': 0.8
            },
            'performance_targets': {
                'min_win_rate': 0.65,
                'min_sharpe_ratio': 3.5,
                'max_drawdown': 0.03,
                'min_signal_accuracy': 0.80,
                'max_execution_slippage': 0.5
            },
            'symbols': {
                'primary': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
                'secondary': ['ADAUSDT', 'LINKUSDT', 'MATICUSDT', 'AVAXUSDT'],
                'watchlist': ['DOGEUSDT', 'XRPUSDT', 'BNBUSDT']
            }
        }
    
    def add_symbol(self, symbol: str, primary_timeframes: List[TimeFrame] = None):
        """Add symbol to active trading universe"""
        if primary_timeframes is None:
            primary_timeframes = [TimeFrame.H1, TimeFrame.M15, TimeFrame.M5, TimeFrame.M1]
        
        with self._lock:
            self.active_symbols.add(symbol)
            self.symbol_data_cache[symbol] = {
                'timeframes': primary_timeframes,
                'last_update': {},
                'regime_history': deque(maxlen=100),
                'parameter_history': deque(maxlen=100)
            }
        
        self.logger.info(f"Added {symbol} to active trading universe with timeframes: {[tf.value for tf in primary_timeframes]}")
    
    def update_market_data(self, symbol: str, timeframe: TimeFrame, data: pd.DataFrame):
        """Update market data for symbol and timeframe"""
        if symbol not in self.active_symbols:
            self.logger.warning(f"Received data for inactive symbol: {symbol}")
            return
        
        # Update signal engine
        self.signal_engine.update_market_data(symbol, timeframe, data)
        
        # Update cache
        with self._lock:
            if symbol in self.symbol_data_cache:
                self.symbol_data_cache[symbol]['last_update'][timeframe] = datetime.now()
    
    def generate_strategy_decision(self, symbol: str, 
                                 portfolio_context: Optional[PortfolioContext] = None) -> Optional[StrategyDecision]:
        """
        Generate comprehensive strategy decision for a symbol
        为符号生成综合策略决策
        """
        if symbol not in self.active_symbols:
            return None
        
        if self.strategy_state != StrategyState.ACTIVE:
            self.logger.info(f"Strategy not active, current state: {self.strategy_state}")
            return None
        
        try:
            with self._decision_lock:
                return self._generate_unified_decision(symbol, portfolio_context)
        except Exception as e:
            self.logger.error(f"Error generating strategy decision for {symbol}: {e}")
            return None
    
    def _generate_unified_decision(self, symbol: str, 
                                 portfolio_context: Optional[PortfolioContext]) -> Optional[StrategyDecision]:
        """Generate unified strategy decision integrating all components"""
        
        # Step 1: Get multi-timeframe signal
        mtf_signal = self.signal_engine.generate_multitimeframe_signal(symbol)
        if not mtf_signal:
            self.logger.debug(f"No multi-timeframe signal available for {symbol}")
            return None
        
        # Step 2: Detect market regime
        regime_signal = self._get_current_regime(symbol)
        if not regime_signal:
            self.logger.debug(f"No regime signal available for {symbol}")
            return None
        
        # Step 3: Get adaptive parameters
        adaptive_params = self._get_adaptive_parameters(symbol, regime_signal.regime)
        
        # Step 4: Risk assessment and position sizing
        risk_assessment = self._assess_comprehensive_risk(
            symbol, mtf_signal, regime_signal, portfolio_context
        )
        
        # Step 5: Make unified decision
        decision = self._make_unified_decision(
            symbol, mtf_signal, regime_signal, adaptive_params, 
            risk_assessment, portfolio_context
        )
        
        # Step 6: Store decision for tracking
        if decision:
            with self._lock:
                self.decision_history[symbol].append(decision)
        
        return decision
    
    def _get_current_regime(self, symbol: str) -> Optional[RegimeSignal]:
        """Get current market regime for symbol"""
        # Check cache first
        cache_key = f"{symbol}_regime"
        if cache_key in self.regime_cache:
            cached_regime, timestamp = self.regime_cache[cache_key]
            if datetime.now() - timestamp < timedelta(minutes=5):  # 5-minute cache
                return cached_regime
        
        # Get fresh regime detection
        if symbol not in self.signal_engine.market_data_cache:
            return None
        
        if TimeFrame.M15 not in self.signal_engine.market_data_cache[symbol]:
            return None
        
        data = self.signal_engine.market_data_cache[symbol][TimeFrame.M15]
        regime_signal = self.regime_detector.identify_regime(data, symbol)
        
        # Update cache
        self.regime_cache[cache_key] = (regime_signal, datetime.now())
        
        return regime_signal
    
    def _get_adaptive_parameters(self, symbol: str, regime: MarketRegime) -> ParameterSet:
        """Get adaptive parameters for symbol and regime"""
        # Check cache
        cache_key = f"{symbol}_{regime.value}_params"
        if cache_key in self.parameter_cache:
            cached_params, timestamp = self.parameter_cache[cache_key]
            if datetime.now() - timestamp < timedelta(hours=1):  # 1-hour cache
                return cached_params
        
        # Get current parameters
        if symbol in self.signal_engine.market_data_cache and TimeFrame.M15 in self.signal_engine.market_data_cache[symbol]:
            data = self.signal_engine.market_data_cache[symbol][TimeFrame.M15]
            params = self.parameter_engine.get_current_parameters(symbol, regime, data)
        else:
            # Fallback to regime detector params
            regime_params = self.regime_detector.get_adaptive_parameters(regime, symbol)
            params = ParameterSet(
                rsi_low=regime_params['rsi_low'],
                rsi_high=regime_params['rsi_high'],
                dip_threshold=regime_params['dip_threshold'],
                volume_threshold=regime_params['volume_threshold'],
                target_profit=regime_params['target_profit'],
                stop_loss=regime_params['stop_loss'],
                max_holding_minutes=regime_params['max_holding_minutes'],
                position_size_multiplier=1.0,
                confidence_multiplier=regime_params['confidence_multiplier'],
                correlation_penalty=0.5,
                regime=regime,
                symbol=symbol,
                confidence_score=0.8,
                timestamp=datetime.now()
            )
        
        # Update cache
        self.parameter_cache[cache_key] = (params, datetime.now())
        
        return params
    
    def _assess_comprehensive_risk(self, symbol: str, mtf_signal: MultitimeframeSignal,
                                 regime_signal: RegimeSignal,
                                 portfolio_context: Optional[PortfolioContext]) -> Dict[str, float]:
        """Assess comprehensive risk factors"""
        risk_factors = {}
        
        # Signal-based risk
        signal_risk = 1.0 - abs(mtf_signal.confluence_score)  # Lower confluence = higher risk
        risk_factors['signal_risk'] = signal_risk
        
        # Regime-based risk
        regime_risk_map = {
            MarketRegime.RANGE_BOUND: 0.2,
            MarketRegime.STRONG_UPTREND: 0.3,
            MarketRegime.STRONG_DOWNTREND: 0.8,
            MarketRegime.HIGH_VOLATILITY: 0.9,
            MarketRegime.LOW_VOLATILITY: 0.1,
            MarketRegime.TRANSITION: 0.7
        }
        regime_risk = regime_risk_map.get(regime_signal.regime, 0.5)
        risk_factors['regime_risk'] = regime_risk
        
        # Timeframe alignment risk
        timeframe_signals = mtf_signal.timeframe_signals
        if len(timeframe_signals) >= 2:
            trend_directions = []
            for tf_signal in timeframe_signals.values():
                if hasattr(tf_signal, 'trend_direction'):
                    trend_directions.append(tf_signal.trend_direction.value)
            
            # Calculate agreement
            if trend_directions:
                unique_directions = set(trend_directions)
                alignment_risk = (len(unique_directions) - 1) / max(len(trend_directions) - 1, 1)
                risk_factors['alignment_risk'] = alignment_risk
            else:
                risk_factors['alignment_risk'] = 0.5
        else:
            risk_factors['alignment_risk'] = 0.8  # High risk if insufficient timeframes
        
        # Portfolio context risk
        if portfolio_context:
            # Correlation risk
            if symbol in portfolio_context.correlation_matrix.columns:
                max_correlation = 0.0
                for pos_symbol in portfolio_context.positions:
                    if pos_symbol != symbol and pos_symbol in portfolio_context.correlation_matrix.columns:
                        correlation = abs(portfolio_context.correlation_matrix.loc[symbol, pos_symbol])
                        max_correlation = max(max_correlation, correlation)
                
                correlation_risk = max_correlation if max_correlation > self.max_correlation else 0.0
                risk_factors['correlation_risk'] = correlation_risk
            
            # Concentration risk
            if portfolio_context.total_value > 0:
                position_concentration = sum(
                    pos.get('value', 0) for pos in portfolio_context.positions.values()
                ) / portfolio_context.total_value
                
                concentration_risk = max(0.0, position_concentration - 0.8)  # Risk above 80% invested
                risk_factors['concentration_risk'] = concentration_risk
            
            # Portfolio VaR risk
            portfolio_var = portfolio_context.risk_metrics.get('var_95', 0.0)
            var_risk = max(0.0, portfolio_var - self.max_portfolio_risk) / self.max_portfolio_risk
            risk_factors['portfolio_var_risk'] = var_risk
        
        # Execution risk (from signal engine)
        execution_urgency = mtf_signal.execution_urgency
        execution_risk = execution_urgency  # Higher urgency = higher execution risk
        risk_factors['execution_risk'] = execution_risk
        
        # Overall risk score
        weights = {
            'signal_risk': 0.25,
            'regime_risk': 0.20,
            'alignment_risk': 0.15,
            'correlation_risk': 0.15,
            'concentration_risk': 0.10,
            'portfolio_var_risk': 0.10,
            'execution_risk': 0.05
        }
        
        overall_risk = sum(
            risk_factors.get(factor, 0.0) * weight 
            for factor, weight in weights.items()
        )
        risk_factors['overall_risk'] = min(1.0, overall_risk)
        
        return risk_factors
    
    def _make_unified_decision(self, symbol: str, mtf_signal: MultitimeframeSignal,
                             regime_signal: RegimeSignal, adaptive_params: ParameterSet,
                             risk_assessment: Dict[str, float],
                             portfolio_context: Optional[PortfolioContext]) -> Optional[StrategyDecision]:
        """Make final unified trading decision"""
        
        # Check if trading should be paused for this regime
        if not self.regime_detector.should_trade_in_regime(regime_signal.regime, regime_signal.confidence):
            return StrategyDecision(
                symbol=symbol,
                action='hold',
                size=0.0,
                confidence=0.0,
                priority=DecisionPriority.LOW,
                execution_method='none',
                timeframe_signals=mtf_signal.timeframe_signals,
                regime_context=regime_signal.regime,
                risk_assessment=risk_assessment,
                expected_holding_minutes=0,
                stop_loss=None,
                take_profit=None,
                reasoning="Trading paused for current market regime",
                timestamp=datetime.now()
            )
        
        # Check minimum confluence threshold
        min_confluence = self.config['decision_framework']['min_confluence_score']
        if abs(mtf_signal.confluence_score) < min_confluence:
            return StrategyDecision(
                symbol=symbol,
                action='hold',
                size=0.0,
                confidence=0.0,
                priority=DecisionPriority.LOW,
                execution_method='none',
                timeframe_signals=mtf_signal.timeframe_signals,
                regime_context=regime_signal.regime,
                risk_assessment=risk_assessment,
                expected_holding_minutes=0,
                stop_loss=None,
                take_profit=None,
                reasoning=f"Confluence score {mtf_signal.confluence_score:.2f} below minimum {min_confluence}",
                timestamp=datetime.now()
            )
        
        # Risk-based position sizing
        base_size = self._calculate_position_size(
            symbol, mtf_signal, adaptive_params, risk_assessment, portfolio_context
        )
        
        # Determine action and priority
        action, priority = self._determine_action_priority(mtf_signal, risk_assessment)
        
        # Calculate confidence
        confidence = self._calculate_decision_confidence(
            mtf_signal, regime_signal, adaptive_params, risk_assessment
        )
        
        # Get execution recommendation
        exec_rec = mtf_signal.recommended_action
        execution_method = exec_rec.get('entry_method', 'limit')
        
        # Risk levels
        stop_loss = exec_rec.get('stop_loss_price')
        take_profit = exec_rec.get('take_profit_price')
        
        # Reasoning
        reasoning = self._generate_decision_reasoning(
            mtf_signal, regime_signal, adaptive_params, risk_assessment
        )
        
        return StrategyDecision(
            symbol=symbol,
            action=action,
            size=base_size,
            confidence=confidence,
            priority=priority,
            execution_method=execution_method,
            timeframe_signals=mtf_signal.timeframe_signals,
            regime_context=regime_signal.regime,
            risk_assessment=risk_assessment,
            expected_holding_minutes=mtf_signal.expected_holding_period,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
    
    def _calculate_position_size(self, symbol: str, mtf_signal: MultitimeframeSignal,
                               adaptive_params: ParameterSet, risk_assessment: Dict[str, float],
                               portfolio_context: Optional[PortfolioContext]) -> float:
        """Calculate optimal position size"""
        
        # Base size from adaptive parameters
        base_multiplier = adaptive_params.position_size_multiplier
        
        # Confidence adjustment
        confidence_adjustment = abs(mtf_signal.confluence_score)
        
        # Risk adjustment
        overall_risk = risk_assessment.get('overall_risk', 0.5)
        risk_adjustment = max(0.1, 1.0 - overall_risk)
        
        # Regime adjustment
        regime_adjustments = {
            MarketRegime.RANGE_BOUND: 1.0,
            MarketRegime.STRONG_UPTREND: 0.8,
            MarketRegime.STRONG_DOWNTREND: 0.3,
            MarketRegime.HIGH_VOLATILITY: 0.4,
            MarketRegime.LOW_VOLATILITY: 1.2,
            MarketRegime.TRANSITION: 0.5
        }
        regime_adjustment = regime_adjustments.get(adaptive_params.regime, 1.0)
        
        # Portfolio context adjustment
        portfolio_adjustment = 1.0
        if portfolio_context:
            # Reduce size if portfolio is highly concentrated
            concentration = len(portfolio_context.positions) / max(self.config['risk_management']['max_positions'], 1)
            if concentration > 0.8:
                portfolio_adjustment *= 0.7
            
            # Reduce size if available cash is low
            cash_ratio = portfolio_context.available_cash / max(portfolio_context.total_value, 1)
            if cash_ratio < 0.2:  # Less than 20% cash
                portfolio_adjustment *= 0.6
        
        # Calculate final size
        final_size = (base_multiplier * confidence_adjustment * risk_adjustment * 
                     regime_adjustment * portfolio_adjustment)
        
        # Apply absolute limits
        max_position_size = self.config['risk_management']['max_position_size_pct'] / 100
        final_size = min(final_size, max_position_size)
        final_size = max(final_size, 0.0)
        
        return final_size
    
    def _determine_action_priority(self, mtf_signal: MultitimeframeSignal,
                                 risk_assessment: Dict[str, float]) -> Tuple[str, DecisionPriority]:
        """Determine action and priority level"""
        
        confluence = mtf_signal.confluence_score
        overall_risk = risk_assessment.get('overall_risk', 0.5)
        
        # Emergency situations
        if overall_risk > 0.8:
            return 'hold', DecisionPriority.EMERGENCY
        
        # Strong signals
        if abs(confluence) > 0.8:
            if confluence > 0:
                return 'buy', DecisionPriority.HIGH
            else:
                return 'sell', DecisionPriority.HIGH
        
        # Medium signals
        elif abs(confluence) > 0.6:
            if confluence > 0:
                return 'buy', DecisionPriority.MEDIUM
            else:
                return 'sell', DecisionPriority.MEDIUM
        
        # Weak signals
        elif abs(confluence) > 0.4:
            if confluence > 0:
                return 'buy', DecisionPriority.LOW
            else:
                return 'sell', DecisionPriority.LOW
        
        # No clear signal
        else:
            return 'hold', DecisionPriority.LOW
    
    def _calculate_decision_confidence(self, mtf_signal: MultitimeframeSignal,
                                     regime_signal: RegimeSignal, adaptive_params: ParameterSet,
                                     risk_assessment: Dict[str, float]) -> float:
        """Calculate overall decision confidence"""
        
        factors = []
        
        # Signal confluence confidence
        factors.append(abs(mtf_signal.confluence_score))
        
        # Regime confidence
        factors.append(regime_signal.confidence)
        
        # Parameter confidence
        factors.append(adaptive_params.confidence_score)
        
        # Risk confidence (inverse of risk)
        risk_confidence = 1.0 - risk_assessment.get('overall_risk', 0.5)
        factors.append(risk_confidence)
        
        # Timeframe agreement confidence
        if len(mtf_signal.timeframe_signals) >= 3:
            alignment_risk = risk_assessment.get('alignment_risk', 0.5)
            alignment_confidence = 1.0 - alignment_risk
            factors.append(alignment_confidence)
        
        return np.mean(factors)
    
    def _generate_decision_reasoning(self, mtf_signal: MultitimeframeSignal,
                                   regime_signal: RegimeSignal, adaptive_params: ParameterSet,
                                   risk_assessment: Dict[str, float]) -> str:
        """Generate human-readable decision reasoning"""
        
        reasoning_parts = []
        
        # Signal analysis
        confluence = mtf_signal.confluence_score
        if abs(confluence) > 0.8:
            direction = "bullish" if confluence > 0 else "bearish"
            reasoning_parts.append(f"Strong {direction} signal (confluence: {confluence:.2f})")
        elif abs(confluence) > 0.6:
            direction = "bullish" if confluence > 0 else "bearish"
            reasoning_parts.append(f"Moderate {direction} signal (confluence: {confluence:.2f})")
        else:
            reasoning_parts.append(f"Weak signal (confluence: {confluence:.2f})")
        
        # Regime context
        regime_desc = self.regime_detector.get_regime_description(regime_signal.regime)
        reasoning_parts.append(f"Market regime: {regime_desc}")
        
        # Risk factors
        overall_risk = risk_assessment.get('overall_risk', 0.5)
        if overall_risk > 0.7:
            reasoning_parts.append(f"High risk environment ({overall_risk:.2f})")
        elif overall_risk < 0.3:
            reasoning_parts.append(f"Low risk environment ({overall_risk:.2f})")
        
        # Key risk factors
        high_risk_factors = [
            factor for factor, value in risk_assessment.items() 
            if value > 0.6 and factor != 'overall_risk'
        ]
        if high_risk_factors:
            reasoning_parts.append(f"Risk factors: {', '.join(high_risk_factors)}")
        
        # Timeframe analysis
        timeframe_count = len(mtf_signal.timeframe_signals)
        reasoning_parts.append(f"Analysis across {timeframe_count} timeframes")
        
        return "; ".join(reasoning_parts)
    
    def process_trade_completion(self, trade_result: TradeResult):
        """Process completed trade and update all systems"""
        
        # Record in performance tracker
        self.performance_tracker.record_trade_result(trade_result)
        
        # Update parameter engine with performance
        regime_signal = self._get_current_regime(trade_result.symbol)
        if regime_signal:
            trade_performance = {
                'pnl_pct': trade_result.pnl_pct,
                'holding_minutes': trade_result.holding_period_minutes,
                'signal_accuracy': trade_result.signal_accuracy,
                'execution_quality': trade_result.execution_quality,
                'timestamp': trade_result.exit_time
            }
            self.parameter_engine.update_performance(
                trade_result.symbol, regime_signal.regime, trade_performance
            )
        
        # Check for optimization triggers
        self._check_optimization_triggers(trade_result.symbol)
        
        self.logger.info(f"Processed trade completion: {trade_result.trade_id} - "
                        f"PnL: {trade_result.pnl_pct:.2%}")
    
    def _check_optimization_triggers(self, symbol: str):
        """Check if optimization should be triggered"""
        
        # Get recent performance
        recent_trades = [
            tr for tr in self.performance_tracker.trade_results
            if tr.symbol == symbol and 
            tr.exit_time >= datetime.now() - timedelta(days=7)
        ]
        
        if len(recent_trades) < 20:  # Need minimum sample size
            return
        
        # Calculate recent metrics
        recent_returns = [tr.pnl_pct for tr in recent_trades]
        recent_win_rate = np.mean([r > 0 for r in recent_returns])
        recent_sharpe = self.performance_tracker._calculate_sharpe_ratio(recent_returns)
        
        # Check targets
        targets = self.config['performance_targets']
        optimization_needed = False
        
        if recent_win_rate < targets['min_win_rate']:
            optimization_needed = True
            self.logger.warning(f"Win rate below target for {symbol}: {recent_win_rate:.2%}")
        
        if recent_sharpe < targets['min_sharpe_ratio']:
            optimization_needed = True
            self.logger.warning(f"Sharpe ratio below target for {symbol}: {recent_sharpe:.2f}")
        
        # Trigger optimization if needed
        if optimization_needed and symbol not in self.optimization_in_progress:
            self.logger.info(f"Triggering parameter optimization for {symbol}")
            asyncio.create_task(self._async_optimize_symbol(symbol))
    
    async def _async_optimize_symbol(self, symbol: str):
        """Asynchronously optimize parameters for symbol"""
        self.optimization_in_progress[symbol] = datetime.now()
        
        try:
            # Get recent performance data
            recent_trades = [
                {
                    'pnl_pct': tr.pnl_pct,
                    'holding_minutes': tr.holding_period_minutes,
                    'signal_accuracy': tr.signal_accuracy,
                    'execution_quality': tr.execution_quality,
                    'timestamp': tr.exit_time
                }
                for tr in self.performance_tracker.trade_results
                if tr.symbol == symbol and 
                tr.exit_time >= datetime.now() - timedelta(days=30)
            ]
            
            # Get current regime
            regime_signal = self._get_current_regime(symbol)
            if not regime_signal:
                return
            
            # Run optimization
            optimization_result = await asyncio.get_event_loop().run_in_executor(
                ThreadPoolExecutor(),
                self.parameter_engine.optimize_parameters,
                symbol, regime_signal.regime, recent_trades, OptimizationMethod.BAYESIAN, True
            )
            
            self.logger.info(f"Parameter optimization completed for {symbol}: "
                           f"objective={optimization_result.objective_value:.4f}")
            
        except Exception as e:
            self.logger.error(f"Parameter optimization failed for {symbol}: {e}")
        
        finally:
            if symbol in self.optimization_in_progress:
                del self.optimization_in_progress[symbol]
    
    def get_strategy_status(self) -> Dict:
        """Get comprehensive strategy status"""
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'strategy_state': self.strategy_state.value,
            'active_symbols': list(self.active_symbols),
            'optimization_in_progress': list(self.optimization_in_progress.keys()),
            'performance_summary': {},
            'risk_summary': {},
            'recent_decisions': {}
        }
        
        # Performance summary
        if self.performance_tracker.trade_results:
            recent_trades = [
                tr for tr in self.performance_tracker.trade_results
                if tr.exit_time >= datetime.now() - timedelta(days=1)
            ]
            
            if recent_trades:
                returns = [tr.pnl_pct for tr in recent_trades]
                status['performance_summary'] = {
                    'total_trades_24h': len(recent_trades),
                    'win_rate_24h': np.mean([r > 0 for r in returns]),
                    'avg_return_24h': np.mean(returns),
                    'total_pnl_24h': sum([tr.pnl for tr in recent_trades]),
                    'best_trade_24h': max(returns) if returns else 0,
                    'worst_trade_24h': min(returns) if returns else 0
                }
        
        # Recent decisions per symbol
        for symbol in self.active_symbols:
            if symbol in self.decision_history and self.decision_history[symbol]:
                latest_decision = self.decision_history[symbol][-1]
                status['recent_decisions'][symbol] = {
                    'action': latest_decision.action,
                    'confidence': latest_decision.confidence,
                    'timestamp': latest_decision.timestamp.isoformat(),
                    'reasoning': latest_decision.reasoning[:100] + "..." if len(latest_decision.reasoning) > 100 else latest_decision.reasoning
                }
        
        return status
    
    def emergency_stop(self, reason: str = "Manual emergency stop"):
        """Emergency stop all trading activities"""
        with self._lock:
            previous_state = self.strategy_state
            self.strategy_state = StrategyState.EMERGENCY_STOP
        
        self.logger.critical(f"EMERGENCY STOP activated. Reason: {reason}. Previous state: {previous_state.value}")
        
        # TODO: Close all positions, cancel orders, etc.
        # This would integrate with the actual trading system
    
    def resume_trading(self):
        """Resume trading after emergency stop or pause"""
        with self._lock:
            if self.strategy_state in [StrategyState.EMERGENCY_STOP, StrategyState.PAUSED]:
                self.strategy_state = StrategyState.ACTIVE
                self.logger.info("Trading resumed - strategy state set to ACTIVE")
            else:
                self.logger.warning(f"Cannot resume trading from current state: {self.strategy_state}")
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive strategy performance report"""
        
        # Get performance report from tracker
        performance_report = self.performance_tracker.generate_performance_report()
        
        # Add strategy-specific metrics
        strategy_metrics = {
            'integration_metrics': {
                'cache_hit_rates': {
                    'regime_cache': len(self.regime_cache),
                    'parameter_cache': len(self.parameter_cache)
                },
                'decision_generation_rate': {},
                'optimization_efficiency': {}
            },
            'system_health': {
                'strategy_state': self.strategy_state.value,
                'active_symbols': len(self.active_symbols),
                'optimization_queue': len(self.optimization_in_progress),
                'last_performance_check': self.last_performance_check.isoformat()
            }
        }
        
        # Decision generation rates
        for symbol in self.active_symbols:
            if symbol in self.decision_history:
                recent_decisions = [
                    d for d in self.decision_history[symbol]
                    if d.timestamp >= datetime.now() - timedelta(hours=24)
                ]
                strategy_metrics['integration_metrics']['decision_generation_rate'][symbol] = len(recent_decisions)
        
        # Combine reports
        comprehensive_report = {
            'performance_report': performance_report,
            'strategy_metrics': strategy_metrics,
            'configuration': self.config,
            'generated_at': datetime.now().isoformat()
        }
        
        return comprehensive_report

# Factory function
def create_strategy_orchestrator(config: Optional[Dict] = None) -> MultitimeframeStrategyOrchestrator:
    """Factory function to create strategy orchestrator"""
    return MultitimeframeStrategyOrchestrator(config)