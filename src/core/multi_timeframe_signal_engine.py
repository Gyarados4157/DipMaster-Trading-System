#!/usr/bin/env python3
"""
Multi-Timeframe Signal Integration Engine for DipMaster Strategy
多时间框架信号整合引擎 - DipMaster策略专用

This module implements a sophisticated hierarchical signal system that coordinates
trading signals across multiple timeframes (1H, 15M, 5M, 1M) to achieve optimal
execution timing and improved win rates. The system addresses the critical need
for better signal confluence and execution quality in the DipMaster strategy.

Key Features:
- Hierarchical signal coordination with timeframe-specific weights
- Signal confluence scoring for execution confidence
- Smart order timing and sizing optimization
- Integration with market regime detection and adaptive parameters
- Real-time performance tracking and feedback optimization

Target Performance Improvements:
- BTCUSDT Win Rate: 47.7% → 70%+
- Sharpe Ratio: 3.65 → 4.0+
- Execution Quality: <0.5 bps slippage
- Signal Accuracy: +15% improvement via confluence

Author: Execution Microstructure OMS Agent
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
import talib
import ta

# Core strategy components
from .market_regime_detector import MarketRegimeDetector, MarketRegime, RegimeSignal
from .adaptive_parameter_engine import AdaptiveParameterEngine, ParameterSet
from .signal_detector import TradingSignal, SignalType
from ..types.common_types import *

warnings.filterwarnings('ignore')

class TimeFrame(Enum):
    """Supported timeframes for multi-TF analysis"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class SignalStrength(Enum):
    """Signal strength classifications"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    NEUTRAL = "neutral"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class TrendDirection(Enum):
    """Trend direction classifications"""
    STRONG_BEARISH = "strong_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    STRONG_BULLISH = "strong_bullish"

@dataclass
class TimeFrameSignal:
    """Signal data for a specific timeframe"""
    timeframe: TimeFrame
    symbol: str
    timestamp: datetime
    trend_direction: TrendDirection
    trend_strength: float  # 0-1
    momentum: float  # -1 to 1
    volatility: float  # 0-1
    volume_profile: float  # 0-1
    support_resistance: Dict[str, float]
    technical_indicators: Dict[str, float]
    signal_confidence: float  # 0-1
    dip_quality: float  # 0-1 (for DipMaster specific)
    exit_timing_score: float  # 0-1

@dataclass
class MultitimeframeSignal:
    """Comprehensive multi-timeframe signal"""
    symbol: str
    timestamp: datetime
    timeframe_signals: Dict[TimeFrame, TimeFrameSignal]
    confluence_score: float  # 0-1
    execution_signal: str  # STRONG_BUY, BUY, WEAK_BUY, HOLD, WEAK_SELL, SELL, STRONG_SELL
    recommended_action: Dict[str, Any]
    risk_assessment: Dict[str, float]
    execution_urgency: float  # 0-1
    expected_holding_period: int  # minutes

@dataclass
class ExecutionRecommendation:
    """Optimal execution strategy recommendation"""
    action: str  # buy/sell/hold
    size_multiplier: float  # 0.1 - 2.0
    entry_method: str  # market/limit/twap/vwap
    urgency: float  # 0-1
    max_slippage_bps: float
    time_horizon_minutes: int
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    confidence: float

class TrendAlignmentAnalyzer:
    """
    Multi-timeframe trend alignment analysis
    多时间框架趋势一致性分析器
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """Default configuration for trend analysis"""
        return {
            'trend_windows': {
                TimeFrame.H1: [24, 48, 96],  # 1-4 days
                TimeFrame.M15: [16, 32, 64],  # 4-16 hours
                TimeFrame.M5: [12, 24, 48],   # 1-4 hours
                TimeFrame.M1: [15, 30, 60]    # 15-60 minutes
            },
            'momentum_windows': {
                TimeFrame.H1: [14, 21],
                TimeFrame.M15: [14, 21],
                TimeFrame.M5: [14, 21],
                TimeFrame.M1: [14, 21]
            },
            'strength_thresholds': {
                'very_strong': 0.8,
                'strong': 0.6,
                'moderate': 0.4,
                'weak': 0.2
            }
        }
    
    def analyze_trend_direction(self, df: pd.DataFrame, timeframe: TimeFrame) -> TrendDirection:
        """Analyze trend direction for given timeframe"""
        if len(df) < 50:
            return TrendDirection.NEUTRAL
        
        # Multiple trend indicators
        windows = self.config['trend_windows'][timeframe]
        
        # EMA trends
        ema_short = df['close'].ewm(span=windows[0]).mean()
        ema_medium = df['close'].ewm(span=windows[1]).mean()
        ema_long = df['close'].ewm(span=windows[2]) if len(windows) > 2 else ema_medium
        
        # Current price relative to EMAs
        current_price = df['close'].iloc[-1]
        ema_alignment_score = 0
        
        if current_price > ema_short.iloc[-1]:
            ema_alignment_score += 1
        if current_price > ema_medium.iloc[-1]:
            ema_alignment_score += 1
        if len(windows) > 2 and current_price > ema_long.iloc[-1]:
            ema_alignment_score += 1
        
        # EMA sequence alignment
        if ema_short.iloc[-1] > ema_medium.iloc[-1]:
            ema_alignment_score += 1
        if len(windows) > 2 and ema_medium.iloc[-1] > ema_long.iloc[-1]:
            ema_alignment_score += 1
        
        # Normalize score
        max_score = 5 if len(windows) > 2 else 3
        ema_score = ema_alignment_score / max_score
        
        # ADX for trend strength
        if len(df) >= 14:
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
            adx_value = adx.adx().iloc[-1] if not pd.isna(adx.adx().iloc[-1]) else 25
            di_plus = adx.adx_pos().iloc[-1] if not pd.isna(adx.adx_pos().iloc[-1]) else 25
            di_minus = adx.adx_neg().iloc[-1] if not pd.isna(adx.adx_neg().iloc[-1]) else 25
            
            # Trend direction from DI
            if di_plus > di_minus:
                adx_direction = 1
            else:
                adx_direction = -1
        else:
            adx_value = 25
            adx_direction = 0
        
        # Linear regression slope
        if len(df) >= windows[0]:
            x = np.arange(windows[0])
            slope, _ = np.polyfit(x, df['close'].iloc[-windows[0]:].values, 1)
            slope_normalized = slope / df['close'].iloc[-1]  # Normalize by price
        else:
            slope_normalized = 0
        
        # Combine indicators
        if ema_score >= 0.8 and slope_normalized > 0.002 and adx_value > 30:
            return TrendDirection.STRONG_BULLISH
        elif ema_score >= 0.6 and slope_normalized > 0.001:
            return TrendDirection.BULLISH
        elif ema_score <= 0.2 and slope_normalized < -0.002 and adx_value > 30:
            return TrendDirection.STRONG_BEARISH
        elif ema_score <= 0.4 and slope_normalized < -0.001:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL
    
    def calculate_trend_strength(self, df: pd.DataFrame, timeframe: TimeFrame) -> float:
        """Calculate trend strength score (0-1)"""
        if len(df) < 20:
            return 0.0
        
        strength_score = 0.0
        
        # ADX-based strength
        if len(df) >= 14:
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
            adx_value = adx.adx().iloc[-1] if not pd.isna(adx.adx().iloc[-1]) else 25
            strength_score += min(adx_value / 50, 1.0) * 0.4  # 40% weight
        
        # Volatility-adjusted returns
        returns = df['close'].pct_change().dropna()
        if len(returns) > 10:
            # Consistent direction returns
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            if len(positive_returns) > len(negative_returns):
                consistency_score = len(positive_returns) / len(returns)
            else:
                consistency_score = len(negative_returns) / len(returns)
            
            strength_score += consistency_score * 0.3  # 30% weight
        
        # Price momentum
        windows = self.config['trend_windows'][timeframe]
        short_window = windows[0]
        
        if len(df) >= short_window:
            momentum = (df['close'].iloc[-1] - df['close'].iloc[-short_window]) / df['close'].iloc[-short_window]
            momentum_score = min(abs(momentum) * 10, 1.0)  # Scale momentum
            strength_score += momentum_score * 0.3  # 30% weight
        
        return min(strength_score, 1.0)

class ConfluenceCalculator:
    """
    Signal confluence calculation engine
    信号汇合度计算引擎
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """Default configuration for confluence calculation"""
        return {
            'timeframe_weights': {
                TimeFrame.H1: 0.40,   # Trend direction (highest weight)
                TimeFrame.M15: 0.35,  # Primary signals
                TimeFrame.M5: 0.20,   # Execution timing
                TimeFrame.M1: 0.05    # Order management
            },
            'confluence_thresholds': {
                'STRONG_BUY': 0.8,     # High confidence entry
                'BUY': 0.6,            # Standard entry
                'WEAK_BUY': 0.4,       # Low confidence, small size
                'HOLD': 0.2,           # No action
                'WEAK_SELL': -0.4,     # Reduce position
                'SELL': -0.6,          # Standard exit
                'STRONG_SELL': -0.8    # Emergency exit
            },
            'signal_components': {
                'trend_alignment': 0.3,
                'momentum': 0.25,
                'volume_confirmation': 0.2,
                'volatility': 0.15,
                'dip_quality': 0.1
            }
        }
    
    def calculate_confluence_score(self, timeframe_signals: Dict[TimeFrame, TimeFrameSignal]) -> float:
        """Calculate overall confluence score from multiple timeframes"""
        if not timeframe_signals:
            return 0.0
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for timeframe, signal in timeframe_signals.items():
            if timeframe in self.config['timeframe_weights']:
                weight = self.config['timeframe_weights'][timeframe]
                
                # Calculate individual signal score
                signal_score = self._calculate_individual_signal_score(signal)
                
                weighted_score += signal_score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        final_score = weighted_score / total_weight
        return max(-1.0, min(1.0, final_score))  # Clamp to [-1, 1]
    
    def _calculate_individual_signal_score(self, signal: TimeFrameSignal) -> float:
        """Calculate signal score for individual timeframe"""
        components = self.config['signal_components']
        score = 0.0
        
        # Trend alignment score
        trend_score = self._trend_direction_to_score(signal.trend_direction)
        trend_score *= signal.trend_strength  # Weight by strength
        score += trend_score * components['trend_alignment']
        
        # Momentum score
        momentum_score = signal.momentum  # Already -1 to 1
        score += momentum_score * components['momentum']
        
        # Volume confirmation
        volume_score = signal.volume_profile * 2 - 1  # Convert 0-1 to -1 to 1
        score += volume_score * components['volume_confirmation']
        
        # Volatility (inverse for DipMaster - prefer lower volatility)
        volatility_score = (1 - signal.volatility) * 2 - 1
        score += volatility_score * components['volatility']
        
        # DipMaster specific dip quality
        dip_score = signal.dip_quality * 2 - 1  # Convert 0-1 to -1 to 1
        score += dip_score * components['dip_quality']
        
        # Weight by signal confidence
        score *= signal.signal_confidence
        
        return score
    
    def _trend_direction_to_score(self, trend_direction: TrendDirection) -> float:
        """Convert trend direction to numerical score"""
        direction_scores = {
            TrendDirection.STRONG_BULLISH: 1.0,
            TrendDirection.BULLISH: 0.5,
            TrendDirection.NEUTRAL: 0.0,
            TrendDirection.BEARISH: -0.5,
            TrendDirection.STRONG_BEARISH: -1.0
        }
        return direction_scores.get(trend_direction, 0.0)
    
    def classify_signal_strength(self, confluence_score: float) -> str:
        """Classify signal strength based on confluence score"""
        thresholds = self.config['confluence_thresholds']
        
        if confluence_score >= thresholds['STRONG_BUY']:
            return 'STRONG_BUY'
        elif confluence_score >= thresholds['BUY']:
            return 'BUY'
        elif confluence_score >= thresholds['WEAK_BUY']:
            return 'WEAK_BUY'
        elif confluence_score >= thresholds['HOLD']:
            return 'HOLD'
        elif confluence_score >= thresholds['WEAK_SELL']:
            return 'WEAK_SELL'
        elif confluence_score >= thresholds['SELL']:
            return 'SELL'
        else:
            return 'STRONG_SELL'

class ExecutionOptimizer:
    """
    Smart order execution optimization engine
    智能订单执行优化引擎
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """Default configuration for execution optimization"""
        return {
            'execution_methods': {
                'market': {'urgency_threshold': 0.9, 'max_size': 0.1},
                'limit': {'urgency_threshold': 0.7, 'max_size': 0.3},
                'twap': {'urgency_threshold': 0.5, 'max_size': 0.5},
                'vwap': {'urgency_threshold': 0.3, 'max_size': 1.0}
            },
            'size_multipliers': {
                'STRONG_BUY': 1.5,
                'BUY': 1.0,
                'WEAK_BUY': 0.5,
                'HOLD': 0.0,
                'WEAK_SELL': 0.3,
                'SELL': 0.7,
                'STRONG_SELL': 1.0
            },
            'slippage_targets': {
                'market': 5.0,   # 5 bps
                'limit': 1.0,    # 1 bp
                'twap': 2.0,     # 2 bps
                'vwap': 1.5      # 1.5 bps
            },
            'holding_periods': {
                TimeFrame.H1: [180, 360],    # 3-6 hours
                TimeFrame.M15: [60, 180],    # 1-3 hours
                TimeFrame.M5: [15, 60],      # 15-60 minutes
                TimeFrame.M1: [5, 15]        # 5-15 minutes
            }
        }
    
    def optimize_execution(self, signal: MultitimeframeSignal, 
                         current_market_data: Dict[TimeFrame, pd.DataFrame],
                         portfolio_context: Optional[Dict] = None) -> ExecutionRecommendation:
        """Generate optimal execution recommendation"""
        
        # Determine execution urgency
        urgency = self._calculate_execution_urgency(signal, current_market_data)
        
        # Select execution method
        execution_method = self._select_execution_method(urgency, signal.confluence_score)
        
        # Calculate optimal size
        size_multiplier = self._calculate_size_multiplier(signal, portfolio_context)
        
        # Estimate holding period
        holding_period = self._estimate_holding_period(signal, current_market_data)
        
        # Calculate risk levels
        stop_loss, take_profit = self._calculate_risk_levels(signal, current_market_data)
        
        # Execution confidence
        confidence = self._calculate_execution_confidence(signal, current_market_data)
        
        return ExecutionRecommendation(
            action=signal.execution_signal.lower().replace('_', ' '),
            size_multiplier=size_multiplier,
            entry_method=execution_method,
            urgency=urgency,
            max_slippage_bps=self.config['slippage_targets'][execution_method],
            time_horizon_minutes=holding_period,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            confidence=confidence
        )
    
    def _calculate_execution_urgency(self, signal: MultitimeframeSignal,
                                   current_market_data: Dict[TimeFrame, pd.DataFrame]) -> float:
        """Calculate execution urgency (0-1)"""
        urgency_factors = []
        
        # Signal strength urgency
        if signal.confluence_score > 0.8:
            urgency_factors.append(0.9)
        elif signal.confluence_score > 0.6:
            urgency_factors.append(0.7)
        else:
            urgency_factors.append(0.3)
        
        # Volatility urgency (higher volatility = higher urgency)
        if TimeFrame.M5 in signal.timeframe_signals:
            volatility = signal.timeframe_signals[TimeFrame.M5].volatility
            urgency_factors.append(volatility)
        
        # Market momentum urgency
        if TimeFrame.M1 in current_market_data:
            recent_data = current_market_data[TimeFrame.M1].tail(10)
            price_momentum = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            momentum_urgency = min(abs(price_momentum) * 20, 1.0)
            urgency_factors.append(momentum_urgency)
        
        return np.mean(urgency_factors)
    
    def _select_execution_method(self, urgency: float, confluence_score: float) -> str:
        """Select optimal execution method based on urgency and signal strength"""
        methods = self.config['execution_methods']
        
        if urgency >= methods['market']['urgency_threshold']:
            return 'market'
        elif urgency >= methods['limit']['urgency_threshold']:
            return 'limit'
        elif urgency >= methods['twap']['urgency_threshold']:
            return 'twap'
        else:
            return 'vwap'
    
    def _calculate_size_multiplier(self, signal: MultitimeframeSignal,
                                 portfolio_context: Optional[Dict] = None) -> float:
        """Calculate optimal position size multiplier"""
        base_multiplier = self.config['size_multipliers'].get(signal.execution_signal, 1.0)
        
        # Adjust for confluence score
        confluence_adjustment = signal.confluence_score if signal.confluence_score > 0 else 0.5
        
        # Portfolio risk adjustment
        portfolio_adjustment = 1.0
        if portfolio_context:
            # Reduce size if portfolio is concentrated
            portfolio_concentration = portfolio_context.get('concentration', 0.0)
            if portfolio_concentration > 0.7:
                portfolio_adjustment *= 0.7
            
            # Reduce size if high correlation with existing positions
            correlation_risk = portfolio_context.get('correlation_risk', 0.0)
            if correlation_risk > 0.6:
                portfolio_adjustment *= 0.8
        
        final_multiplier = base_multiplier * confluence_adjustment * portfolio_adjustment
        return max(0.1, min(2.0, final_multiplier))  # Clamp between 0.1 and 2.0
    
    def _estimate_holding_period(self, signal: MultitimeframeSignal,
                               current_market_data: Dict[TimeFrame, pd.DataFrame]) -> int:
        """Estimate optimal holding period in minutes"""
        
        # Primary timeframe determines base holding period
        primary_timeframe = TimeFrame.M15  # DipMaster primary timeframe
        
        if primary_timeframe in signal.timeframe_signals:
            trend_strength = signal.timeframe_signals[primary_timeframe].trend_strength
            base_range = self.config['holding_periods'][primary_timeframe]
            
            # Stronger trends = longer holding
            holding_minutes = base_range[0] + (base_range[1] - base_range[0]) * trend_strength
        else:
            holding_minutes = 90  # Default
        
        # Adjust for volatility (higher volatility = shorter holding)
        if TimeFrame.M5 in signal.timeframe_signals:
            volatility = signal.timeframe_signals[TimeFrame.M5].volatility
            volatility_adjustment = 1.0 - (volatility * 0.3)  # Reduce up to 30%
            holding_minutes *= volatility_adjustment
        
        return int(max(15, min(300, holding_minutes)))  # Clamp between 15-300 minutes
    
    def _calculate_risk_levels(self, signal: MultitimeframeSignal,
                             current_market_data: Dict[TimeFrame, pd.DataFrame]) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""
        
        if TimeFrame.M15 not in current_market_data:
            return None, None
        
        current_price = current_market_data[TimeFrame.M15]['close'].iloc[-1]
        
        # ATR-based stops
        if len(current_market_data[TimeFrame.M15]) >= 14:
            atr = ta.volatility.AverageTrueRange(
                current_market_data[TimeFrame.M15]['high'],
                current_market_data[TimeFrame.M15]['low'],
                current_market_data[TimeFrame.M15]['close'],
                window=14
            ).average_true_range().iloc[-1]
            
            atr_pct = atr / current_price
        else:
            atr_pct = 0.01  # Default 1%
        
        # Dynamic stop loss based on volatility and signal strength
        if signal.confluence_score > 0:  # Buy signal
            stop_loss_pct = -(1.5 + signal.confluence_score) * atr_pct  # 1.5-2.5x ATR
            take_profit_pct = (2.0 + signal.confluence_score) * atr_pct   # 2.0-3.0x ATR
            
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        else:  # Sell signal
            stop_loss_pct = (1.5 + abs(signal.confluence_score)) * atr_pct
            take_profit_pct = -(2.0 + abs(signal.confluence_score)) * atr_pct
            
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        
        return stop_loss, take_profit
    
    def _calculate_execution_confidence(self, signal: MultitimeframeSignal,
                                      current_market_data: Dict[TimeFrame, pd.DataFrame]) -> float:
        """Calculate execution confidence score"""
        confidence_factors = []
        
        # Signal confluence confidence
        confidence_factors.append(abs(signal.confluence_score))
        
        # Timeframe agreement confidence
        if len(signal.timeframe_signals) >= 3:
            trend_directions = [s.trend_direction for s in signal.timeframe_signals.values()]
            bullish_count = sum(1 for d in trend_directions if d in [TrendDirection.BULLISH, TrendDirection.STRONG_BULLISH])
            bearish_count = sum(1 for d in trend_directions if d in [TrendDirection.BEARISH, TrendDirection.STRONG_BEARISH])
            
            agreement = max(bullish_count, bearish_count) / len(trend_directions)
            confidence_factors.append(agreement)
        
        # Market stability confidence
        if TimeFrame.M5 in current_market_data:
            recent_volatility = current_market_data[TimeFrame.M5]['close'].pct_change().tail(20).std()
            stability_score = max(0, 1 - recent_volatility * 50)  # Lower volatility = higher confidence
            confidence_factors.append(stability_score)
        
        return np.mean(confidence_factors)

class MultiTimeframeSignalEngine:
    """
    Master Multi-Timeframe Signal Integration Engine
    主多时间框架信号整合引擎
    
    Coordinates signal generation across multiple timeframes, calculates confluence,
    and provides optimal execution recommendations for the DipMaster strategy.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Initialize sub-components
        self.trend_analyzer = TrendAlignmentAnalyzer(self.config.get('trend_analysis', {}))
        self.confluence_calculator = ConfluenceCalculator(self.config.get('confluence', {}))
        self.execution_optimizer = ExecutionOptimizer(self.config.get('execution', {}))
        
        # Integration with existing systems
        self.regime_detector = MarketRegimeDetector()
        self.parameter_engine = AdaptiveParameterEngine()
        
        # Data management
        self.market_data_cache = {}
        self.signal_history = defaultdict(lambda: deque(maxlen=1000))
        self.performance_tracker = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        self.logger.info("MultiTimeframeSignalEngine initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Default configuration for the multi-timeframe engine"""
        return {
            'supported_timeframes': [TimeFrame.H1, TimeFrame.M15, TimeFrame.M5, TimeFrame.M1],
            'required_timeframes': [TimeFrame.M15, TimeFrame.M5],  # Minimum for signal generation
            'data_requirements': {
                TimeFrame.H1: 168,    # 1 week
                TimeFrame.M15: 672,   # 1 week
                TimeFrame.M5: 288,    # 1 day
                TimeFrame.M1: 240     # 4 hours
            },
            'signal_validity_minutes': {
                TimeFrame.H1: 60,     # 1 hour
                TimeFrame.M15: 15,    # 15 minutes
                TimeFrame.M5: 5,      # 5 minutes
                TimeFrame.M1: 1       # 1 minute
            },
            'performance_tracking': {
                'min_samples': 50,
                'reoptimization_threshold': 0.1,  # 10% performance degradation
                'confidence_decay_rate': 0.95
            }
        }
    
    def update_market_data(self, symbol: str, timeframe: TimeFrame, data: pd.DataFrame):
        """Update market data for specific symbol and timeframe"""
        with self._lock:
            if symbol not in self.market_data_cache:
                self.market_data_cache[symbol] = {}
            
            self.market_data_cache[symbol][timeframe] = data.copy()
            
            # Keep only required amount of data
            required_bars = self.config['data_requirements'][timeframe]
            if len(data) > required_bars:
                self.market_data_cache[symbol][timeframe] = data.tail(required_bars).copy()
    
    def generate_timeframe_signal(self, symbol: str, timeframe: TimeFrame,
                                data: Optional[pd.DataFrame] = None) -> Optional[TimeFrameSignal]:
        """Generate signal for specific timeframe"""
        
        # Get data
        if data is None:
            if symbol not in self.market_data_cache or timeframe not in self.market_data_cache[symbol]:
                self.logger.warning(f"No data available for {symbol} {timeframe.value}")
                return None
            data = self.market_data_cache[symbol][timeframe]
        
        if len(data) < 50:  # Minimum data requirement
            return None
        
        try:
            # Trend analysis
            trend_direction = self.trend_analyzer.analyze_trend_direction(data, timeframe)
            trend_strength = self.trend_analyzer.calculate_trend_strength(data, timeframe)
            
            # Technical indicators
            indicators = self._calculate_technical_indicators(data, timeframe)
            
            # DipMaster specific analysis
            dip_quality = self._analyze_dip_quality(data, timeframe)
            exit_timing_score = self._calculate_exit_timing_score(data, timeframe)
            
            # Support/Resistance levels
            sr_levels = self._calculate_support_resistance(data, timeframe)
            
            # Volume analysis
            volume_profile = self._analyze_volume_profile(data, timeframe)
            
            # Signal confidence
            signal_confidence = self._calculate_signal_confidence(
                data, timeframe, trend_strength, indicators, dip_quality
            )
            
            return TimeFrameSignal(
                timeframe=timeframe,
                symbol=symbol,
                timestamp=datetime.now(),
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                momentum=indicators.get('momentum', 0.0),
                volatility=indicators.get('volatility', 0.5),
                volume_profile=volume_profile,
                support_resistance=sr_levels,
                technical_indicators=indicators,
                signal_confidence=signal_confidence,
                dip_quality=dip_quality,
                exit_timing_score=exit_timing_score
            )
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol} {timeframe.value}: {e}")
            return None
    
    def _calculate_technical_indicators(self, data: pd.DataFrame, timeframe: TimeFrame) -> Dict[str, float]:
        """Calculate comprehensive technical indicators"""
        indicators = {}
        
        try:
            # RSI
            if len(data) >= 14:
                rsi = ta.momentum.RSIIndicator(data['close'], window=14)
                indicators['rsi'] = rsi.rsi().iloc[-1] if not pd.isna(rsi.rsi().iloc[-1]) else 50
            
            # MACD
            if len(data) >= 26:
                macd = ta.trend.MACD(data['close'])
                indicators['macd'] = macd.macd().iloc[-1] if not pd.isna(macd.macd().iloc[-1]) else 0
                indicators['macd_signal'] = macd.macd_signal().iloc[-1] if not pd.isna(macd.macd_signal().iloc[-1]) else 0
                indicators['macd_histogram'] = macd.macd_diff().iloc[-1] if not pd.isna(macd.macd_diff().iloc[-1]) else 0
                
                # Momentum from MACD
                if indicators['macd'] > indicators['macd_signal']:
                    indicators['momentum'] = min(abs(indicators['macd_histogram']) * 10, 1.0)
                else:
                    indicators['momentum'] = -min(abs(indicators['macd_histogram']) * 10, 1.0)
            
            # Bollinger Bands
            if len(data) >= 20:
                bb = ta.volatility.BollingerBands(data['close'], window=20)
                bb_upper = bb.bollinger_hband().iloc[-1]
                bb_lower = bb.bollinger_lband().iloc[-1]
                bb_middle = bb.bollinger_mavg().iloc[-1]
                current_price = data['close'].iloc[-1]
                
                # Position within bands
                if bb_upper > bb_lower:
                    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                    indicators['bb_position'] = bb_position
                
                # Volatility from BB width
                bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0.1
                indicators['volatility'] = min(bb_width * 10, 1.0)
            
            # Stochastic
            if len(data) >= 14:
                stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'])
                indicators['stoch_k'] = stoch.stoch().iloc[-1] if not pd.isna(stoch.stoch().iloc[-1]) else 50
                indicators['stoch_d'] = stoch.stoch_signal().iloc[-1] if not pd.isna(stoch.stoch_signal().iloc[-1]) else 50
            
            # ADX
            if len(data) >= 14:
                adx = ta.trend.ADXIndicator(data['high'], data['low'], data['close'])
                indicators['adx'] = adx.adx().iloc[-1] if not pd.isna(adx.adx().iloc[-1]) else 25
            
            # ATR for volatility
            if len(data) >= 14:
                atr = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'])
                atr_value = atr.average_true_range().iloc[-1]
                indicators['atr_pct'] = atr_value / data['close'].iloc[-1] if data['close'].iloc[-1] > 0 else 0.01
                
                # Update volatility with ATR
                if 'volatility' not in indicators:
                    indicators['volatility'] = min(indicators['atr_pct'] * 50, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
        
        return indicators
    
    def _analyze_dip_quality(self, data: pd.DataFrame, timeframe: TimeFrame) -> float:
        """Analyze dip quality for DipMaster strategy (0-1)"""
        if len(data) < 20:
            return 0.0
        
        quality_score = 0.0
        current_price = data['close'].iloc[-1]
        
        try:
            # Price vs recent open (core DipMaster condition)
            if timeframe in [TimeFrame.M15, TimeFrame.M5]:
                recent_open = data['open'].iloc[-1]
                if current_price < recent_open:
                    dip_magnitude = (recent_open - current_price) / recent_open
                    quality_score += min(dip_magnitude * 100, 0.4)  # Up to 40% for dip
            
            # RSI oversold condition
            if len(data) >= 14:
                rsi = ta.momentum.RSIIndicator(data['close'], window=14)
                rsi_value = rsi.rsi().iloc[-1] if not pd.isna(rsi.rsi().iloc[-1]) else 50
                
                if 30 <= rsi_value <= 50:  # DipMaster sweet spot
                    quality_score += 0.3
                elif rsi_value < 30:  # Oversold
                    quality_score += 0.2
            
            # Volume confirmation
            if len(data) >= 20:
                volume_ma = data['volume'].rolling(20).mean().iloc[-1]
                current_volume = data['volume'].iloc[-1]
                if current_volume > volume_ma * 1.2:  # Volume spike
                    quality_score += 0.2
            
            # Support level proximity
            if len(data) >= 50:
                recent_low = data['low'].rolling(20).min().iloc[-1]
                if current_price <= recent_low * 1.02:  # Within 2% of support
                    quality_score += 0.1
            
        except Exception as e:
            self.logger.error(f"Error analyzing dip quality: {e}")
        
        return min(quality_score, 1.0)
    
    def _calculate_exit_timing_score(self, data: pd.DataFrame, timeframe: TimeFrame) -> float:
        """Calculate exit timing score for DipMaster boundary exits (0-1)"""
        if timeframe != TimeFrame.M15:  # DipMaster primary timeframe
            return 0.5
        
        try:
            # Check if we're near 15-minute boundaries
            current_minute = datetime.now().minute
            
            # Preferred exit windows: 15-29min, 45-59min
            if 15 <= current_minute < 30 or 45 <= current_minute < 60:
                timing_score = 0.8
            elif 30 <= current_minute < 45 or 0 <= current_minute < 15:
                timing_score = 0.4
            else:
                timing_score = 0.2
            
            # Adjust for profitability potential
            if len(data) >= 5:
                recent_momentum = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
                if recent_momentum > 0:  # Upward momentum
                    timing_score *= 1.2
                else:
                    timing_score *= 0.8
            
            return min(timing_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating exit timing score: {e}")
            return 0.5
    
    def _calculate_support_resistance(self, data: pd.DataFrame, timeframe: TimeFrame) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        levels = {}
        
        try:
            if len(data) >= 50:
                # Pivot point analysis
                high_period = 20 if timeframe in [TimeFrame.M5, TimeFrame.M1] else 50
                low_period = high_period
                
                recent_high = data['high'].rolling(high_period).max().iloc[-1]
                recent_low = data['low'].rolling(low_period).min().iloc[-1]
                
                levels['resistance'] = recent_high
                levels['support'] = recent_low
                
                # Distance from levels
                current_price = data['close'].iloc[-1]
                levels['resistance_distance'] = (recent_high - current_price) / current_price
                levels['support_distance'] = (current_price - recent_low) / current_price
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
        
        return levels
    
    def _analyze_volume_profile(self, data: pd.DataFrame, timeframe: TimeFrame) -> float:
        """Analyze volume profile (0-1)"""
        try:
            if len(data) >= 20:
                volume_ma = data['volume'].rolling(20).mean().iloc[-1]
                current_volume = data['volume'].iloc[-1]
                
                volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
                
                # Normalize to 0-1 scale
                return min(volume_ratio / 3.0, 1.0)  # 3x volume = max score
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume profile: {e}")
        
        return 0.5  # Default neutral
    
    def _calculate_signal_confidence(self, data: pd.DataFrame, timeframe: TimeFrame,
                                   trend_strength: float, indicators: Dict,
                                   dip_quality: float) -> float:
        """Calculate overall signal confidence (0-1)"""
        confidence_factors = []
        
        # Trend strength confidence
        confidence_factors.append(trend_strength)
        
        # Technical indicator confidence
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            # Higher confidence when RSI is in DipMaster range
            if 30 <= rsi <= 50:
                confidence_factors.append(0.8)
            elif 20 <= rsi <= 60:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.3)
        
        # Volume confidence
        if 'volume_ratio' in indicators:
            volume_confidence = min(indicators['volume_ratio'] / 2.0, 1.0)
            confidence_factors.append(volume_confidence)
        
        # Dip quality confidence
        confidence_factors.append(dip_quality)
        
        # Data quality confidence
        data_quality = min(len(data) / 100, 1.0)  # More data = higher confidence
        confidence_factors.append(data_quality)
        
        return np.mean(confidence_factors)
    
    def generate_multitimeframe_signal(self, symbol: str) -> Optional[MultitimeframeSignal]:
        """Generate comprehensive multi-timeframe signal"""
        
        # Check data availability
        if symbol not in self.market_data_cache:
            self.logger.warning(f"No market data available for {symbol}")
            return None
        
        # Generate signals for each timeframe
        timeframe_signals = {}
        
        for timeframe in self.config['supported_timeframes']:
            if timeframe in self.market_data_cache[symbol]:
                signal = self.generate_timeframe_signal(symbol, timeframe)
                if signal:
                    timeframe_signals[timeframe] = signal
        
        # Check if we have minimum required timeframes
        required_tfs = set(self.config['required_timeframes'])
        available_tfs = set(timeframe_signals.keys())
        
        if not required_tfs.issubset(available_tfs):
            missing_tfs = required_tfs - available_tfs
            self.logger.warning(f"Missing required timeframes for {symbol}: {missing_tfs}")
            return None
        
        # Calculate confluence score
        confluence_score = self.confluence_calculator.calculate_confluence_score(timeframe_signals)
        
        # Classify execution signal
        execution_signal = self.confluence_calculator.classify_signal_strength(confluence_score)
        
        # Generate execution recommendation
        execution_rec = self.execution_optimizer.optimize_execution(
            MultitimeframeSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                timeframe_signals=timeframe_signals,
                confluence_score=confluence_score,
                execution_signal=execution_signal,
                recommended_action={},
                risk_assessment={},
                execution_urgency=0.0,
                expected_holding_period=0
            ),
            self.market_data_cache[symbol]
        )
        
        # Risk assessment
        risk_assessment = self._assess_risk(symbol, timeframe_signals, confluence_score)
        
        # Create final signal
        multitf_signal = MultitimeframeSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            timeframe_signals=timeframe_signals,
            confluence_score=confluence_score,
            execution_signal=execution_signal,
            recommended_action=asdict(execution_rec),
            risk_assessment=risk_assessment,
            execution_urgency=execution_rec.urgency,
            expected_holding_period=execution_rec.time_horizon_minutes
        )
        
        # Store signal history
        with self._lock:
            self.signal_history[symbol].append(multitf_signal)
        
        return multitf_signal
    
    def _assess_risk(self, symbol: str, timeframe_signals: Dict[TimeFrame, TimeFrameSignal],
                    confluence_score: float) -> Dict[str, float]:
        """Assess risk factors for the signal"""
        risk_factors = {}
        
        try:
            # Volatility risk
            volatilities = [signal.volatility for signal in timeframe_signals.values()]
            risk_factors['volatility_risk'] = np.mean(volatilities)
            
            # Trend alignment risk (higher when timeframes disagree)
            trend_directions = [signal.trend_direction for signal in timeframe_signals.values()]
            bullish_count = sum(1 for d in trend_directions if d in [TrendDirection.BULLISH, TrendDirection.STRONG_BULLISH])
            bearish_count = sum(1 for d in trend_directions if d in [TrendDirection.BEARISH, TrendDirection.STRONG_BEARISH])
            neutral_count = len(trend_directions) - bullish_count - bearish_count
            
            max_agreement = max(bullish_count, bearish_count, neutral_count)
            alignment_risk = 1.0 - (max_agreement / len(trend_directions))
            risk_factors['alignment_risk'] = alignment_risk
            
            # Signal confidence risk
            confidences = [signal.signal_confidence for signal in timeframe_signals.values()]
            avg_confidence = np.mean(confidences)
            risk_factors['confidence_risk'] = 1.0 - avg_confidence
            
            # Market regime risk
            try:
                # Get market regime from most recent 15M data
                if TimeFrame.M15 in self.market_data_cache[symbol]:
                    regime_signal = self.regime_detector.identify_regime(
                        self.market_data_cache[symbol][TimeFrame.M15], symbol
                    )
                    
                    # Higher risk in volatile or downtrending regimes
                    if regime_signal.regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.STRONG_DOWNTREND]:
                        risk_factors['regime_risk'] = 0.8
                    elif regime_signal.regime == MarketRegime.TRANSITION:
                        risk_factors['regime_risk'] = 0.6
                    else:
                        risk_factors['regime_risk'] = 0.3
            except Exception:
                risk_factors['regime_risk'] = 0.5  # Default
            
            # Overall risk score
            risk_factors['overall_risk'] = np.mean(list(risk_factors.values()))
            
        except Exception as e:
            self.logger.error(f"Error assessing risk for {symbol}: {e}")
            risk_factors = {'overall_risk': 0.5}
        
        return risk_factors
    
    def get_signal_summary(self, symbol: str, lookback_minutes: int = 60) -> Dict:
        """Get summary of recent signals for a symbol"""
        if symbol not in self.signal_history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        recent_signals = [
            signal for signal in self.signal_history[symbol]
            if signal.timestamp >= cutoff_time
        ]
        
        if not recent_signals:
            return {}
        
        # Calculate summary statistics
        confluence_scores = [s.confluence_score for s in recent_signals]
        execution_signals = [s.execution_signal for s in recent_signals]
        
        summary = {
            'symbol': symbol,
            'total_signals': len(recent_signals),
            'avg_confluence_score': np.mean(confluence_scores),
            'latest_signal': recent_signals[-1].execution_signal,
            'latest_confluence': recent_signals[-1].confluence_score,
            'latest_timestamp': recent_signals[-1].timestamp.isoformat(),
            'signal_distribution': {
                signal: execution_signals.count(signal) 
                for signal in set(execution_signals)
            },
            'avg_execution_urgency': np.mean([s.execution_urgency for s in recent_signals]),
            'avg_holding_period': np.mean([s.expected_holding_period for s in recent_signals])
        }
        
        return summary
    
    def generate_performance_report(self, symbol: Optional[str] = None) -> Dict:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'symbol_performance': {},
            'signal_quality': {},
            'execution_stats': {}
        }
        
        # Overall summary
        if symbol:
            symbols = [symbol]
        else:
            symbols = list(self.signal_history.keys())
        
        total_signals = sum(len(self.signal_history[s]) for s in symbols)
        
        report['summary'] = {
            'total_symbols': len(symbols),
            'total_signals_generated': total_signals,
            'avg_signals_per_symbol': total_signals / len(symbols) if symbols else 0,
            'supported_timeframes': [tf.value for tf in self.config['supported_timeframes']]
        }
        
        # Per-symbol analysis
        for sym in symbols:
            if sym in self.signal_history and self.signal_history[sym]:
                signals = list(self.signal_history[sym])
                
                confluence_scores = [s.confluence_score for s in signals]
                urgencies = [s.execution_urgency for s in signals]
                holding_periods = [s.expected_holding_period for s in signals]
                
                report['symbol_performance'][sym] = {
                    'total_signals': len(signals),
                    'avg_confluence_score': np.mean(confluence_scores),
                    'avg_execution_urgency': np.mean(urgencies),
                    'avg_holding_period': np.mean(holding_periods),
                    'signal_distribution': {},
                    'latest_signal_time': signals[-1].timestamp.isoformat(),
                    'confluence_trend': confluence_scores[-10:] if len(confluence_scores) >= 10 else confluence_scores
                }
                
                # Signal distribution
                execution_signals = [s.execution_signal for s in signals]
                for signal_type in ['STRONG_BUY', 'BUY', 'WEAK_BUY', 'HOLD', 'WEAK_SELL', 'SELL', 'STRONG_SELL']:
                    count = execution_signals.count(signal_type)
                    report['symbol_performance'][sym]['signal_distribution'][signal_type] = count
        
        return report

# Factory functions for easy integration
def create_multitimeframe_engine(config: Optional[Dict] = None) -> MultiTimeframeSignalEngine:
    """Factory function to create multi-timeframe signal engine"""
    return MultiTimeframeSignalEngine(config)

def analyze_symbol_multitimeframe(symbol: str, market_data: Dict[TimeFrame, pd.DataFrame],
                                 engine: Optional[MultiTimeframeSignalEngine] = None) -> Optional[MultitimeframeSignal]:
    """Analyze multi-timeframe signals for a single symbol"""
    if engine is None:
        engine = create_multitimeframe_engine()
    
    # Update market data
    for timeframe, data in market_data.items():
        engine.update_market_data(symbol, timeframe, data)
    
    # Generate signal
    return engine.generate_multitimeframe_signal(symbol)