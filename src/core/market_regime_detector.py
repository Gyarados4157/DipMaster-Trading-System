#!/usr/bin/env python3
"""
Market Regime Detection Module for DipMaster Strategy
市场体制检测模块 - DipMaster策略专用

This module implements a sophisticated market regime detection system to address
the core weakness of DipMaster strategy - poor performance in trending markets.
Current BTCUSDT win rate is only 47.7%, which needs to be improved to 65%+.

Market Regimes Identified:
1. RANGE_BOUND: Sideways/consolidation (optimal for original DipMaster)
2. STRONG_UPTREND: Bull market conditions (modify entry criteria)
3. STRONG_DOWNTREND: Bear market conditions (pause or reverse strategy)
4. HIGH_VOLATILITY: Extreme volatility periods (enhanced risk controls)
5. LOW_VOLATILITY: Low vol grinding (different parameters)

Author: Strategy Orchestrator
Date: 2025-08-17
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import json
from pathlib import Path
import ta
from scipy.stats import zscore, percentileofscore
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numba
from numba import jit

warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    """Market regime classifications"""
    RANGE_BOUND = "RANGE_BOUND"
    STRONG_UPTREND = "STRONG_UPTREND" 
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRANSITION = "TRANSITION"

@dataclass
class RegimeParams:
    """Adaptive parameters for each market regime"""
    rsi_low: float
    rsi_high: float
    dip_threshold: float
    volume_threshold: float
    target_profit: float
    stop_loss: float
    max_holding_minutes: int
    confidence_multiplier: float

@dataclass
class RegimeSignal:
    """Market regime detection signal"""
    regime: MarketRegime
    confidence: float
    timestamp: datetime
    features: Dict
    transition_probability: float
    stability_score: float

class MarketRegimeDetector:
    """
    Advanced Market Regime Detection System
    高级市场体制检测系统
    
    Uses multi-timeframe analysis to identify market conditions and adapt
    DipMaster strategy parameters accordingly. Addresses the core issue of
    poor performance in trending markets.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the market regime detector"""
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self.scaler = StandardScaler()
        self.regime_history = {}
        self.feature_cache = {}
        self.adaptive_params = self._initialize_adaptive_parameters()
        
        # Regime detection thresholds
        self.trend_strength_threshold = 0.6
        self.volatility_percentile_high = 80
        self.volatility_percentile_low = 20
        self.confidence_threshold = 0.7
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for regime detection"""
        return {
            'lookback_periods': {
                'short': 24,    # 2 hours (5min bars)
                'medium': 96,   # 8 hours
                'long': 288     # 24 hours
            },
            'timeframes': ['5m', '15m', '1h', '4h'],
            'volatility_window': 48,  # 4 hours for volatility calculation
            'trend_windows': [24, 48, 96],  # Multiple trend windows
            'volume_ma_window': 20,
            'regime_stability_window': 12  # Minimum periods for regime stability
        }
    
    def _initialize_adaptive_parameters(self) -> Dict[MarketRegime, RegimeParams]:
        """Initialize adaptive parameters for each market regime"""
        return {
            MarketRegime.RANGE_BOUND: RegimeParams(
                rsi_low=30, rsi_high=50, dip_threshold=0.002,
                volume_threshold=1.5, target_profit=0.008, stop_loss=-0.015,
                max_holding_minutes=180, confidence_multiplier=1.0
            ),
            MarketRegime.STRONG_UPTREND: RegimeParams(
                rsi_low=20, rsi_high=40, dip_threshold=0.003,
                volume_threshold=2.0, target_profit=0.012, stop_loss=-0.008,
                max_holding_minutes=90, confidence_multiplier=0.8
            ),
            MarketRegime.STRONG_DOWNTREND: RegimeParams(
                rsi_low=15, rsi_high=35, dip_threshold=0.005,
                volume_threshold=2.5, target_profit=0.006, stop_loss=-0.005,
                max_holding_minutes=60, confidence_multiplier=0.5
            ),
            MarketRegime.HIGH_VOLATILITY: RegimeParams(
                rsi_low=25, rsi_high=45, dip_threshold=0.004,
                volume_threshold=3.0, target_profit=0.015, stop_loss=-0.010,
                max_holding_minutes=45, confidence_multiplier=0.6
            ),
            MarketRegime.LOW_VOLATILITY: RegimeParams(
                rsi_low=35, rsi_high=55, dip_threshold=0.001,
                volume_threshold=1.2, target_profit=0.005, stop_loss=-0.020,
                max_holding_minutes=240, confidence_multiplier=1.2
            ),
            MarketRegime.TRANSITION: RegimeParams(
                rsi_low=25, rsi_high=45, dip_threshold=0.0025,
                volume_threshold=1.8, target_profit=0.006, stop_loss=-0.012,
                max_holding_minutes=120, confidence_multiplier=0.7
            )
        }
    
    def _calculate_trend_strength_numba(self, prices: np.ndarray, window: int) -> float:
        """Calculate trend strength using numba optimization"""
        if len(prices) < window:
            return 0.0
        
        recent_prices = prices[-window:]
        trend_slope = (recent_prices[-1] - recent_prices[0]) / len(recent_prices)
        price_range = np.max(recent_prices) - np.min(recent_prices)
        
        if price_range == 0:
            return 0.0
        
        # Normalize trend strength
        trend_strength = abs(trend_slope) / (price_range / len(recent_prices))
        return min(trend_strength, 1.0)
    
    def calculate_trend_features(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive trend analysis features"""
        features = {}
        
        # Multiple timeframe trend strength
        for window in self.config['trend_windows']:
            if len(df) >= window:
                prices = df['close'].values
                trend_strength = self._calculate_trend_strength_numba(prices, window)
                features[f'trend_strength_{window}'] = trend_strength
                
                # Trend direction
                price_change = (df['close'].iloc[-1] - df['close'].iloc[-window]) / df['close'].iloc[-window]
                features[f'trend_direction_{window}'] = 1 if price_change > 0 else -1
                features[f'trend_magnitude_{window}'] = abs(price_change)
        
        # ADX for trend strength confirmation
        if len(df) >= 14:
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
            features['adx'] = adx.adx().iloc[-1] if not pd.isna(adx.adx().iloc[-1]) else 25
            features['adx_di_plus'] = adx.adx_pos().iloc[-1] if not pd.isna(adx.adx_pos().iloc[-1]) else 25
            features['adx_di_minus'] = adx.adx_neg().iloc[-1] if not pd.isna(adx.adx_neg().iloc[-1]) else 25
        
        # Linear regression slope
        if len(df) >= 20:
            x = np.arange(len(df))
            slope, intercept = np.polyfit(x[-20:], df['close'].iloc[-20:].values, 1)
            features['lr_slope_20'] = slope / df['close'].iloc[-1]  # Normalized slope
        
        return features
    
    def calculate_volatility_features(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive volatility analysis features"""
        features = {}
        
        # Rolling volatility (annualized)
        if len(df) >= self.config['volatility_window']:
            returns = df['close'].pct_change().dropna()
            rolling_vol = returns.rolling(window=self.config['volatility_window']).std() * np.sqrt(365*24*12)  # 5min bars
            features['volatility'] = rolling_vol.iloc[-1] if not pd.isna(rolling_vol.iloc[-1]) else 0.5
            
            # Volatility percentile
            vol_history = rolling_vol.dropna()
            if len(vol_history) > 100:
                features['volatility_percentile'] = percentileofscore(vol_history, rolling_vol.iloc[-1])
        
        # GARCH-like volatility clustering
        if len(df) >= 50:
            returns = df['close'].pct_change().dropna()
            returns_squared = returns ** 2
            vol_clustering = returns_squared.rolling(window=20).mean().iloc[-1]
            features['volatility_clustering'] = vol_clustering
        
        # Bollinger Band width
        if len(df) >= 20:
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            features['bb_width'] = bb_width.iloc[-1] if not pd.isna(bb_width.iloc[-1]) else 0.1
        
        # ATR (Average True Range)
        if len(df) >= 14:
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
            features['atr'] = atr.average_true_range().iloc[-1] / df['close'].iloc[-1]
        
        return features
    
    def calculate_momentum_features(self, df: pd.DataFrame) -> Dict:
        """Calculate momentum and oscillator features"""
        features = {}
        
        # RSI multiple periods
        for period in [14, 21, 30]:
            if len(df) >= period:
                rsi = ta.momentum.RSIIndicator(df['close'], window=period)
                features[f'rsi_{period}'] = rsi.rsi().iloc[-1] if not pd.isna(rsi.rsi().iloc[-1]) else 50
        
        # MACD
        if len(df) >= 26:
            macd = ta.trend.MACD(df['close'])
            features['macd'] = macd.macd().iloc[-1] if not pd.isna(macd.macd().iloc[-1]) else 0
            features['macd_signal'] = macd.macd_signal().iloc[-1] if not pd.isna(macd.macd_signal().iloc[-1]) else 0
            features['macd_histogram'] = macd.macd_diff().iloc[-1] if not pd.isna(macd.macd_diff().iloc[-1]) else 0
        
        # Stochastic
        if len(df) >= 14:
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            features['stoch_k'] = stoch.stoch().iloc[-1] if not pd.isna(stoch.stoch().iloc[-1]) else 50
            features['stoch_d'] = stoch.stoch_signal().iloc[-1] if not pd.isna(stoch.stoch_signal().iloc[-1]) else 50
        
        # Williams %R
        if len(df) >= 14:
            williams = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'])
            features['williams_r'] = williams.williams_r().iloc[-1] if not pd.isna(williams.williams_r().iloc[-1]) else -50
        
        # Rate of Change
        for period in [5, 10, 20]:
            if len(df) >= period:
                roc = ta.momentum.ROCIndicator(df['close'], window=period)
                features[f'roc_{period}'] = roc.roc().iloc[-1] if not pd.isna(roc.roc().iloc[-1]) else 0
        
        return features
    
    def calculate_volume_features(self, df: pd.DataFrame) -> Dict:
        """Calculate volume flow and accumulation features"""
        features = {}
        
        # Volume moving average ratio
        if len(df) >= self.config['volume_ma_window']:
            vol_ma = df['volume'].rolling(window=self.config['volume_ma_window']).mean()
            features['volume_ratio'] = df['volume'].iloc[-1] / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 1
        
        # On-Balance Volume
        if len(df) >= 20:
            obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
            obv_values = obv.on_balance_volume()
            obv_ma = obv_values.rolling(window=20).mean()
            features['obv_ratio'] = obv_values.iloc[-1] / obv_ma.iloc[-1] if obv_ma.iloc[-1] != 0 else 1
        
        # Accumulation/Distribution Line
        if len(df) >= 20:
            ad = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume'])
            ad_values = ad.acc_dist_index()
            features['ad_line'] = ad_values.iloc[-1] if not pd.isna(ad_values.iloc[-1]) else 0
        
        # Chaikin Money Flow
        if len(df) >= 20:
            cmf = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'])
            features['cmf'] = cmf.chaikin_money_flow().iloc[-1] if not pd.isna(cmf.chaikin_money_flow().iloc[-1]) else 0
        
        return features
    
    def calculate_market_structure_features(self, df: pd.DataFrame) -> Dict:
        """Calculate market structure and pattern features"""
        features = {}
        
        # Support/Resistance strength
        if len(df) >= 50:
            highs = df['high'].rolling(window=10).max()
            lows = df['low'].rolling(window=10).min()
            
            # Distance from recent highs/lows
            features['distance_from_high'] = (highs.iloc[-1] - df['close'].iloc[-1]) / df['close'].iloc[-1]
            features['distance_from_low'] = (df['close'].iloc[-1] - lows.iloc[-1]) / df['close'].iloc[-1]
        
        # Price position within recent range
        if len(df) >= 20:
            recent_high = df['high'].rolling(window=20).max().iloc[-1]
            recent_low = df['low'].rolling(window=20).min().iloc[-1]
            if recent_high > recent_low:
                features['price_position'] = (df['close'].iloc[-1] - recent_low) / (recent_high - recent_low)
        
        # Fractal analysis
        if len(df) >= 10:
            # Simple fractal detection
            highs = df['high']
            lows = df['low']
            
            recent_fractals_up = 0
            recent_fractals_down = 0
            
            for i in range(5, len(df)-5):
                # Fractal up
                if all(highs.iloc[i] >= highs.iloc[i-j] for j in range(1, 3)) and \
                   all(highs.iloc[i] >= highs.iloc[i+j] for j in range(1, 3)):
                    if i >= len(df) - 20:  # Recent fractals
                        recent_fractals_up += 1
                
                # Fractal down  
                if all(lows.iloc[i] <= lows.iloc[i-j] for j in range(1, 3)) and \
                   all(lows.iloc[i] <= lows.iloc[i+j] for j in range(1, 3)):
                    if i >= len(df) - 20:  # Recent fractals
                        recent_fractals_down += 1
            
            features['recent_fractals_ratio'] = recent_fractals_up / max(recent_fractals_down, 1)
        
        return features
    
    def identify_regime(self, df: pd.DataFrame, symbol: str = None) -> RegimeSignal:
        """
        Identify current market regime using comprehensive analysis
        主要的市场体制识别函数
        """
        if len(df) < 50:  # Minimum data requirement
            return RegimeSignal(
                regime=MarketRegime.TRANSITION,
                confidence=0.0,
                timestamp=datetime.now(),
                features={},
                transition_probability=1.0,
                stability_score=0.0
            )
        
        # Calculate all feature groups
        trend_features = self.calculate_trend_features(df)
        volatility_features = self.calculate_volatility_features(df) 
        momentum_features = self.calculate_momentum_features(df)
        volume_features = self.calculate_volume_features(df)
        structure_features = self.calculate_market_structure_features(df)
        
        # Combine all features
        all_features = {
            **trend_features,
            **volatility_features, 
            **momentum_features,
            **volume_features,
            **structure_features
        }
        
        # Regime classification logic
        regime_scores = {}
        
        # STRONG_UPTREND detection
        uptrend_score = 0.0
        if 'trend_strength_24' in trend_features and trend_features['trend_strength_24'] > 0.6:
            uptrend_score += 0.3
        if 'trend_direction_24' in trend_features and trend_features['trend_direction_24'] > 0:
            uptrend_score += 0.2
        if 'adx' in trend_features and trend_features['adx'] > 30:
            uptrend_score += 0.2
        if 'lr_slope_20' in trend_features and trend_features['lr_slope_20'] > 0.001:
            uptrend_score += 0.2
        if 'rsi_14' in momentum_features and momentum_features['rsi_14'] > 50:
            uptrend_score += 0.1
        
        regime_scores[MarketRegime.STRONG_UPTREND] = uptrend_score
        
        # STRONG_DOWNTREND detection
        downtrend_score = 0.0
        if 'trend_strength_24' in trend_features and trend_features['trend_strength_24'] > 0.6:
            downtrend_score += 0.3
        if 'trend_direction_24' in trend_features and trend_features['trend_direction_24'] < 0:
            downtrend_score += 0.2
        if 'adx' in trend_features and trend_features['adx'] > 30:
            downtrend_score += 0.2
        if 'lr_slope_20' in trend_features and trend_features['lr_slope_20'] < -0.001:
            downtrend_score += 0.2
        if 'rsi_14' in momentum_features and momentum_features['rsi_14'] < 50:
            downtrend_score += 0.1
        
        regime_scores[MarketRegime.STRONG_DOWNTREND] = downtrend_score
        
        # HIGH_VOLATILITY detection
        high_vol_score = 0.0
        if 'volatility_percentile' in volatility_features and volatility_features['volatility_percentile'] > self.volatility_percentile_high:
            high_vol_score += 0.4
        if 'bb_width' in volatility_features and volatility_features['bb_width'] > 0.15:
            high_vol_score += 0.3
        if 'atr' in volatility_features and volatility_features['atr'] > 0.03:
            high_vol_score += 0.3
        
        regime_scores[MarketRegime.HIGH_VOLATILITY] = high_vol_score
        
        # LOW_VOLATILITY detection  
        low_vol_score = 0.0
        if 'volatility_percentile' in volatility_features and volatility_features['volatility_percentile'] < self.volatility_percentile_low:
            low_vol_score += 0.4
        if 'bb_width' in volatility_features and volatility_features['bb_width'] < 0.05:
            low_vol_score += 0.3
        if 'atr' in volatility_features and volatility_features['atr'] < 0.01:
            low_vol_score += 0.3
        
        regime_scores[MarketRegime.LOW_VOLATILITY] = low_vol_score
        
        # RANGE_BOUND detection (default when no strong trends or extreme volatility)
        range_score = 1.0 - max(uptrend_score, downtrend_score, high_vol_score, low_vol_score)
        if 'trend_strength_24' in trend_features and trend_features['trend_strength_24'] < 0.3:
            range_score += 0.2
        if 'adx' in trend_features and trend_features['adx'] < 25:
            range_score += 0.2
        
        regime_scores[MarketRegime.RANGE_BOUND] = range_score
        
        # Determine primary regime
        primary_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[primary_regime]
        
        # Calculate transition probability and stability
        sorted_scores = sorted(regime_scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            transition_probability = 1 - (sorted_scores[0] - sorted_scores[1])
        else:
            transition_probability = 0.5
            
        # Calculate stability based on recent regime history
        stability_score = self._calculate_regime_stability(symbol, primary_regime)
        
        # Store regime history
        if symbol:
            if symbol not in self.regime_history:
                self.regime_history[symbol] = []
            self.regime_history[symbol].append({
                'timestamp': datetime.now(),
                'regime': primary_regime,
                'confidence': confidence
            })
            # Keep only recent history
            if len(self.regime_history[symbol]) > 100:
                self.regime_history[symbol] = self.regime_history[symbol][-100:]
        
        return RegimeSignal(
            regime=primary_regime,
            confidence=confidence,
            timestamp=datetime.now(),
            features=all_features,
            transition_probability=transition_probability,
            stability_score=stability_score
        )
    
    def _calculate_regime_stability(self, symbol: str, current_regime: MarketRegime) -> float:
        """Calculate regime stability score based on recent history"""
        if not symbol or symbol not in self.regime_history:
            return 0.5
        
        recent_regimes = self.regime_history[symbol][-self.config['regime_stability_window']:]
        if len(recent_regimes) < 3:
            return 0.5
        
        # Count regime consistency
        same_regime_count = sum(1 for r in recent_regimes if r['regime'] == current_regime)
        stability = same_regime_count / len(recent_regimes)
        
        return stability
    
    def calculate_regime_confidence(self, df: pd.DataFrame, symbol: str = None) -> float:
        """Calculate confidence score for current regime classification"""
        regime_signal = self.identify_regime(df, symbol)
        return regime_signal.confidence
    
    def get_adaptive_parameters(self, regime: MarketRegime, symbol: str = None) -> Dict:
        """
        Get strategy parameters adapted for current market regime
        获取适应当前市场体制的策略参数
        """
        base_params = self.adaptive_params[regime]
        
        # Additional symbol-specific adjustments
        if symbol == 'BTCUSDT':
            # Special optimization for BTCUSDT (our best performer)
            adjustments = {
                'rsi_low': base_params.rsi_low - 5,  # More strict entry
                'target_profit': base_params.target_profit * 0.8,  # Lower profit target
                'confidence_multiplier': base_params.confidence_multiplier * 1.1
            }
        elif symbol in ['ETHUSDT', 'SOLUSDT']:  # Major altcoins
            adjustments = {
                'volume_threshold': base_params.volume_threshold * 1.2,
                'stop_loss': base_params.stop_loss * 0.8  # Tighter stops
            }
        else:  # Other altcoins
            adjustments = {
                'volume_threshold': base_params.volume_threshold * 1.5,
                'stop_loss': base_params.stop_loss * 0.6,  # Much tighter stops
                'confidence_multiplier': base_params.confidence_multiplier * 0.8
            }
        
        # Create final parameter set
        final_params = {
            'rsi_low': adjustments.get('rsi_low', base_params.rsi_low),
            'rsi_high': adjustments.get('rsi_high', base_params.rsi_high),
            'dip_threshold': adjustments.get('dip_threshold', base_params.dip_threshold),
            'volume_threshold': adjustments.get('volume_threshold', base_params.volume_threshold),
            'target_profit': adjustments.get('target_profit', base_params.target_profit),
            'stop_loss': adjustments.get('stop_loss', base_params.stop_loss),
            'max_holding_minutes': adjustments.get('max_holding_minutes', base_params.max_holding_minutes),
            'confidence_multiplier': adjustments.get('confidence_multiplier', base_params.confidence_multiplier)
        }
        
        return final_params
    
    def should_trade_in_regime(self, regime: MarketRegime, confidence: float) -> bool:
        """Determine if trading should be enabled in current regime"""
        if confidence < self.confidence_threshold:
            return False
        
        # Trading rules by regime
        if regime == MarketRegime.STRONG_DOWNTREND and confidence > 0.8:
            return False  # Pause trading in strong bear markets
        elif regime == MarketRegime.HIGH_VOLATILITY and confidence > 0.9:
            return False  # Pause in extreme volatility
        elif regime == MarketRegime.TRANSITION:
            return False  # Wait for regime clarity
        
        return True
    
    def get_regime_description(self, regime: MarketRegime) -> str:
        """Get human-readable regime description"""
        descriptions = {
            MarketRegime.RANGE_BOUND: "Sideways/Consolidation Market - Optimal for DipMaster",
            MarketRegime.STRONG_UPTREND: "Strong Bull Market - Modified Entry Criteria",
            MarketRegime.STRONG_DOWNTREND: "Strong Bear Market - Reduced/Paused Trading",
            MarketRegime.HIGH_VOLATILITY: "High Volatility Environment - Enhanced Risk Controls",
            MarketRegime.LOW_VOLATILITY: "Low Volatility Grinding - Extended Holding Periods",
            MarketRegime.TRANSITION: "Market Regime Transition - Wait for Clarity"
        }
        return descriptions.get(regime, "Unknown Regime")
    
    def export_regime_analysis(self, symbol: str, df: pd.DataFrame, 
                              output_path: Optional[str] = None) -> Dict:
        """Export comprehensive regime analysis for a symbol"""
        regime_signal = self.identify_regime(df, symbol)
        adaptive_params = self.get_adaptive_parameters(regime_signal.regime, symbol)
        should_trade = self.should_trade_in_regime(regime_signal.regime, regime_signal.confidence)
        
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_regime': regime_signal.regime.value,
            'regime_description': self.get_regime_description(regime_signal.regime),
            'confidence': regime_signal.confidence,
            'stability_score': regime_signal.stability_score,
            'transition_probability': regime_signal.transition_probability,
            'should_trade': should_trade,
            'adaptive_parameters': adaptive_params,
            'regime_features': regime_signal.features,
            'regime_history': self.regime_history.get(symbol, [])[-10:]  # Last 10 regimes
        }
        
        if output_path:
            output_file = Path(output_path) / f"{symbol}_regime_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            self.logger.info(f"Regime analysis exported to {output_file}")
        
        return analysis

# Utility functions for integration
def create_regime_detector(config: Optional[Dict] = None) -> MarketRegimeDetector:
    """Factory function to create regime detector"""
    return MarketRegimeDetector(config)

def analyze_symbol_regime(symbol: str, df: pd.DataFrame, 
                         detector: Optional[MarketRegimeDetector] = None) -> Dict:
    """Analyze market regime for a single symbol"""
    if detector is None:
        detector = create_regime_detector()
    
    return detector.export_regime_analysis(symbol, df)

def get_regime_optimized_params(symbol: str, df: pd.DataFrame,
                               detector: Optional[MarketRegimeDetector] = None) -> Dict:
    """Get regime-optimized parameters for a symbol"""
    if detector is None:
        detector = create_regime_detector()
    
    regime_signal = detector.identify_regime(df, symbol)
    return detector.get_adaptive_parameters(regime_signal.regime, symbol)