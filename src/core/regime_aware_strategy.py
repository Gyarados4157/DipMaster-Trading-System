#!/usr/bin/env python3
"""
Regime-Aware DipMaster Strategy Integration Module
市场体制感知DipMaster策略集成模块

This module integrates the market regime detection system with the existing
DipMaster strategy to create an adaptive trading system that addresses the
core issue of poor performance in trending markets.

Integration Components:
1. Regime-aware signal detection
2. Adaptive parameter management  
3. Risk management by regime
4. Performance monitoring and optimization
5. Real-time regime adaptation

Target: Improve BTCUSDT win rate from 47.7% to 65%+

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
import logging
import json
from pathlib import Path
import asyncio
from enum import Enum

# Import existing DipMaster components
from .signal_detector import RealTimeSignalDetector, TradingSignal, SignalType
from .market_regime_detector import MarketRegimeDetector, MarketRegime, RegimeSignal
from .position_manager import PositionManager
from .risk_manager import RiskManager

warnings.filterwarnings('ignore')

@dataclass
class RegimeAdaptiveConfig:
    """Configuration for regime-aware strategy"""
    enable_regime_adaptation: bool = True
    regime_confidence_threshold: float = 0.7
    regime_stability_threshold: float = 0.6
    parameter_adaptation_speed: float = 0.5
    risk_scaling_by_regime: bool = True
    position_sizing_by_regime: bool = True

class RegimeAwareSignalDetector(RealTimeSignalDetector):
    """
    Enhanced signal detector with market regime awareness
    具备市场体制感知能力的增强版信号检测器
    """
    
    def __init__(self, config: Dict, regime_detector: MarketRegimeDetector = None,
                 adaptive_params: Dict = None):
        """Initialize regime-aware signal detector"""
        super().__init__(config)
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.adaptive_params = adaptive_params or self._load_adaptive_params()
        self.current_regimes = {}
        self.regime_confidence = {}
        self.adaptive_config = RegimeAdaptiveConfig()
        
        # Enhanced tracking
        self.regime_performance = {}
        self.parameter_history = {}
        
    def _load_adaptive_params(self) -> Dict:
        """Load adaptive parameters from configuration"""
        try:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'regime_adaptive_parameters.json'
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return config_data['regime_parameters']
        except Exception as e:
            self.logger.warning(f"Could not load adaptive parameters: {e}")
            return {}
    
    def update_price_data_with_regime(self, symbol: str, data: Dict):
        """Update price data and detect current market regime"""
        # Update base price data
        super().update_price_data(symbol, data)
        
        # Detect current regime if we have sufficient data
        if symbol in self.price_buffer and len(self.price_buffer[symbol]) >= 50:
            df = pd.DataFrame(self.price_buffer[symbol])
            regime_signal = self.regime_detector.identify_regime(df, symbol)
            
            self.current_regimes[symbol] = regime_signal.regime
            self.regime_confidence[symbol] = regime_signal.confidence
            
            # Log regime changes
            if symbol in self.regime_performance:
                last_regime = self.regime_performance[symbol].get('last_regime')
                if last_regime != regime_signal.regime:
                    self.logger.info(f"{symbol} regime changed: {last_regime} -> {regime_signal.regime} "
                                   f"(confidence: {regime_signal.confidence:.2f})")
            
            # Update performance tracking
            self.regime_performance[symbol] = {
                'last_regime': regime_signal.regime,
                'confidence': regime_signal.confidence,
                'stability': regime_signal.stability_score,
                'last_update': datetime.now()
            }
    
    def get_adaptive_parameters(self, symbol: str) -> Dict:
        """Get regime-adapted parameters for the symbol"""
        if not self.adaptive_config.enable_regime_adaptation:
            return self.config
        
        # Get current regime
        current_regime = self.current_regimes.get(symbol, MarketRegime.RANGE_BOUND)
        confidence = self.regime_confidence.get(symbol, 0.5)
        
        # Check if we should use adaptive parameters
        if confidence < self.adaptive_config.regime_confidence_threshold:
            # Use default parameters if confidence is low
            self.logger.debug(f"{symbol}: Using default params (low regime confidence: {confidence:.2f})")
            return self.config
        
        # Get regime-specific parameters
        if current_regime.value in self.adaptive_params:
            regime_params = self.adaptive_params[current_regime.value]['base_params'].copy()
            
            # Apply symbol-specific adjustments
            symbol_adjustments = self.adaptive_params[current_regime.value].get('symbol_adjustments', {})
            
            if symbol in symbol_adjustments:
                regime_params.update(symbol_adjustments[symbol])
            elif symbol not in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'] and 'altcoins' in symbol_adjustments:
                regime_params.update(symbol_adjustments['altcoins'])
            
            # Blend with current parameters based on adaptation speed
            blended_params = {}
            for key, new_value in regime_params.items():
                if key in self.config:
                    current_value = getattr(self, key, self.config.get(key, new_value))
                    if isinstance(current_value, (int, float)) and isinstance(new_value, (int, float)):
                        # Smooth parameter transitions
                        blended_value = (current_value * (1 - self.adaptive_config.parameter_adaptation_speed) +
                                       new_value * self.adaptive_config.parameter_adaptation_speed)
                        blended_params[key] = blended_value
                    else:
                        blended_params[key] = new_value
                else:
                    blended_params[key] = new_value
            
            # Track parameter changes
            self.parameter_history[symbol] = {
                'regime': current_regime.value,
                'confidence': confidence,
                'parameters': blended_params.copy(),
                'timestamp': datetime.now()
            }
            
            return blended_params
        
        return self.config
    
    def should_trade_in_current_regime(self, symbol: str) -> bool:
        """Determine if trading should be enabled based on current regime"""
        current_regime = self.current_regimes.get(symbol, MarketRegime.RANGE_BOUND)
        confidence = self.regime_confidence.get(symbol, 0.5)
        
        # Check trading rules from configuration
        try:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'regime_adaptive_parameters.json'
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            disable_rules = config_data['trading_rules']['disable_trading_regimes']
            
            for rule in disable_rules:
                if (rule['regime'] == current_regime.value and 
                    confidence >= rule['confidence_threshold']):
                    self.logger.info(f"{symbol}: Trading disabled - {rule['reason']} "
                                   f"(regime: {current_regime.value}, confidence: {confidence:.2f})")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Could not check trading rules: {e}")
            return True
    
    def detect_entry_signal_adaptive(self, symbol: str, current_price: float) -> Optional[TradingSignal]:
        """Detect entry signals with regime adaptation"""
        if not self.should_trade_in_current_regime(symbol):
            return None
        
        # Get adaptive parameters
        adaptive_params = self.get_adaptive_parameters(symbol)
        
        # Temporarily update strategy parameters
        original_params = {
            'rsi_entry_range': self.rsi_entry_range,
            'dip_threshold': self.dip_threshold,
            'min_confidence': self.min_confidence
        }
        
        try:
            # Apply adaptive parameters
            if 'rsi_low' in adaptive_params and 'rsi_high' in adaptive_params:
                self.rsi_entry_range = (adaptive_params['rsi_low'], adaptive_params['rsi_high'])
            if 'dip_threshold' in adaptive_params:
                self.dip_threshold = adaptive_params['dip_threshold']
            if 'confidence_multiplier' in adaptive_params:
                self.min_confidence = self.min_confidence * adaptive_params['confidence_multiplier']
            
            # Detect signal with adaptive parameters
            signal = self.detect_entry_signal(symbol, current_price)
            
            if signal:
                # Enhance signal with regime information
                current_regime = self.current_regimes.get(symbol, MarketRegime.RANGE_BOUND)
                signal.indicators['regime'] = current_regime.value
                signal.indicators['regime_confidence'] = self.regime_confidence.get(symbol, 0.5)
                signal.reason += f" [Regime: {current_regime.value}]"
                
                # Adjust signal confidence based on regime suitability
                if current_regime == MarketRegime.RANGE_BOUND:
                    signal.confidence *= 1.1  # Boost confidence in optimal regime
                elif current_regime in [MarketRegime.STRONG_DOWNTREND, MarketRegime.HIGH_VOLATILITY]:
                    signal.confidence *= 0.8  # Reduce confidence in challenging regimes
                
                signal.confidence = min(signal.confidence, 1.0)
            
            return signal
            
        finally:
            # Restore original parameters
            self.rsi_entry_range = original_params['rsi_entry_range']
            self.dip_threshold = original_params['dip_threshold']
            self.min_confidence = original_params['min_confidence']
    
    def detect_exit_signal_adaptive(self, symbol: str, position: Dict, current_price: float,
                                  is_boundary: bool = False) -> Optional[TradingSignal]:
        """Detect exit signals with regime adaptation"""
        # Get adaptive parameters for exit logic
        adaptive_params = self.get_adaptive_parameters(symbol)
        current_regime = self.current_regimes.get(symbol, MarketRegime.RANGE_BOUND)
        
        # Apply regime-specific exit logic
        original_max_holding = self.max_holding_minutes
        try:
            if 'max_holding_minutes' in adaptive_params:
                self.max_holding_minutes = adaptive_params['max_holding_minutes']
            
            # Get base exit signal
            signal = self.detect_exit_signal(symbol, position, current_price, is_boundary)
            
            if signal:
                # Enhance with regime information
                signal.indicators['regime'] = current_regime.value
                signal.indicators['regime_confidence'] = self.regime_confidence.get(symbol, 0.5)
                
                # Regime-specific exit adjustments
                if current_regime == MarketRegime.HIGH_VOLATILITY:
                    # Earlier exits in high volatility
                    if signal.signal_type == SignalType.EXIT_BOUNDARY:
                        signal.confidence *= 1.2
                elif current_regime == MarketRegime.LOW_VOLATILITY:
                    # Later exits in low volatility
                    if signal.signal_type == SignalType.EXIT_BOUNDARY:
                        signal.confidence *= 0.9
            
            return signal
            
        finally:
            self.max_holding_minutes = original_max_holding
    
    def get_regime_performance_summary(self, symbol: str = None) -> Dict:
        """Get performance summary by regime"""
        if symbol:
            return self.regime_performance.get(symbol, {})
        else:
            return self.regime_performance

class RegimeAwarePositionManager(PositionManager):
    """
    Enhanced position manager with regime-aware sizing and risk management
    具备体制感知能力的增强版仓位管理器
    """
    
    def __init__(self, config: Dict, regime_detector: MarketRegimeDetector = None):
        super().__init__(config)
        self.regime_detector = regime_detector
        self.regime_position_limits = self._load_regime_position_limits()
    
    def _load_regime_position_limits(self) -> Dict:
        """Load regime-specific position limits"""
        try:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'regime_adaptive_parameters.json'
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return config_data['trading_rules']['max_positions_per_regime']
        except Exception as e:
            self.logger.warning(f"Could not load regime position limits: {e}")
            return {'default': 3}
    
    def can_open_position(self, symbol: str, regime: MarketRegime = None) -> bool:
        """Check if new position can be opened considering regime limits"""
        # Base position checks
        if not super().can_open_position(symbol):
            return False
        
        if regime is None:
            return True
        
        # Check regime-specific limits
        regime_limit = self.regime_position_limits.get(regime.value, 3)
        if regime_limit == 0:
            return False
        
        # Count current positions in this regime
        current_regime_positions = sum(
            1 for pos in self.active_positions.values()
            if pos.get('regime') == regime.value
        )
        
        return current_regime_positions < regime_limit
    
    def calculate_position_size_adaptive(self, symbol: str, signal: TradingSignal,
                                       regime: MarketRegime = None) -> float:
        """Calculate position size adapted to market regime"""
        base_size = self.calculate_position_size(symbol, signal.price)
        
        if regime is None:
            return base_size
        
        # Regime-specific position sizing adjustments
        size_multipliers = {
            MarketRegime.RANGE_BOUND: 1.0,      # Standard size
            MarketRegime.STRONG_UPTREND: 1.2,   # Larger positions in trends
            MarketRegime.STRONG_DOWNTREND: 0.5, # Smaller positions in bear markets
            MarketRegime.HIGH_VOLATILITY: 0.7,  # Smaller positions in volatility
            MarketRegime.LOW_VOLATILITY: 1.1,   # Slightly larger in low vol
            MarketRegime.TRANSITION: 0.6        # Conservative during transitions
        }
        
        multiplier = size_multipliers.get(regime, 1.0)
        
        # Additional adjustment based on regime confidence
        regime_confidence = signal.indicators.get('regime_confidence', 0.5)
        confidence_adjustment = 0.5 + (regime_confidence * 0.5)  # 0.5 to 1.0 range
        
        adjusted_size = base_size * multiplier * confidence_adjustment
        
        self.logger.debug(f"{symbol}: Position size adjusted for {regime.value} "
                         f"(base: {base_size}, multiplier: {multiplier:.2f}, "
                         f"confidence_adj: {confidence_adjustment:.2f}, final: {adjusted_size})")
        
        return adjusted_size

class RegimeAwareDipMasterStrategy:
    """
    Main regime-aware DipMaster strategy orchestrator
    主要的市场体制感知DipMaster策略协调器
    """
    
    def __init__(self, config: Dict):
        """Initialize the regime-aware strategy"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize components
        self.regime_detector = MarketRegimeDetector()
        self.signal_detector = RegimeAwareSignalDetector(config, self.regime_detector)
        self.position_manager = RegimeAwarePositionManager(config, self.regime_detector)
        self.risk_manager = RiskManager(config)
        
        # Performance tracking
        self.performance_by_regime = {}
        self.trade_history = []
        
        self.logger.info("Regime-aware DipMaster strategy initialized")
    
    async def process_market_data(self, symbol: str, market_data: Dict) -> Dict:
        """Process market data and generate trading decisions"""
        try:
            # Update price data with regime detection
            self.signal_detector.update_price_data_with_regime(symbol, market_data)
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'regime': None,
                'regime_confidence': 0.0,
                'signals': [],
                'positions': [],
                'actions': []
            }
            
            # Get current regime
            current_regime = self.signal_detector.current_regimes.get(symbol)
            regime_confidence = self.signal_detector.regime_confidence.get(symbol, 0.0)
            
            result['regime'] = current_regime.value if current_regime else 'UNKNOWN'
            result['regime_confidence'] = regime_confidence
            
            current_price = market_data['close']
            
            # Check for entry signals
            if self.position_manager.can_open_position(symbol, current_regime):
                entry_signal = self.signal_detector.detect_entry_signal_adaptive(symbol, current_price)
                
                if entry_signal:
                    # Calculate position size
                    position_size = self.position_manager.calculate_position_size_adaptive(
                        symbol, entry_signal, current_regime
                    )
                    
                    # Risk management check
                    if self.risk_manager.can_open_position(symbol, position_size, current_price):
                        # Execute entry
                        action = {
                            'type': 'ENTRY',
                            'symbol': symbol,
                            'signal': entry_signal,
                            'position_size': position_size,
                            'regime': current_regime.value if current_regime else 'UNKNOWN'
                        }
                        result['actions'].append(action)
                        result['signals'].append(entry_signal)
                        
                        self.logger.info(f"Entry signal: {symbol} @ {current_price} "
                                       f"(regime: {current_regime.value if current_regime else 'UNKNOWN'}, "
                                       f"confidence: {entry_signal.confidence:.2f})")
            
            # Check for exit signals on existing positions
            active_positions = self.position_manager.get_active_positions(symbol)
            
            for position in active_positions:
                # Check if we're at a 15-minute boundary
                minutes_since_hour = datetime.now().minute
                is_boundary = minutes_since_hour % 15 == 0
                
                exit_signal = self.signal_detector.detect_exit_signal_adaptive(
                    symbol, position, current_price, is_boundary
                )
                
                if exit_signal:
                    # Execute exit
                    action = {
                        'type': 'EXIT',
                        'symbol': symbol,
                        'signal': exit_signal,
                        'position': position,
                        'regime': current_regime.value if current_regime else 'UNKNOWN'
                    }
                    result['actions'].append(action)
                    result['signals'].append(exit_signal)
                    
                    # Track performance by regime
                    self._track_regime_performance(symbol, position, exit_signal, current_regime)
                    
                    self.logger.info(f"Exit signal: {symbol} @ {current_price} "
                                   f"(regime: {current_regime.value if current_regime else 'UNKNOWN'}, "
                                   f"reason: {exit_signal.reason})")
            
            result['positions'] = active_positions
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing market data for {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}
    
    def _track_regime_performance(self, symbol: str, position: Dict, 
                                exit_signal: TradingSignal, regime: MarketRegime):
        """Track performance by regime for analysis"""
        if regime is None:
            return
        
        entry_price = position['entry_price']
        exit_price = exit_signal.price
        pnl = (exit_price - entry_price) / entry_price
        
        regime_key = regime.value
        if regime_key not in self.performance_by_regime:
            self.performance_by_regime[regime_key] = {
                'trades': 0,
                'wins': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_pnl': 0.0
            }
        
        stats = self.performance_by_regime[regime_key]
        stats['trades'] += 1
        stats['total_pnl'] += pnl
        
        if pnl > 0:
            stats['wins'] += 1
        
        stats['win_rate'] = stats['wins'] / stats['trades']
        stats['avg_pnl'] = stats['total_pnl'] / stats['trades']
        
        # Log performance milestones
        if stats['trades'] % 10 == 0:
            self.logger.info(f"Regime {regime_key} performance: "
                           f"{stats['trades']} trades, "
                           f"{stats['win_rate']:.1%} win rate, "
                           f"{stats['avg_pnl']:.3%} avg PnL")
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary by regime"""
        return {
            'by_regime': self.performance_by_regime,
            'regime_detector_summary': self.signal_detector.get_regime_performance_summary(),
            'total_trades': sum(stats['trades'] for stats in self.performance_by_regime.values()),
            'overall_performance': self._calculate_overall_performance()
        }
    
    def _calculate_overall_performance(self) -> Dict:
        """Calculate overall strategy performance"""
        if not self.performance_by_regime:
            return {}
        
        total_trades = sum(stats['trades'] for stats in self.performance_by_regime.values())
        total_wins = sum(stats['wins'] for stats in self.performance_by_regime.values())
        total_pnl = sum(stats['total_pnl'] for stats in self.performance_by_regime.values())
        
        return {
            'total_trades': total_trades,
            'win_rate': total_wins / total_trades if total_trades > 0 else 0,
            'avg_pnl': total_pnl / total_trades if total_trades > 0 else 0,
            'total_return': total_pnl
        }
    
    def export_regime_analysis(self, output_dir: str) -> str:
        """Export comprehensive regime analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_file = output_path / f"regime_aware_strategy_analysis_{timestamp}.json"
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'performance_summary': self.get_performance_summary(),
            'regime_transitions': self.signal_detector.regime_performance,
            'parameter_adaptations': self.signal_detector.parameter_history,
            'configuration': self.config
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self.logger.info(f"Regime analysis exported to {analysis_file}")
        return str(analysis_file)

# Utility functions for easy integration
def create_regime_aware_strategy(config: Dict) -> RegimeAwareDipMasterStrategy:
    """Factory function to create regime-aware strategy"""
    return RegimeAwareDipMasterStrategy(config)

def load_adaptive_config() -> Dict:
    """Load adaptive configuration from file"""
    try:
        config_path = Path(__file__).parent.parent.parent / 'config' / 'regime_adaptive_parameters.json'
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.getLogger(__name__).error(f"Could not load adaptive config: {e}")
        return {}