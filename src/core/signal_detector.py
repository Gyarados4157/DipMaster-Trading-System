import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals"""
    ENTRY_DIP = "entry_dip"
    ENTRY_MOMENTUM = "entry_momentum"
    EXIT_BOUNDARY = "exit_boundary"
    EXIT_TARGET = "exit_target"
    EXIT_TIMEOUT = "exit_timeout"
    

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    signal_type: SignalType
    timestamp: datetime
    price: float
    confidence: float
    indicators: Dict
    reason: str
    action: str  # 'buy' or 'sell'


class RealTimeSignalDetector:
    """Real-time signal detection for DipMaster strategy"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.price_buffer = {}
        self.indicator_cache = {}
        self.active_positions = {}
        
        # Strategy parameters from analysis
        self.rsi_entry_range = (30, 50)
        self.dip_threshold = 0.002  # 0.2% below open
        self.max_holding_minutes = 180
        self.min_confidence = 0.6
        
    def update_price_data(self, symbol: str, data: Dict):
        """Update price buffer with new data"""
        if symbol not in self.price_buffer:
            self.price_buffer[symbol] = []
            
        self.price_buffer[symbol].append({
            'timestamp': data['timestamp'],
            'open': data['open'],
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'volume': data['volume']
        })
        
        # Keep only recent data (last 200 candles)
        if len(self.price_buffer[symbol]) > 200:
            self.price_buffer[symbol] = self.price_buffer[symbol][-200:]
            
    def calculate_indicators(self, symbol: str) -> Dict:
        """Calculate technical indicators"""
        if symbol not in self.price_buffer or len(self.price_buffer[symbol]) < 20:
            return {}
            
        df = pd.DataFrame(self.price_buffer[symbol])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Moving averages
        ma20 = df['close'].rolling(window=20).mean()
        ma50 = df['close'].rolling(window=50).mean() if len(df) >= 50 else None
        
        # Bollinger Bands
        std20 = df['close'].rolling(window=20).std()
        bb_upper = ma20 + (std20 * 2)
        bb_lower = ma20 - (std20 * 2)
        
        # Volume analysis
        volume_ma = df['volume'].rolling(window=20).mean()
        volume_ratio = df['volume'].iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
        
        # Price position
        current_price = df['close'].iloc[-1]
        open_price = df['open'].iloc[-1]
        
        indicators = {
            'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
            'ma20': ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else current_price,
            'ma50': ma50.iloc[-1] if ma50 is not None and not pd.isna(ma50.iloc[-1]) else current_price,
            'bb_upper': bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else current_price * 1.02,
            'bb_lower': bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else current_price * 0.98,
            'volume_ratio': volume_ratio,
            'price_vs_open': (current_price - open_price) / open_price,
            'current_price': current_price,
            'open_price': open_price
        }
        
        self.indicator_cache[symbol] = indicators
        return indicators
        
    def detect_entry_signal(self, symbol: str, current_price: float) -> Optional[TradingSignal]:
        """Detect entry signals based on DipMaster strategy"""
        indicators = self.calculate_indicators(symbol)
        
        if not indicators:
            return None
            
        signals_met = []
        confidence = 0.0
        
        # Check RSI condition (30-50 range)
        if self.rsi_entry_range[0] <= indicators['rsi'] <= self.rsi_entry_range[1]:
            signals_met.append("RSI in range")
            confidence += 0.3
            
        # Check dip buying condition (87.9% of entries)
        if indicators['price_vs_open'] < -self.dip_threshold:
            signals_met.append("Dip detected")
            confidence += 0.4
            
        # Check MA condition
        if current_price < indicators['ma20']:
            signals_met.append("Below MA20")
            confidence += 0.2
            
        # Check Bollinger Band condition
        if current_price < indicators['bb_lower']:
            signals_met.append("Below BB lower")
            confidence += 0.1
            
        # Volume spike
        if indicators['volume_ratio'] > 1.5:
            signals_met.append("Volume spike")
            confidence += 0.1
            
        # Generate signal if confidence is sufficient
        if confidence >= self.min_confidence:
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.ENTRY_DIP if "Dip detected" in signals_met else SignalType.ENTRY_MOMENTUM,
                timestamp=datetime.now(),
                price=current_price,
                confidence=min(confidence, 1.0),
                indicators=indicators,
                reason=", ".join(signals_met),
                action='buy'
            )
            
        return None
        
    def detect_exit_signal(self, symbol: str, position: Dict, current_price: float, 
                          is_boundary: bool = False) -> Optional[TradingSignal]:
        """Detect exit signals based on position and timing"""
        if symbol not in self.indicator_cache:
            indicators = self.calculate_indicators(symbol)
        else:
            indicators = self.indicator_cache[symbol]
            
        entry_time = position['entry_time']
        entry_price = position['entry_price']
        holding_time = (datetime.now() - entry_time).total_seconds() / 60
        pnl_percent = (current_price - entry_price) / entry_price * 100
        
        # Priority 1: 15-minute boundary exit (100% compliance)
        if is_boundary:
            # Check if we're in preferred exit slots (15-29min or 45-59min)
            minute_in_hour = datetime.now().minute
            is_preferred_slot = (15 <= minute_in_hour < 30) or (45 <= minute_in_hour < 60)
            
            # Exit if profitable or in preferred slot
            if pnl_percent > 0 or is_preferred_slot or holding_time > 30:
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.EXIT_BOUNDARY,
                    timestamp=datetime.now(),
                    price=current_price,
                    confidence=1.0 if pnl_percent > 0 else 0.8,
                    indicators=indicators,
                    reason=f"15-min boundary exit, PnL: {pnl_percent:.2f}%, Holding: {holding_time:.0f}min",
                    action='sell'
                )
                
        # Priority 2: Target reached (quick profit taking)
        if pnl_percent > 0.8:  # 0.8% profit target
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.EXIT_TARGET,
                timestamp=datetime.now(),
                price=current_price,
                confidence=0.9,
                indicators=indicators,
                reason=f"Target reached: {pnl_percent:.2f}%",
                action='sell'
            )
            
        # Priority 3: Timeout (max holding period)
        if holding_time > self.max_holding_minutes:
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.EXIT_TIMEOUT,
                timestamp=datetime.now(),
                price=current_price,
                confidence=0.7,
                indicators=indicators,
                reason=f"Max holding time exceeded: {holding_time:.0f}min",
                action='sell'
            )
            
        return None
        
    def register_position(self, symbol: str, entry_price: float, entry_time: datetime):
        """Register a new position"""
        self.active_positions[symbol] = {
            'entry_price': entry_price,
            'entry_time': entry_time,
            'symbol': symbol
        }
        
    def remove_position(self, symbol: str):
        """Remove closed position"""
        if symbol in self.active_positions:
            del self.active_positions[symbol]
            
    def get_signal_strength(self, signal: TradingSignal) -> str:
        """Get human-readable signal strength"""
        if signal.confidence >= 0.8:
            return "STRONG"
        elif signal.confidence >= 0.6:
            return "MEDIUM"
        else:
            return "WEAK"