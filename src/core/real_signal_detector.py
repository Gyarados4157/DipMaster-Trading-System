#!/usr/bin/env python3
"""
Real DipMaster Signal Detector
基于1184笔真实交易数据修正的信号检测器
不再依赖虚假的"逢跌买入"宣传，而是基于实际特征
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型"""
    ENTRY_RSI_REVERSAL = "rsi_reversal"  # RSI均值回归
    ENTRY_MOMENTUM = "momentum"          # 动量跟随
    EXIT_BOUNDARY = "boundary_exit"      # 边界出场
    EXIT_TARGET = "target_profit"        # 目标利润
    EXIT_TIMEOUT = "timeout"             # 超时出场


@dataclass
class RealTradingSignal:
    """基于真实数据的交易信号"""
    symbol: str
    signal_type: SignalType
    timestamp: datetime
    price: float
    confidence: float
    indicators: Dict
    reason: str
    action: str
    position_size_usd: float = 0
    

class RealDipMasterDetector:
    """基于真实交易数据的信号检测器"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 基于真实数据的策略参数 (来自1184笔交易分析)
        self.rsi_range = [25, 55]  # 实际范围更宽
        self.rsi_optimal = [40, 50]  # 最优RSI区间 (基于44.66平均值)
        
        # 真实的"逢跌"定义 (实际只有46.1%是真正逢跌)
        self.weak_dip_threshold = 0.001  # 0.1%的轻微回调
        self.true_dip_threshold = 0.003  # 0.3%的真正逢跌
        
        # 基于实际交易的币种偏好
        self.preferred_symbols = {
            'ICPUSDT': 1.0,   # 最活跃
            'XRPUSDT': 0.99,  # 次活跃
            'ALGOUSDT': 0.73, 
            'DOGEUSDT': 0.70,
            'SUIUSDT': 0.66,
            'IOTAUSDT': 0.57,
            'SOLUSDT': 0.44
        }
        
        # 基于实际数据的时间偏好
        self.preferred_time_slots = {
            1: 1.0,    # 15-29分钟 (30.8%最活跃)
            3: 0.86,   # 45-59分钟 (26.4%)
            2: 0.73,   # 30-44分钟 (22.4%) 
            0: 0.66    # 0-14分钟 (20.5%)
        }
        
        # 实际仓位规模 (基于真实数据)
        self.base_position_usd = 1843  # 实际平均仓位
        self.max_position_usd = 5122   # 实际最大仓位
        
        # 数据缓存
        self.price_buffer = {}
        self.indicator_cache = {}
        
    def calculate_indicators(self, symbol: str, df: pd.DataFrame) -> Dict:
        """计算技术指标"""
        if len(df) < 50:
            return {}
            
        # RSI (关键指标，92.4%准确率)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 移动平均线
        ma20 = df['close'].rolling(window=20).mean()
        ma50 = df['close'].rolling(window=50).mean()
        
        # 布林带
        std20 = df['close'].rolling(window=20).std()
        bb_upper = ma20 + (std20 * 2)
        bb_lower = ma20 - (std20 * 2)
        
        # 成交量分析
        volume_ma = df['volume'].rolling(window=20).mean()
        volume_ratio = df['volume'].iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
        
        # 价格位置分析
        current_price = df['close'].iloc[-1]
        candle_open = df['open'].iloc[-1]
        candle_high = df['high'].iloc[-5:].max()  # 最近5根K线最高价
        
        # 计算不同类型的"逢跌"
        price_vs_open = (current_price - candle_open) / candle_open
        price_vs_recent_high = (current_price - candle_high) / candle_high
        
        indicators = {
            'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
            'ma20': ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else current_price,
            'ma50': ma50.iloc[-1] if not pd.isna(ma50.iloc[-1]) else current_price,
            'bb_upper': bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else current_price * 1.02,
            'bb_lower': bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else current_price * 0.98,
            'volume_ratio': volume_ratio,
            'current_price': current_price,
            'candle_open': candle_open,
            'candle_high': candle_high,
            'price_vs_open': price_vs_open,
            'price_vs_recent_high': price_vs_recent_high,
            'is_weak_dip': price_vs_open < -self.weak_dip_threshold,
            'is_true_dip': price_vs_recent_high < -self.true_dip_threshold
        }
        
        return indicators
    
    def detect_real_entry_signal(self, symbol: str, df: pd.DataFrame, current_time: datetime = None) -> Optional[RealTradingSignal]:
        """基于真实交易特征检测入场信号"""
        
        indicators = self.calculate_indicators(symbol, df)
        if not indicators:
            return None
            
        # 使用传入的时间或K线时间
        if current_time is None:
            current_time = df.index[-1] if hasattr(df.index[-1], 'minute') else datetime.now()
            
        # 基础条件检查
        current_rsi = indicators['rsi']
        current_price = indicators['current_price']
        
        # 计算基础置信度
        confidence = 0.0
        signals_met = []
        
        # 1. RSI策略 (92.4%准确率，最重要的信号)
        rsi_score = 0
        if self.rsi_optimal[0] <= current_rsi <= self.rsi_optimal[1]:
            rsi_score = 0.5  # 最优RSI区间
            signals_met.append(f"RSI_OPTIMAL_{current_rsi:.1f}")
        elif self.rsi_range[0] <= current_rsi <= self.rsi_range[1]:
            rsi_score = 0.3  # 可接受RSI区间
            signals_met.append(f"RSI_ACCEPTABLE_{current_rsi:.1f}")
        else:
            return None  # RSI不在范围内，直接退出
            
        confidence += rsi_score
        
        # 2. 币种偏好系数 (基于实际交易频率)
        symbol_preference = self.preferred_symbols.get(symbol, 0.1)  # 未知币种默认0.1
        confidence += 0.2 * symbol_preference
        if symbol_preference > 0.5:
            signals_met.append(f"PREFERRED_SYMBOL_{symbol_preference:.1f}")
        
        # 3. 时间窗口偏好 (基于实际交易时间)
        current_minute = current_time.minute
        time_slot = current_minute // 15
        time_preference = self.preferred_time_slots.get(time_slot, 0.5)
        confidence += 0.15 * time_preference
        if time_preference > 0.8:
            signals_met.append(f"OPTIMAL_TIME_SLOT_{time_slot}")
        
        # 4. 价格位置分析 (修正版逢跌检测)
        dip_score = 0
        if indicators['is_true_dip']:
            dip_score = 0.2
            signals_met.append("TRUE_DIP")
        elif indicators['is_weak_dip']:
            dip_score = 0.1  
            signals_met.append("WEAK_DIP")
        else:
            # 即使不是逢跌也可以交易 (符合46.1%的实际情况)
            dip_score = 0.05
            signals_met.append("NO_DIP_OK")
            
        confidence += dip_score
        
        # 5. 技术面确认
        tech_score = 0
        if current_price < indicators['ma20']:
            tech_score += 0.1
            signals_met.append("BELOW_MA20")
        if current_price < indicators['bb_lower']:
            tech_score += 0.05
            signals_met.append("BELOW_BB_LOWER")
        if indicators['volume_ratio'] > 1.2:
            tech_score += 0.05
            signals_met.append("VOLUME_CONFIRM")
            
        confidence += tech_score
        
        # 信号强度判断
        if confidence >= 0.6:
            # 计算仓位规模
            position_size = self.calculate_position_size(confidence, symbol_preference)
            
            return RealTradingSignal(
                symbol=symbol,
                signal_type=SignalType.ENTRY_RSI_REVERSAL,
                timestamp=current_time,
                price=current_price,
                confidence=min(confidence, 1.0),
                indicators=indicators,
                reason=" + ".join(signals_met),
                action='buy',
                position_size_usd=position_size
            )
        
        return None
    
    def calculate_position_size(self, confidence: float, symbol_preference: float) -> float:
        """基于真实数据计算仓位规模"""
        
        # 基础仓位 (基于实际平均仓位1843美元)
        base_size = self.base_position_usd
        
        # 置信度调整 (0.6-1.0 -> 0.8-1.2倍)
        confidence_multiplier = 0.8 + (confidence - 0.6) * 1.0  
        
        # 币种偏好调整
        symbol_multiplier = 0.5 + symbol_preference * 0.8
        
        # 计算最终仓位
        final_size = base_size * confidence_multiplier * symbol_multiplier
        
        # 限制在实际范围内
        return min(max(final_size, 500), self.max_position_usd)
    
    def detect_exit_signal(self, symbol: str, position: Dict, df: pd.DataFrame, 
                          is_boundary: bool = False) -> Optional[RealTradingSignal]:
        """基于真实交易模式检测出场信号"""
        
        indicators = self.calculate_indicators(symbol, df)
        if not indicators:
            return None
            
        entry_time = position['entry_time']
        entry_price = position['entry_price']
        current_price = indicators['current_price']
        
        holding_minutes = (datetime.now() - entry_time).total_seconds() / 60
        pnl_percent = (current_price - entry_price) / entry_price * 100
        
        # 基于实际交易时间模式的出场策略
        current_minute = datetime.now().minute
        time_slot = current_minute // 15
        
        # 1. 15分钟边界出场 (基于实际30.8%在15-29分钟交易)
        if is_boundary and time_slot == 1:  # 在最活跃时段
            if pnl_percent > 0.2 or holding_minutes > 20:  # 微利或持仓超20分钟
                return RealTradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.EXIT_BOUNDARY,
                    timestamp=datetime.now(),
                    price=current_price,
                    confidence=0.9,
                    indicators=indicators,
                    reason=f"OPTIMAL_BOUNDARY_EXIT_PnL_{pnl_percent:.2f}%",
                    action='sell'
                )
        
        # 2. 快速获利出场 (基于中小盘波动特性)
        target_profit = 0.8  # 0.8%目标利润
        if pnl_percent >= target_profit:
            return RealTradingSignal(
                symbol=symbol,
                signal_type=SignalType.EXIT_TARGET,
                timestamp=datetime.now(),
                price=current_price,
                confidence=0.95,
                indicators=indicators,
                reason=f"TARGET_REACHED_{pnl_percent:.2f}%",
                action='sell'
            )
        
        # 3. 超时出场 (基于实际持仓时间)
        max_holding = 90  # 1.5小时最大持仓
        if holding_minutes > max_holding:
            return RealTradingSignal(
                symbol=symbol,
                signal_type=SignalType.EXIT_TIMEOUT,
                timestamp=datetime.now(),
                price=current_price,
                confidence=0.7,
                indicators=indicators,
                reason=f"TIMEOUT_{holding_minutes:.0f}min",
                action='sell'
            )
        
        # 4. 止损出场 (保护资本)
        if pnl_percent <= -2.0:  # 2%止损
            return RealTradingSignal(
                symbol=symbol,
                signal_type=SignalType.EXIT_TIMEOUT,
                timestamp=datetime.now(),
                price=current_price,
                confidence=1.0,
                indicators=indicators,
                reason=f"STOP_LOSS_{pnl_percent:.2f}%",
                action='sell'
            )
        
        return None
    
    def get_signal_quality(self, signal: RealTradingSignal) -> str:
        """获取信号质量评级"""
        if signal.confidence >= 0.85:
            return "EXCELLENT"
        elif signal.confidence >= 0.75:
            return "GOOD"
        elif signal.confidence >= 0.65:
            return "FAIR"
        else:
            return "WEAK"
    
    def is_market_suitable(self, market_data: Dict) -> bool:
        """判断当前市场环境是否适合交易"""
        
        # 基于实际交易的币种，检查是否有活跃的中小盘币种
        active_symbols = 0
        for symbol in self.preferred_symbols.keys():
            if symbol in market_data and market_data[symbol].get('volume_24h', 0) > 1000000:
                active_symbols += 1
        
        # 至少要有3个偏好币种活跃
        if active_symbols < 3:
            logger.warning("Market not suitable: insufficient active preferred symbols")
            return False
            
        # 检查时间窗口
        current_minute = datetime.now().minute
        time_slot = current_minute // 15
        time_preference = self.preferred_time_slots.get(time_slot, 0.5)
        
        if time_preference < 0.7:
            logger.info(f"Sub-optimal trading time slot: {time_slot} (preference: {time_preference})")
            
        return True