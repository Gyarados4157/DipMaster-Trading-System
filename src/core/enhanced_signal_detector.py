"""
Enhanced Signal Detection System - Phase 1 Optimization
6层过滤系统，大幅提升信号质量和胜率

核心改进：
1. RSI严格化：35-42
2. 趋势过滤：避免强下跌  
3. 动量确认：下跌减缓
4. 成交量确认：放量下跌
5. 布林带位置：接近下轨
6. 多时间框架确认

目标：将胜率从69.5%提升至78-82%
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """市场状态"""
    HIGH_VOLATILITY = "high_vol"      # 高波动，暂停交易
    STRONG_DOWNTREND = "strong_down"  # 强势下跌，减少交易
    STRONG_UPTREND = "strong_up"      # 强势上涨，谨慎交易
    SIDEWAYS = "sideways"             # 横盘，正常交易
    SUITABLE = "suitable"             # 适合抄底，正常交易


@dataclass
class SignalConfidence:
    """信号置信度评分"""
    rsi_score: float = 0.0         # RSI评分
    trend_score: float = 0.0       # 趋势评分
    momentum_score: float = 0.0    # 动量评分
    volume_score: float = 0.0      # 成交量评分
    bollinger_score: float = 0.0   # 布林带评分
    timeframe_score: float = 0.0   # 多时间框架评分
    
    @property
    def total_score(self) -> float:
        """总置信度评分"""
        return (self.rsi_score + self.trend_score + self.momentum_score + 
                self.volume_score + self.bollinger_score + self.timeframe_score) / 6
    
    @property
    def grade(self) -> str:
        """信号等级"""
        if self.total_score >= 0.8:
            return "A"
        elif self.total_score >= 0.6:
            return "B" 
        elif self.total_score >= 0.4:
            return "C"
        else:
            return "D"


class EnhancedSignalDetector:
    """增强信号检测器 - 6层过滤系统"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # === 严格化参数 ===
        self.rsi_range = (35, 42)  # 收紧RSI范围
        self.rsi_optimal = 38      # 最优RSI值
        
        # === 趋势过滤参数 ===
        self.max_consecutive_red = 4   # 最多连续4根阴线
        self.trend_ema_period = 10     # 趋势EMA周期
        self.trend_threshold = -0.02   # 趋势强度阈值-2%
        
        # === 动量参数 ===
        self.momentum_periods = [3, 5, 10]  # 多周期动量
        self.momentum_decay_threshold = 0.5  # 动量衰减阈值
        
        # === 成交量参数 ===
        self.volume_ma_period = 20     # 成交量均线周期
        self.volume_multiplier = 1.5   # 成交量倍数要求
        self.volume_surge_threshold = 2.0  # 成交量激增阈值
        
        # === 布林带参数 ===
        self.bb_period = 20           # 布林带周期
        self.bb_std = 2.0             # 标准差倍数
        self.bb_lower_zone = 0.2      # 下轨区间20%
        
        # === 多时间框架 ===
        self.use_higher_timeframe = True
        self.higher_tf_multiplier = 3  # 3倍时间框架
        
        # 统计数据
        self.signal_stats = {
            'total_signals': 0,
            'filtered_signals': 0,
            'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        }
        
    def detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """识别当前市场状态"""
        if len(df) < 20:
            return MarketRegime.SUITABLE
            
        # 计算ATR（波动率）
        atr = self._calculate_atr(df)
        baseline_atr = df['close'].rolling(50).std().iloc[-1] / df['close'].iloc[-1]
        volatility_ratio = atr / baseline_atr if baseline_atr > 0 else 1
        
        # 高波动率检测
        if volatility_ratio > 3.0:
            return MarketRegime.HIGH_VOLATILITY
            
        # 趋势强度检测
        ema10 = df['close'].ewm(span=10).mean()
        ema20 = df['close'].ewm(span=20).mean()
        trend_strength = (ema10.iloc[-1] - ema20.iloc[-1]) / ema20.iloc[-1]
        
        if trend_strength < -0.03:  # 强势下跌
            return MarketRegime.STRONG_DOWNTREND
        elif trend_strength > 0.03:  # 强势上涨
            return MarketRegime.STRONG_UPTREND
        
        # 横盘检测
        price_range = (df['high'].rolling(10).max().iloc[-1] - 
                      df['low'].rolling(10).min().iloc[-1]) / df['close'].iloc[-1]
        
        if price_range < 0.02:  # 2%以内横盘
            return MarketRegime.SIDEWAYS
            
        return MarketRegime.SUITABLE
        
    def layer1_rsi_filter(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """第1层：严格RSI过滤"""
        if len(df) < 20:
            return False, 0.0
            
        rsi = self._calculate_rsi(df['close'])[-1]
        
        # 严格RSI范围检查
        if not (self.rsi_range[0] <= rsi <= self.rsi_range[1]):
            return False, 0.0
            
        # RSI评分：越接近38分数越高
        distance_from_optimal = abs(rsi - self.rsi_optimal)
        score = max(0, 1 - distance_from_optimal / 7)  # 7是RSI范围的一半
        
        return True, score
        
    def layer2_trend_filter(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """第2层：趋势过滤，避免强势下跌"""
        if len(df) < 10:
            return False, 0.0
            
        # 检查连续阴线
        recent_closes = df['close'].iloc[-5:].values
        recent_opens = df['open'].iloc[-5:].values
        consecutive_red = 0
        
        for i in range(len(recent_closes)-1, -1, -1):
            if recent_closes[i] < recent_opens[i]:
                consecutive_red += 1
            else:
                break
                
        if consecutive_red > self.max_consecutive_red:
            return False, 0.0
            
        # 计算趋势EMA
        ema_trend = df['close'].ewm(span=self.trend_ema_period).mean()
        trend_change = (ema_trend.iloc[-1] - ema_trend.iloc[-5]) / ema_trend.iloc[-5]
        
        if trend_change < self.trend_threshold:  # 强势下跌
            return False, 0.0
            
        # 趋势评分：下跌但不过于剧烈的得分高
        if -0.01 <= trend_change <= 0:  # 轻微下跌最佳
            score = 1.0
        elif -0.02 <= trend_change < -0.01:  # 中等下跌
            score = 0.7
        else:  # 其他情况
            score = 0.4
            
        return True, score
        
    def layer3_momentum_filter(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """第3层：动量确认，当前跌幅需要小于前期"""
        if len(df) < 10:
            return False, 0.0
            
        # 计算多周期动量
        momentums = []
        for period in self.momentum_periods:
            if len(df) >= period:
                momentum = (df['close'].iloc[-1] - df['close'].iloc[-period]) / df['close'].iloc[-period]
                momentums.append(momentum)
                
        if not momentums:
            return False, 0.0
            
        # 检查动量是否在衰减（下跌减缓）
        current_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        prev_change = (df['close'].iloc[-2] - df['close'].iloc[-3]) / df['close'].iloc[-3]
        
        # 当前跌幅应小于前一根K线
        if current_change >= prev_change:  # 下跌减缓或转涨
            momentum_score = 1.0
        else:
            return False, 0.0
            
        # 综合多周期动量评分
        avg_momentum = np.mean(momentums)
        if -0.02 <= avg_momentum <= -0.005:  # 轻微下跌动量最佳
            momentum_score *= 1.0
        elif avg_momentum < -0.02:  # 下跌过快
            momentum_score *= 0.6
        else:  # 上涨动量
            momentum_score *= 0.8
            
        return True, momentum_score
        
    def layer4_volume_filter(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """第4层：成交量确认"""
        if len(df) < self.volume_ma_period:
            return False, 0.0
            
        current_volume = df['volume'].iloc[-1]
        volume_ma = df['volume'].rolling(self.volume_ma_period).mean().iloc[-1]
        
        if volume_ma <= 0:
            return False, 0.0
            
        volume_ratio = current_volume / volume_ma
        
        # 成交量必须放大
        if volume_ratio < self.volume_multiplier:
            return False, 0.0
            
        # 成交量评分
        if volume_ratio >= self.volume_surge_threshold:  # 成交量激增
            score = 1.0
        elif volume_ratio >= self.volume_multiplier * 1.3:  # 显著放量
            score = 0.8
        else:  # 一般放量
            score = 0.6
            
        return True, score
        
    def layer5_bollinger_filter(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """第5层：布林带位置确认"""
        if len(df) < self.bb_period:
            return False, 0.0
            
        # 计算布林带
        ma = df['close'].rolling(self.bb_period).mean()
        std = df['close'].rolling(self.bb_period).std()
        bb_upper = ma + (std * self.bb_std)
        bb_lower = ma - (std * self.bb_std)
        
        current_price = df['close'].iloc[-1]
        bb_lower_val = bb_lower.iloc[-1]
        bb_upper_val = bb_upper.iloc[-1]
        
        # 价格必须接近布林带下轨
        bb_range = bb_upper_val - bb_lower_val
        distance_to_lower = current_price - bb_lower_val
        position_ratio = distance_to_lower / bb_range if bb_range > 0 else 1
        
        # 必须在下轨附近
        if position_ratio > self.bb_lower_zone:
            return False, 0.0
            
        # 布林带评分：越接近下轨评分越高
        score = max(0, 1 - position_ratio / self.bb_lower_zone)
        
        return True, score
        
    def layer6_timeframe_filter(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """第6层：多时间框架确认"""
        if not self.use_higher_timeframe or len(df) < 30:
            return True, 0.8  # 默认通过
            
        # 模拟更高时间框架（3倍周期）
        higher_tf_data = df.iloc[::self.higher_tf_multiplier].copy()
        
        if len(higher_tf_data) < 10:
            return True, 0.8
            
        # 更高时间框架的RSI
        higher_rsi = self._calculate_rsi(higher_tf_data['close'])[-1]
        
        # 更高时间框架趋势
        higher_ma20 = higher_tf_data['close'].rolling(min(20, len(higher_tf_data))).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # 多时间框架一致性检查
        timeframe_score = 0.8  # 基础分
        
        # RSI一致性
        if 30 <= higher_rsi <= 50:  # 更高时间框架也在合理范围
            timeframe_score += 0.1
            
        # 趋势一致性
        if current_price < higher_ma20:  # 更高时间框架也在均线下方
            timeframe_score += 0.1
            
        return True, min(timeframe_score, 1.0)
        
    def generate_enhanced_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """生成增强信号"""
        if len(df) < 30:
            return None
            
        self.signal_stats['total_signals'] += 1
        
        # 市场状态检查
        market_regime = self.detect_market_regime(df)
        if market_regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.STRONG_DOWNTREND]:
            logger.debug(f"{symbol}: Market regime {market_regime.value} - skipping")
            return None
            
        # 基础条件检查
        current_row = df.iloc[-1]
        if not (current_row['close'] < current_row['open'] and  # 必须下跌
                len(df) >= 20):
            return None
            
        # 6层过滤
        confidence = SignalConfidence()
        
        # Layer 1: RSI
        rsi_pass, confidence.rsi_score = self.layer1_rsi_filter(df)
        if not rsi_pass:
            return None
            
        # Layer 2: Trend
        trend_pass, confidence.trend_score = self.layer2_trend_filter(df)
        if not trend_pass:
            return None
            
        # Layer 3: Momentum  
        momentum_pass, confidence.momentum_score = self.layer3_momentum_filter(df)
        if not momentum_pass:
            return None
            
        # Layer 4: Volume
        volume_pass, confidence.volume_score = self.layer4_volume_filter(df)
        if not volume_pass:
            return None
            
        # Layer 5: Bollinger
        bb_pass, confidence.bollinger_score = self.layer5_bollinger_filter(df)
        if not bb_pass:
            return None
            
        # Layer 6: Timeframe
        tf_pass, confidence.timeframe_score = self.layer6_timeframe_filter(df)
        if not tf_pass:
            return None
            
        # 最终置信度检查
        if confidence.total_score < 0.5:  # 最低置信度阈值
            return None
            
        self.signal_stats['filtered_signals'] += 1
        self.signal_stats['grade_distribution'][confidence.grade] += 1
        
        # 生成信号
        signal = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'price': current_row['close'],
            'rsi': self._calculate_rsi(df['close'])[-1],
            'confidence': confidence.total_score,
            'grade': confidence.grade,
            'market_regime': market_regime.value,
            'signal_type': 'ENHANCED_DIP_BUY',
            'confidence_breakdown': {
                'rsi': confidence.rsi_score,
                'trend': confidence.trend_score,
                'momentum': confidence.momentum_score,
                'volume': confidence.volume_score,
                'bollinger': confidence.bollinger_score,
                'timeframe': confidence.timeframe_score
            }
        }
        
        logger.info(f"Enhanced signal: {symbol} @ {current_row['close']:.4f} "
                   f"[Grade: {confidence.grade}, Confidence: {confidence.total_score:.2f}]")
        
        return signal
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        
        return atr / df['close'].iloc[-1]  # 标准化
        
    def get_filter_stats(self) -> Dict:
        """获取过滤统计"""
        total = self.signal_stats['total_signals']
        filtered = self.signal_stats['filtered_signals']
        
        return {
            'total_signals': total,
            'filtered_signals': filtered,
            'filter_rate': (total - filtered) / total * 100 if total > 0 else 0,
            'pass_rate': filtered / total * 100 if total > 0 else 0,
            'grade_distribution': self.signal_stats['grade_distribution']
        }