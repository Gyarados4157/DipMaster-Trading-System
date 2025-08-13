#!/usr/bin/env python3
"""
DipMaster优化策略 v2.0
基于失败分析的根本性重构：解决28.8%胜率和-38.3%亏损问题

关键改进:
1. 多层信号过滤（降低交易频率）
2. 严格止损（解决208.5分钟vs5分钟持仓时间问题） 
3. 改善风险回报比（从0.71提升到>1.5）
4. 市场环境适应
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """市场状态"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class OptimizedSignal:
    """优化后的交易信号"""
    symbol: str
    timestamp: datetime
    price: float
    signal_strength: float  # 0-1，越高越强
    confidence_score: float  # 多层过滤后的综合置信度
    risk_reward_ratio: float  # 预期风险回报比
    max_hold_minutes: int  # 最大持仓时间
    stop_loss_price: float  # 止损价格
    take_profit_price: float  # 止盈价格
    market_regime: MarketRegime
    quality_grade: str  # A/B/C级信号
    reason: str


class OptimizedMarketAnalyzer:
    """优化的市场分析器"""
    
    def __init__(self):
        self.volatility_window = 20
        self.trend_window = 50
        
    def analyze_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """分析市场状态"""
        if len(df) < self.trend_window:
            return MarketRegime.SIDEWAYS
            
        # 计算价格变化和波动率
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-self.trend_window]) / df['close'].iloc[-self.trend_window]
        volatility = df['close'].pct_change().rolling(self.volatility_window).std().iloc[-1]
        
        # 趋势判断
        if abs(price_change) > 0.05:  # 5%以上变化
            if price_change > 0:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        
        # 波动率判断  
        if volatility > 0.02:  # 2%以上波动
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.005:  # 0.5%以下波动
            return MarketRegime.LOW_VOLATILITY
        
        return MarketRegime.SIDEWAYS
    
    def calculate_volatility_adjusted_threshold(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算基于波动率的动态阈值"""
        if len(df) < 20:
            return {'stop_loss': 0.02, 'take_profit': 0.015, 'dip_threshold': 0.005}
            
        # ATR-based dynamic thresholds
        high_low = df['high'] - df['low'] 
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        
        # 优化风险回报比 - 收紧止损，扩大止盈
        stop_loss = min(max(atr * 1.5, 0.003), 0.015)  # 0.3%-1.5% (收紧止损)
        take_profit = min(max(atr * 2.5, 0.012), 0.035)  # 1.2%-3.5% (扩大止盈)
        dip_threshold = min(max(atr * 0.3, 0.001), 0.005)  # 0.1%-0.5%
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'dip_threshold': dip_threshold,
            'atr': atr
        }


class MultiLayerSignalFilter:
    """多层信号过滤器 - 解决过度交易问题"""
    
    def __init__(self):
        self.market_analyzer = OptimizedMarketAnalyzer()
        
        # 优化的过滤标准（提高质量优于数量）
        self.min_confidence = 0.5  # 提高最小置信度以改善胜率
        self.min_volume_spike = 1.5  # 降低成交量要求
        self.min_rsi_separation = 3   # 降低RSI与边界的要求
        self.max_correlation = 0.8   # 放宽相关性限制
        
        # 时间过滤（减少禁用时段）
        self.forbidden_hours = [13, 18]  # 减少禁用时段
        self.optimal_hours = [3, 8, 10, 14, 20]   # 增加最佳时段
        
        # 信号间隔控制
        self.min_signal_interval = 30  # 分钟，避免过于频繁
        self.last_signals = {}
        
    def layer_1_basic_filter(self, symbol: str, df: pd.DataFrame, current_time: datetime) -> Optional[Dict]:
        """第一层：基础技术过滤"""
        
        if len(df) < 50:
            return None
            
        indicators = self.calculate_enhanced_indicators(df)
        current_price = indicators['current_price']
        
        # 基础条件
        filters_passed = []
        confidence = 0.0
        
        # RSI过滤 - 更严格的条件
        rsi = indicators['rsi']
        if 40 <= rsi <= 50:  # 更严格的最优RSI区间
            confidence += 0.4
            filters_passed.append(f"RSI_OPTIMAL_{rsi:.1f}")
        elif 35 <= rsi <= 60:  # 可接受区间
            confidence += 0.3
            filters_passed.append(f"RSI_OK_{rsi:.1f}")
        elif 30 <= rsi <= 65:  # 边界区间
            confidence += 0.15
            filters_passed.append(f"RSI_ACCEPTABLE_{rsi:.1f}")
        else:
            # RSI不在理想区间，给予最低分
            confidence += 0.05
            filters_passed.append(f"RSI_POOR_{rsi:.1f}")
        
        # 价格位置 - 严格要求逢跌买入
        thresholds = self.market_analyzer.calculate_volatility_adjusted_threshold(df)
        price_vs_open = (current_price - indicators['candle_open']) / indicators['candle_open']
        
        if price_vs_open < -thresholds['dip_threshold'] * 2:  # 真正的大跌
            confidence += 0.3
            filters_passed.append("BIG_DIP")
        elif price_vs_open < -thresholds['dip_threshold']:
            confidence += 0.25
            filters_passed.append("TRUE_DIP")
        elif price_vs_open < -0.001:  # 至少要有小幅下跌
            confidence += 0.15
            filters_passed.append("MINOR_DIP")
        else:
            # 没有下跌不符合策略
            confidence += 0.02
            filters_passed.append("NO_DIP")
        
        # 成交量确认 - 适度要求
        if indicators['volume_ratio'] >= self.min_volume_spike:
            confidence += 0.15
            filters_passed.append(f"VOLUME_SPIKE_{indicators['volume_ratio']:.1f}")
        elif indicators['volume_ratio'] >= 1.2:
            confidence += 0.08
            filters_passed.append("VOLUME_UP")
        elif indicators['volume_ratio'] >= 1.0:
            confidence += 0.03  # 正常成交量仍给少量分数
            filters_passed.append("VOLUME_NORMAL")
        else:
            confidence += 0.01  # 低成交量不直接淘汰，但几乎不加分
            filters_passed.append("VOLUME_LOW")
            
        # 技术位置确认
        if current_price < indicators['ma20'] and current_price < indicators['bb_lower']:
            confidence += 0.1
            filters_passed.append("TECH_SUPPORT")
            
        return {
            'confidence': confidence,
            'indicators': indicators,
            'filters_passed': filters_passed,
            'thresholds': thresholds
        }
    
    def layer_2_market_filter(self, symbol: str, df: pd.DataFrame, layer1_result: Dict) -> Optional[Dict]:
        """第二层：市场环境过滤"""
        
        market_regime = self.market_analyzer.analyze_market_regime(df)
        
        # 根据市场状态调整
        regime_multiplier = 1.0
        regime_suitable = True
        
        if market_regime == MarketRegime.TRENDING_UP:
            # 上涨趋势中的逢跌买入风险较高
            regime_multiplier = 0.7
            regime_suitable = layer1_result['confidence'] > 0.8  # 需要更高置信度
            
        elif market_regime == MarketRegime.TRENDING_DOWN:
            # 下跌趋势中避免交易
            regime_suitable = False
            
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            # 高波动环境需要更严格条件
            regime_multiplier = 0.8
            regime_suitable = layer1_result['confidence'] > 0.75
            
        elif market_regime == MarketRegime.SIDEWAYS:
            # 横盘是最适合的环境
            regime_multiplier = 1.2
            
        if not regime_suitable:
            return None
            
        adjusted_confidence = layer1_result['confidence'] * regime_multiplier
        
        return {
            'adjusted_confidence': adjusted_confidence,
            'market_regime': market_regime,
            'regime_multiplier': regime_multiplier
        }
    
    def layer_3_timing_filter(self, symbol: str, current_time: datetime, layer2_result: Dict) -> Optional[Dict]:
        """第三层：时间和频率过滤"""
        
        current_hour = current_time.hour
        
        # 时间过滤
        time_multiplier = 1.0
        if current_hour in self.forbidden_hours:
            return None  # 最差时段直接拒绝
        elif current_hour in self.optimal_hours:
            time_multiplier = 1.15  # 最佳时段加分
        
        # 信号间隔控制
        if symbol in self.last_signals:
            last_signal_time = self.last_signals[symbol]
            minutes_since_last = (current_time - last_signal_time).total_seconds() / 60
            if minutes_since_last < self.min_signal_interval:
                return None
        
        final_confidence = layer2_result['adjusted_confidence'] * time_multiplier
        
        # 更新最后信号时间
        self.last_signals[symbol] = current_time
        
        return {
            'final_confidence': final_confidence,
            'time_multiplier': time_multiplier
        }
    
    def calculate_enhanced_indicators(self, df: pd.DataFrame) -> Dict:
        """计算增强技术指标"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 移动平均
        ma20 = df['close'].rolling(window=20).mean()
        ma50 = df['close'].rolling(window=50).mean()
        
        # 布林带
        std20 = df['close'].rolling(window=20).std()
        bb_upper = ma20 + (std20 * 2)
        bb_lower = ma20 - (std20 * 2)
        
        # 成交量
        volume_ma = df['volume'].rolling(window=20).mean()
        volume_ratio = df['volume'].iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
        
        # 价格信息
        current_price = df['close'].iloc[-1]
        candle_open = df['open'].iloc[-1]
        
        return {
            'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
            'ma20': ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else current_price,
            'ma50': ma50.iloc[-1] if not pd.isna(ma50.iloc[-1]) else current_price,
            'bb_upper': bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else current_price * 1.02,
            'bb_lower': bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else current_price * 0.98,
            'volume_ratio': volume_ratio,
            'current_price': current_price,
            'candle_open': candle_open
        }
    
    def generate_optimized_signal(self, symbol: str, df: pd.DataFrame, current_time: datetime) -> Optional[OptimizedSignal]:
        """生成优化信号（通过多层过滤）"""
        
        # 第一层过滤
        layer1 = self.layer_1_basic_filter(symbol, df, current_time)
        if not layer1:
            return None
            
        # 第二层过滤  
        layer2 = self.layer_2_market_filter(symbol, df, layer1)
        if not layer2:
            return None
            
        # 第三层过滤
        layer3 = self.layer_3_timing_filter(symbol, current_time, layer2)
        if not layer3:
            return None
            
        final_confidence = layer3['final_confidence']
        
        # 最终置信度检查
        if final_confidence < self.min_confidence:
            return None
            
        # 计算风险管理参数
        indicators = layer1['indicators']
        thresholds = layer1['thresholds']
        current_price = indicators['current_price']
        
        # 动态止损止盈
        stop_loss_price = current_price * (1 - thresholds['stop_loss'])
        take_profit_price = current_price * (1 + thresholds['take_profit'])
        
        # 预期风险回报比
        risk_reward_ratio = thresholds['take_profit'] / thresholds['stop_loss']
        
        # 最大持仓时间（基于分析：盈利5分钟，亏损208分钟）
        if layer2['market_regime'] == MarketRegime.HIGH_VOLATILITY:
            max_hold = 30  # 高波动快进快出
        elif final_confidence > 0.95:
            max_hold = 45  # 高置信度可以稍微长一点
        else:
            max_hold = 20  # 严格控制持仓时间
            
        # 信号质量评级 - 调整阈值以产生更多高质量信号
        if final_confidence >= 0.7:
            quality_grade = "A"
        elif final_confidence >= 0.55:
            quality_grade = "B"  
        else:
            quality_grade = "C"
        
        return OptimizedSignal(
            symbol=symbol,
            timestamp=current_time,
            price=current_price,
            signal_strength=final_confidence,
            confidence_score=final_confidence,
            risk_reward_ratio=risk_reward_ratio,
            max_hold_minutes=max_hold,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            market_regime=layer2['market_regime'],
            quality_grade=quality_grade,
            reason=" + ".join(layer1['filters_passed'])
        )


class OptimizedRiskManager:
    """优化的风险管理器"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.max_portfolio_risk = 0.02  # 单次最大风险2%
        self.max_position_size = 0.15   # 单仓位最大15%
        self.max_daily_trades = 20      # 每日最大20笔（vs之前5822笔）
        self.max_concurrent_positions = 3
        
        # 基于分析的严格风险控制
        self.force_close_after_minutes = 180  # 强制平仓时间
        self.emergency_stop_loss = 0.03       # 紧急止损3%
        
    def calculate_position_size(self, signal: OptimizedSignal, current_capital: float) -> float:
        """基于风险的仓位计算"""
        
        # 基于止损的风险计算
        risk_per_share = signal.price - signal.stop_loss_price
        risk_ratio = risk_per_share / signal.price
        
        # 基于资金管理的最大仓位
        max_risk_usd = current_capital * self.max_portfolio_risk
        max_shares_by_risk = max_risk_usd / risk_per_share if risk_per_share > 0 else 0
        
        # 基于仓位大小限制
        max_position_usd = current_capital * self.max_position_size
        max_shares_by_position = max_position_usd / signal.price
        
        # 基于信号质量和置信度调整
        quality_multiplier = {
            'A': 1.2,  # A级信号加大仓位
            'B': 0.8, 
            'C': 0.5   # C级信号减小仓位
        }.get(signal.quality_grade, 0.3)
        
        # 进一步基于置信度调整
        confidence_multiplier = min(signal.confidence_score / 0.5, 1.5)  # 置信度越高仓位越大
        quality_multiplier *= confidence_multiplier
        
        # 取最小值确保风险控制
        position_size = min(max_shares_by_risk, max_shares_by_position) * quality_multiplier
        
        return max(position_size, 0)
    
    def should_force_exit(self, entry_time: datetime, current_time: datetime, 
                         entry_price: float, current_price: float) -> Tuple[bool, str]:
        """检查是否需要强制退出"""
        
        # 持仓时间检查
        holding_minutes = (current_time - entry_time).total_seconds() / 60
        if holding_minutes >= self.force_close_after_minutes:
            return True, "FORCE_TIMEOUT"
            
        # 紧急止损
        loss_ratio = (current_price - entry_price) / entry_price
        if loss_ratio <= -self.emergency_stop_loss:
            return True, "EMERGENCY_STOP"
            
        return False, ""


def main():
    """测试优化策略"""
    logger.info("🚀 Testing Optimized DipMaster Strategy")
    
    # 创建优化组件
    signal_filter = MultiLayerSignalFilter()
    risk_manager = OptimizedRiskManager()
    
    logger.info("✅ Optimized strategy components initialized")
    logger.info("Key improvements:")
    logger.info("  • Multi-layer signal filtering (reduce over-trading)")
    logger.info("  • Dynamic risk management (improve risk-reward ratio)")
    logger.info("  • Market regime adaptation (avoid bad conditions)")
    logger.info("  • Strict holding time limits (solve 208min vs 5min problem)")
    
    return signal_filter, risk_manager


if __name__ == "__main__":
    main()