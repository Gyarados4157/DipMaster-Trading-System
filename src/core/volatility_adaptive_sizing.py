"""
Volatility Adaptive Position Sizing System - Phase 3 Optimization
自适应波动率仓位管理：根据市场波动调整仓位大小

核心理念：
- 高波动时减小仓位，降低风险
- 低波动时增大仓位，提高收益
- 基于Kelly公式的最优仓位计算
- 考虑近期表现和回撤调整
- 动态风险预算分配

目标：在控制风险的前提下最大化长期收益
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    """波动率状态"""
    VERY_LOW = "very_low"      # 极低波动 (0-20%)
    LOW = "low"                # 低波动 (20-40%)
    NORMAL = "normal"          # 正常波动 (40-60%)
    HIGH = "high"              # 高波动 (60-80%)
    VERY_HIGH = "very_high"    # 极高波动 (80-100%)


@dataclass
class PositionSizeResult:
    """仓位大小计算结果"""
    base_size_usd: float
    adjusted_size_usd: float
    leverage: float
    effective_position: float
    volatility_multiplier: float
    kelly_fraction: float
    risk_budget_used: float
    max_risk_per_trade: float
    volatility_regime: VolatilityRegime
    confidence_score: float
    
    @property
    def size_adjustment_reason(self) -> str:
        """仓位调整原因"""
        reasons = []
        if self.volatility_multiplier < 0.8:
            reasons.append("高波动率降低")
        elif self.volatility_multiplier > 1.2:
            reasons.append("低波动率增加")
        
        if self.kelly_fraction < 0.5:
            reasons.append("Kelly建议减仓")
        elif self.kelly_fraction > 0.8:
            reasons.append("Kelly建议加仓")
            
        return "; ".join(reasons) if reasons else "正常仓位"


class VolatilityAdaptiveSizing:
    """波动率自适应仓位管理器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # === 基础参数 ===
        self.base_capital = self.config.get('base_capital', 10000)  # 基础资金
        self.base_position_size = self.config.get('base_position_size', 1000)  # 基础仓位
        self.default_leverage = self.config.get('default_leverage', 10)  # 默认杠杆
        
        # === 波动率参数 ===
        self.volatility_window = 20  # 波动率计算窗口
        self.volatility_lookback_days = 90  # 波动率历史对比天数
        
        # === Kelly公式参数 ===
        self.use_kelly_criterion = True
        self.kelly_lookback_trades = 50  # Kelly计算的交易历史
        self.max_kelly_fraction = 0.25  # 最大Kelly分数
        self.min_kelly_fraction = 0.05  # 最小Kelly分数
        
        # === 风险管理参数 ===
        self.max_portfolio_risk = 0.02  # 组合最大风险2%
        self.max_single_trade_risk = 0.005  # 单笔最大风险0.5%
        self.drawdown_adjustment_threshold = 0.05  # 回撤调整阈值5%
        
        # === 波动率阈值 ===
        self.volatility_thresholds = {
            'very_low': 0.02,    # 2%以下
            'low': 0.04,         # 2-4%
            'normal': 0.08,      # 4-8%
            'high': 0.15,        # 8-15%
            'very_high': 0.15    # 15%以上
        }
        
        # === 仓位调整系数 ===
        self.volatility_multipliers = {
            VolatilityRegime.VERY_LOW: 1.5,   # 极低波动增加50%
            VolatilityRegime.LOW: 1.2,        # 低波动增加20%
            VolatilityRegime.NORMAL: 1.0,     # 正常波动不变
            VolatilityRegime.HIGH: 0.7,       # 高波动减少30%
            VolatilityRegime.VERY_HIGH: 0.4   # 极高波动减少60%
        }
        
        # === 杠杆调整 ===
        self.leverage_adjustments = {
            VolatilityRegime.VERY_LOW: 1.2,   # 低波动可以稍微加杠杆
            VolatilityRegime.LOW: 1.1,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.HIGH: 0.8,       # 高波动降低杠杆
            VolatilityRegime.VERY_HIGH: 0.5
        }
        
        # 交易历史记录
        self.trade_history: List[Dict] = []
        self.current_drawdown = 0.0
        self.peak_capital = self.base_capital
        
        # 实时风险监控
        self.current_positions = {}
        self.allocated_risk_budget = 0.0
        
    def calculate_atr_volatility(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ATR波动率"""
        if len(df) < period + 1:
            return 0.05  # 默认5%波动率
            
        # 计算真实范围
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # ATR
        atr = true_range.rolling(period).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        return atr / current_price if current_price > 0 else 0.05
        
    def calculate_realized_volatility(self, df: pd.DataFrame, window: int = 20) -> float:
        """计算已实现波动率"""
        if len(df) < window:
            return 0.05
            
        returns = df['close'].pct_change().dropna()
        if len(returns) < window:
            return 0.05
            
        # 年化波动率（假设1分钟数据）
        volatility = returns.rolling(window).std().iloc[-1]
        annualized_vol = volatility * np.sqrt(365 * 24 * 60)  # 年化
        
        # 转换为日波动率
        daily_vol = annualized_vol / np.sqrt(365)
        
        return max(min(daily_vol, 0.5), 0.01)  # 限制在1%-50%之间
        
    def classify_volatility_regime(self, current_volatility: float, 
                                 historical_data: pd.DataFrame = None) -> VolatilityRegime:
        """分类波动率状态"""
        
        # 如果有历史数据，计算相对位置
        if historical_data is not None and len(historical_data) > 50:
            hist_vol = []
            for i in range(len(historical_data) - 20):
                chunk = historical_data.iloc[i:i+20]
                vol = self.calculate_realized_volatility(chunk, window=10)
                hist_vol.append(vol)
                
            if hist_vol:
                percentile = np.percentile(hist_vol, 
                                         [20, 40, 60, 80])
                
                if current_volatility <= percentile[0]:
                    return VolatilityRegime.VERY_LOW
                elif current_volatility <= percentile[1]:
                    return VolatilityRegime.LOW
                elif current_volatility <= percentile[2]:
                    return VolatilityRegime.NORMAL
                elif current_volatility <= percentile[3]:
                    return VolatilityRegime.HIGH
                else:
                    return VolatilityRegime.VERY_HIGH
        
        # 绝对阈值分类
        if current_volatility <= self.volatility_thresholds['very_low']:
            return VolatilityRegime.VERY_LOW
        elif current_volatility <= self.volatility_thresholds['low']:
            return VolatilityRegime.LOW
        elif current_volatility <= self.volatility_thresholds['normal']:
            return VolatilityRegime.NORMAL
        elif current_volatility <= self.volatility_thresholds['high']:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.VERY_HIGH
            
    def calculate_kelly_fraction(self, win_rate: float = None, 
                               avg_win: float = None, 
                               avg_loss: float = None) -> float:
        """计算Kelly分数"""
        
        # 如果有交易历史，使用实际数据
        if len(self.trade_history) >= 10:
            recent_trades = self.trade_history[-self.kelly_lookback_trades:]
            
            wins = [t['pnl_percent'] for t in recent_trades if t['pnl_percent'] > 0]
            losses = [t['pnl_percent'] for t in recent_trades if t['pnl_percent'] < 0]
            
            if wins and losses:
                win_rate = len(wins) / len(recent_trades)
                avg_win = np.mean(wins) / 100  # 转换为小数
                avg_loss = abs(np.mean(losses)) / 100  # 转换为正数
        
        # 使用默认估值
        if win_rate is None:
            win_rate = 0.75  # 75%胜率估计
        if avg_win is None:
            avg_win = 0.012   # 1.2%平均盈利
        if avg_loss is None:
            avg_loss = 0.008  # 0.8%平均亏损
            
        # Kelly公式: f = (bp - q) / b
        # b = 赔率 = 平均盈利/平均亏损
        # p = 胜率
        # q = 败率 = 1 - p
        if avg_loss > 0:
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
        else:
            kelly_fraction = 0.1
            
        # 限制Kelly分数范围
        kelly_fraction = max(min(kelly_fraction, self.max_kelly_fraction), 
                           self.min_kelly_fraction)
        
        return kelly_fraction
        
    def calculate_drawdown_adjustment(self) -> float:
        """计算回撤调整系数"""
        if self.current_drawdown <= self.drawdown_adjustment_threshold:
            return 1.0  # 无调整
            
        # 回撤越大，仓位越小
        # 5%回撤时仓位不变，10%回撤时仓位减半
        max_drawdown_for_calc = 0.20  # 20%回撤时仓位降至0.2倍
        
        if self.current_drawdown >= max_drawdown_for_calc:
            return 0.2
            
        # 线性调整
        adjustment = 1.0 - (self.current_drawdown - self.drawdown_adjustment_threshold) * 1.6
        return max(adjustment, 0.2)
        
    def calculate_position_size(self, symbol: str, df: pd.DataFrame,
                              signal_confidence: float = 0.8,
                              current_price: float = None) -> PositionSizeResult:
        """计算最优仓位大小"""
        
        if current_price is None:
            current_price = df['close'].iloc[-1]
            
        # === 1. 波动率计算 ===
        atr_volatility = self.calculate_atr_volatility(df)
        realized_volatility = self.calculate_realized_volatility(df)
        
        # 综合波动率（ATR权重60%，已实现波动率40%）
        combined_volatility = atr_volatility * 0.6 + realized_volatility * 0.4
        
        # === 2. 波动率状态分类 ===
        volatility_regime = self.classify_volatility_regime(combined_volatility, df)
        
        # === 3. Kelly分数计算 ===
        kelly_fraction = self.calculate_kelly_fraction()
        
        # === 4. 信号置信度调整 ===
        confidence_adjustment = signal_confidence ** 0.5  # 开平方，减弱影响
        
        # === 5. 回撤调整 ===
        drawdown_adjustment = self.calculate_drawdown_adjustment()
        
        # === 6. 波动率仓位调整 ===
        volatility_multiplier = self.volatility_multipliers[volatility_regime]
        
        # === 7. 计算基础仓位 ===
        # Kelly建议的资本分配
        kelly_suggested_size = self.base_capital * kelly_fraction
        
        # 应用各种调整
        adjusted_size = self.base_position_size * volatility_multiplier * confidence_adjustment * drawdown_adjustment
        
        # 取Kelly建议和调整后仓位的较小值
        final_size_usd = min(kelly_suggested_size, adjusted_size)
        
        # === 8. 风险限制检查 ===
        # 单笔交易最大风险
        max_risk_usd = self.base_capital * self.max_single_trade_risk
        stop_loss_distance = combined_volatility * 2  # 2倍ATR作为止损距离
        
        if stop_loss_distance > 0:
            max_size_by_risk = max_risk_usd / stop_loss_distance
            final_size_usd = min(final_size_usd, max_size_by_risk)
            
        # === 9. 组合风险检查 ===
        # 检查当前已分配的风险预算
        remaining_risk_budget = self.max_portfolio_risk - (self.allocated_risk_budget / self.base_capital)
        if remaining_risk_budget <= 0:
            final_size_usd = min(final_size_usd, 100)  # 最小仓位
            
        # === 10. 杠杆调整 ===
        leverage_adjustment = self.leverage_adjustments[volatility_regime]
        adjusted_leverage = self.default_leverage * leverage_adjustment
        adjusted_leverage = max(min(adjusted_leverage, 20), 1)  # 限制杠杆1-20倍
        
        # === 11. 最终仓位计算 ===
        effective_position = final_size_usd * adjusted_leverage
        
        # 风险预算使用
        risk_budget_used = (final_size_usd * stop_loss_distance) / self.base_capital
        
        result = PositionSizeResult(
            base_size_usd=self.base_position_size,
            adjusted_size_usd=final_size_usd,
            leverage=adjusted_leverage,
            effective_position=effective_position,
            volatility_multiplier=volatility_multiplier,
            kelly_fraction=kelly_fraction,
            risk_budget_used=risk_budget_used,
            max_risk_per_trade=max_risk_usd,
            volatility_regime=volatility_regime,
            confidence_score=signal_confidence
        )
        
        logger.info(f"{symbol}: 仓位计算完成 - "
                   f"基础: ${self.base_position_size:.0f} -> 调整: ${final_size_usd:.0f}, "
                   f"杠杆: {adjusted_leverage:.1f}x, 波动率: {volatility_regime.value}")
        
        return result
        
    def update_trade_result(self, trade_result: Dict):
        """更新交易结果"""
        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': trade_result.get('symbol'),
            'pnl_percent': trade_result.get('pnl_percent', 0),
            'pnl_usd': trade_result.get('pnl_usd', 0),
            'holding_minutes': trade_result.get('holding_minutes', 0),
            'exit_reason': trade_result.get('exit_reason', 'unknown')
        })
        
        # 更新资本和回撤
        pnl_usd = trade_result.get('pnl_usd', 0)
        new_capital = self.base_capital + pnl_usd
        
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_capital - new_capital) / self.peak_capital
            
        self.base_capital = new_capital
        
        # 限制历史记录长度
        if len(self.trade_history) > 200:
            self.trade_history = self.trade_history[-150:]
            
    def allocate_position_risk(self, symbol: str, position_size: float, 
                              stop_loss_distance: float):
        """分配仓位风险预算"""
        risk_amount = position_size * stop_loss_distance
        self.allocated_risk_budget += risk_amount
        self.current_positions[symbol] = {
            'size': position_size,
            'risk': risk_amount,
            'timestamp': datetime.now()
        }
        
    def release_position_risk(self, symbol: str):
        """释放仓位风险预算"""
        if symbol in self.current_positions:
            self.allocated_risk_budget -= self.current_positions[symbol]['risk']
            self.allocated_risk_budget = max(0, self.allocated_risk_budget)
            del self.current_positions[symbol]
            
    def get_sizing_metrics(self) -> Dict:
        """获取仓位管理指标"""
        recent_trades = self.trade_history[-50:] if len(self.trade_history) >= 50 else self.trade_history
        
        if not recent_trades:
            return {'message': 'No trades yet'}
            
        wins = [t for t in recent_trades if t['pnl_percent'] > 0]
        losses = [t for t in recent_trades if t['pnl_percent'] <= 0]
        
        return {
            'total_trades': len(recent_trades),
            'win_rate': len(wins) / len(recent_trades) * 100,
            'avg_win_percent': np.mean([t['pnl_percent'] for t in wins]) if wins else 0,
            'avg_loss_percent': np.mean([t['pnl_percent'] for t in losses]) if losses else 0,
            'current_capital': self.base_capital,
            'peak_capital': self.peak_capital,
            'current_drawdown_percent': self.current_drawdown * 100,
            'allocated_risk_budget': self.allocated_risk_budget,
            'remaining_risk_capacity': (self.max_portfolio_risk * self.base_capital) - self.allocated_risk_budget,
            'active_positions': len(self.current_positions),
            'kelly_fraction': self.calculate_kelly_fraction()
        }
        
    def get_volatility_analysis(self, df: pd.DataFrame) -> Dict:
        """获取波动率分析"""
        atr_vol = self.calculate_atr_volatility(df)
        realized_vol = self.calculate_realized_volatility(df)
        combined_vol = atr_vol * 0.6 + realized_vol * 0.4
        regime = self.classify_volatility_regime(combined_vol, df)
        
        return {
            'atr_volatility': atr_vol * 100,
            'realized_volatility': realized_vol * 100,
            'combined_volatility': combined_vol * 100,
            'volatility_regime': regime.value,
            'position_multiplier': self.volatility_multipliers[regime],
            'leverage_multiplier': self.leverage_adjustments[regime],
            'recommended_action': self._get_volatility_recommendation(regime)
        }
        
    def _get_volatility_recommendation(self, regime: VolatilityRegime) -> str:
        """获取波动率建议"""
        recommendations = {
            VolatilityRegime.VERY_LOW: "低波动期，可适当增加仓位和杠杆",
            VolatilityRegime.LOW: "波动率较低，正常交易",
            VolatilityRegime.NORMAL: "波动率正常，标准仓位管理",
            VolatilityRegime.HIGH: "波动率偏高，减小仓位，降低杠杆",
            VolatilityRegime.VERY_HIGH: "高波动期，大幅减仓，谨慎交易"
        }
        return recommendations[regime]