#!/usr/bin/env python3
"""
Simplified DipMaster - Phase 1 of Overfitting Optimization
超简化版本，从15+参数降至3个核心参数

核心理念：
- 如果复杂版本不工作，简单版本也不会工作
- 如果简单版本工作，那就不需要复杂版本
- 3个参数：RSI阈值、止盈目标、止损水平

Author: DipMaster Optimization Team  
Date: 2025-08-15
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class SimpleTrade:
    """简化交易记录"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl_usd: float
    pnl_percent: float
    holding_minutes: float
    exit_reason: str
    win: bool


@dataclass
class SimpleSignal:
    """简化交易信号"""
    symbol: str
    timestamp: datetime
    price: float
    rsi: float
    should_buy: bool
    confidence: float  # 简单的0-1评分
    reason: str


class SimplifiedDipMaster:
    """超简化DipMaster策略 - 只有3个参数"""
    
    def __init__(self, 
                 rsi_threshold: float = 40.0,
                 take_profit_pct: float = 0.015,  # 1.5%
                 stop_loss_pct: float = 0.008):   # 0.8%
        
        # 核心参数 - ONLY 3 parameters
        self.rsi_threshold = rsi_threshold
        self.take_profit_pct = take_profit_pct 
        self.stop_loss_pct = stop_loss_pct
        
        # 风险管理（固定，不可调）
        self.max_position_size_pct = 0.05  # 5% per trade (固定)
        self.max_holding_minutes = 60      # 1 hour max (固定)
        self.commission_rate = 0.0008      # 0.08% commission (固定)
        
        # 移除的复杂功能
        # - 时间过滤 (forbidden_hours, optimal_hours)
        # - 市场状态检测 (market_regime)
        # - 多层过滤 (3-layer filtering)
        # - 成交量确认 (volume_spike)
        # - 动态阈值 (volatility_adjusted_threshold)
        # - 置信度评分 (confidence_score)
        
        logger.info(f"✅ SimplifiedDipMaster initialized:")
        logger.info(f"   RSI Threshold: {self.rsi_threshold}")
        logger.info(f"   Take Profit: {self.take_profit_pct:.1%}")
        logger.info(f"   Stop Loss: {self.stop_loss_pct:.1%}")
        logger.info(f"   Risk/Reward Ratio: {self.take_profit_pct/self.stop_loss_pct:.1f}")
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算最基础的技术指标"""
        
        # 只计算必需的RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 逢跌确认
        df['is_dip'] = df['close'] < df['open']
        
        return df
    
    def generate_signal(self, df: pd.DataFrame, current_time: datetime) -> Optional[SimpleSignal]:
        """生成超简化交易信号"""
        
        if len(df) < 15:  # 需要足够数据计算RSI
            return None
            
        df = self.calculate_indicators(df)
        current_row = df.iloc[-1]
        
        # 检查空值
        if pd.isna(current_row['rsi']):
            return None
        
        # 超简单入场条件
        rsi = current_row['rsi']
        is_dip = current_row['is_dip']
        current_price = current_row['close']
        
        should_buy = False
        confidence = 0.0
        reason = ""
        
        # CORE LOGIC: RSI低于阈值 + 逢跌
        if rsi <= self.rsi_threshold and is_dip:
            should_buy = True
            
            # 简单置信度计算 (0-1)
            rsi_confidence = max(0, (self.rsi_threshold - rsi) / self.rsi_threshold)
            confidence = min(rsi_confidence, 1.0)
            reason = f"RSI_{rsi:.1f}_DIP"
        else:
            should_buy = False
            reason = f"RSI_{rsi:.1f}_NO_DIP" if not is_dip else f"RSI_{rsi:.1f}_HIGH"
        
        return SimpleSignal(
            symbol=df.get('symbol', 'UNKNOWN')[0] if 'symbol' in df.columns else 'UNKNOWN',
            timestamp=current_time,
            price=current_price,
            rsi=rsi,
            should_buy=should_buy,
            confidence=confidence,
            reason=reason
        )
    
    def calculate_position_size(self, signal: SimpleSignal, current_capital: float) -> float:
        """计算仓位大小 - 基于固定风险百分比"""
        
        # 基于止损的仓位计算
        risk_per_unit = signal.price * self.stop_loss_pct
        max_risk_usd = current_capital * self.max_position_size_pct
        
        # 仓位大小 = 最大风险金额 / 每单位风险
        position_size = max_risk_usd / risk_per_unit if risk_per_unit > 0 else 0
        
        return max(position_size, 0)
    
    def should_exit_position(self, trade: SimpleTrade, current_price: float, 
                           current_time: datetime) -> Tuple[bool, str]:
        """检查是否应该退出仓位"""
        
        # 持仓时间
        holding_minutes = (current_time - trade.entry_time).total_seconds() / 60
        
        # PnL计算
        pnl_pct = (current_price - trade.entry_price) / trade.entry_price
        
        # 出场条件检查
        
        # 1. 止盈
        if pnl_pct >= self.take_profit_pct:
            return True, "take_profit"
        
        # 2. 止损
        if pnl_pct <= -self.stop_loss_pct:
            return True, "stop_loss"
        
        # 3. 最大持仓时间
        if holding_minutes >= self.max_holding_minutes:
            return True, "time_exit"
        
        return False, ""
    
    def backtest_strategy(self, df: pd.DataFrame, 
                         initial_capital: float = 10000) -> Dict[str, Any]:
        """回测简化策略"""
        
        logger.info(f"🚀 Starting simplified backtest with {len(df)} data points...")
        
        df = self.calculate_indicators(df)
        
        # 初始化
        trades: List[SimpleTrade] = []
        current_position: Optional[SimpleTrade] = None
        capital = initial_capital
        
        for i in range(len(df)):
            current_time = df.index[i]
            current_row = df.iloc[i]
            current_price = current_row['close']
            
            # 跳过空值
            if pd.isna(current_row['rsi']):
                continue
            
            # 检查是否需要出场
            if current_position:
                should_exit, exit_reason = self.should_exit_position(
                    current_position, current_price, current_time)
                
                if should_exit:
                    # 计算最终PnL
                    pnl_usd = (current_price - current_position.entry_price) * current_position.quantity
                    commission = abs(pnl_usd) * self.commission_rate
                    net_pnl = pnl_usd - commission
                    
                    # 更新交易记录
                    current_position.exit_time = current_time
                    current_position.exit_price = current_price
                    current_position.pnl_usd = net_pnl
                    current_position.pnl_percent = (current_price - current_position.entry_price) / current_position.entry_price * 100
                    current_position.holding_minutes = (current_time - current_position.entry_time).total_seconds() / 60
                    current_position.exit_reason = exit_reason
                    current_position.win = net_pnl > 0
                    
                    trades.append(current_position)
                    capital += net_pnl
                    current_position = None
            
            # 检查入场机会
            if not current_position:
                # 生成信号
                signal = self.generate_signal(df.iloc[max(0, i-20):i+1], current_time)
                
                if signal and signal.should_buy:
                    # 计算仓位
                    position_size = self.calculate_position_size(signal, capital)
                    
                    if position_size > 0:
                        current_position = SimpleTrade(
                            symbol=signal.symbol,
                            entry_time=current_time,
                            exit_time=current_time,  # 临时，出场时更新
                            entry_price=current_price,
                            exit_price=0,  # 出场时更新
                            quantity=position_size,
                            pnl_usd=0,
                            pnl_percent=0,
                            holding_minutes=0,
                            exit_reason="",
                            win=False
                        )
        
        # 计算绩效指标
        results = self._calculate_performance_metrics(trades, initial_capital, capital)
        
        logger.info(f"✅ Backtest complete: {len(trades)} trades, "
                   f"{results['win_rate']:.1%} win rate, "
                   f"{results['total_return_pct']:+.1f}% return")
        
        return results
    
    def _calculate_performance_metrics(self, trades: List[SimpleTrade], 
                                     initial_capital: float, 
                                     final_capital: float) -> Dict[str, Any]:
        """计算绩效指标"""
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_return_pct': 0.0,
                'final_capital': final_capital,
                'avg_holding_minutes': 0,
                'max_drawdown_pct': 0,
                'profit_factor': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0,
                'risk_reward_ratio': 0,
                'trades': trades
            }
        
        # 基础统计
        winning_trades = [t for t in trades if t.win]
        losing_trades = [t for t in trades if not t.win]
        
        win_rate = len(winning_trades) / len(trades)
        total_return_pct = (final_capital - initial_capital) / initial_capital * 100
        avg_holding = np.mean([t.holding_minutes for t in trades])
        
        # 盈亏分析
        if winning_trades:
            total_profits = sum(t.pnl_usd for t in winning_trades)
            avg_win_pct = np.mean([t.pnl_percent for t in winning_trades])
        else:
            total_profits = 0
            avg_win_pct = 0
        
        if losing_trades:
            total_losses = sum(abs(t.pnl_usd) for t in losing_trades)
            avg_loss_pct = np.mean([abs(t.pnl_percent) for t in losing_trades])
        else:
            total_losses = 0
            avg_loss_pct = 0
        
        # 盈亏比和利润因子
        profit_factor = total_profits / total_losses if total_losses > 0 else float('inf') if total_profits > 0 else 0
        risk_reward_ratio = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else float('inf') if avg_win_pct > 0 else 0
        
        # 最大回撤 (简化计算)
        cumulative_pnl = np.cumsum([t.pnl_usd for t in trades])
        running_max = np.maximum.accumulate(np.concatenate([[0], cumulative_pnl]))
        drawdowns = (running_max - np.concatenate([[0], cumulative_pnl])) / initial_capital * 100
        max_drawdown_pct = max(drawdowns) if len(drawdowns) > 0 else 0
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            'final_capital': final_capital,
            'avg_holding_minutes': avg_holding,
            'max_drawdown_pct': max_drawdown_pct,
            'profit_factor': profit_factor,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'risk_reward_ratio': risk_reward_ratio,
            'trades': trades
        }


def main():
    """测试简化策略"""
    
    logger.info("🧪 Testing Simplified DipMaster Strategy")
    print("="*80)
    
    # 创建简化策略实例
    strategy = SimplifiedDipMaster(
        rsi_threshold=40.0,      # RSI <= 40 入场
        take_profit_pct=0.015,   # 1.5% 止盈
        stop_loss_pct=0.008      # 0.8% 止损 (风险回报比 1.875)
    )
    
    print("📊 STRATEGY CONFIGURATION:")
    print(f"Parameters: 3 (vs 15+ in complex version)")
    print(f"RSI Threshold: {strategy.rsi_threshold}")
    print(f"Take Profit: {strategy.take_profit_pct:.1%}")
    print(f"Stop Loss: {strategy.stop_loss_pct:.1%}")
    print(f"Risk/Reward: {strategy.take_profit_pct/strategy.stop_loss_pct:.1f}")
    print(f"Max Position: {strategy.max_position_size_pct:.1%}")
    print(f"Max Hold Time: {strategy.max_holding_minutes} minutes")
    
    print("\n✅ Simplified DipMaster ready for testing")
    print("🎯 Next: Load data and run edge validation test")
    
    return strategy


if __name__ == "__main__":
    strategy = main()