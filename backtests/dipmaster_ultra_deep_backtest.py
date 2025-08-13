#!/usr/bin/env python3
"""
DipMaster Ultra Deep Backtest Implementation
超深度回测实施 - 全面策略验证与优化

基于6阶段回测计划的完整实现：
Phase 1: 策略验证
Phase 2: 参数优化  
Phase 3: 多币种分析
Phase 4: 市场环境分析
Phase 5: 风险分析
Phase 6: 策略增强

Author: DipMaster Analysis Team
Date: 2025-08-13
Version: 1.0.0
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import itertools
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradeResult(Enum):
    """交易结果类型"""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"

@dataclass
class TradeRecord:
    """交易记录"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl_usd: float
    pnl_percent: float
    holding_minutes: float
    entry_rsi: float
    exit_rsi: float
    exit_reason: str
    commission: float
    slippage: float
    result: TradeResult

@dataclass
class BacktestMetrics:
    """回测指标"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_holding_minutes: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int

class UltraDeepBacktest:
    """DipMaster超深度回测引擎"""
    
    def __init__(self, config: Dict = None):
        """初始化回测引擎"""
        self.config = config or {}
        
        # 基础配置
        self.initial_capital = self.config.get('initial_capital', 10000)
        self.commission_rate = self.config.get('commission_rate', 0.0004)
        self.slippage_bps = self.config.get('slippage_bps', 2.0)
        
        # DipMaster策略参数 (可优化)
        self.rsi_range = self.config.get('rsi_range', (30, 50))
        self.ma_period = self.config.get('ma_period', 20)
        self.min_holding_minutes = self.config.get('min_holding_minutes', 15)
        self.max_holding_minutes = self.config.get('max_holding_minutes', 180)
        self.target_avg_minutes = self.config.get('target_avg_minutes', 72.65)
        self.profit_target = self.config.get('profit_target', 0.008)  # 0.8%
        self.stop_loss = self.config.get('stop_loss', None)  # None = 无止损
        self.leverage = self.config.get('leverage', 10)
        self.base_position_size = self.config.get('base_position_size', 1000)
        
        # 出场边界
        self.exit_boundaries = self.config.get('exit_boundaries', [15, 30, 45, 60])
        self.boundary_exit_probability = self.config.get('boundary_exit_probability', 0.7)
        
        # 交易记录
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        logger.info("🚀 DipMaster Ultra Deep Backtest Engine initialized")
        logger.info(f"📊 配置: RSI{self.rsi_range}, MA{self.ma_period}, "
                   f"持仓{self.min_holding_minutes}-{self.max_holding_minutes}分钟")
        
    def load_market_data(self, symbol: str) -> pd.DataFrame:
        """加载市场数据"""
        data_file = f"data/market_data/{symbol}_5m_2years.csv"
        
        if not Path(data_file).exists():
            logger.error(f"❌ 数据文件不存在: {data_file}")
            return pd.DataFrame()
            
        logger.info(f"📊 加载数据: {symbol}")
        
        try:
            df = pd.read_csv(data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # 计算技术指标
            df = self.calculate_indicators(df)
            
            logger.info(f"✅ {symbol} 数据加载完成: {len(df)}条, "
                       f"时间范围: {df.index[0].strftime('%Y-%m-%d')} 到 {df.index[-1].strftime('%Y-%m-%d')}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败 {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # 移动平均线
        df[f'ma{self.ma_period}'] = df['close'].rolling(self.ma_period).mean()
        
        # 布林带
        std = df['close'].rolling(self.ma_period).std()
        df['bb_upper'] = df[f'ma{self.ma_period}'] + (std * 2)
        df['bb_lower'] = df[f'ma{self.ma_period}'] - (std * 2)
        
        # ATR (平均真实范围)
        df['atr'] = self.calculate_atr(df)
        
        # 成交量指标
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 价格变化
        df['price_change'] = (df['close'] - df['open']) / df['open']
        df['is_dip'] = df['close'] < df['open']  # 逢跌
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        return atr
    
    def check_entry_signal(self, row: pd.Series) -> bool:
        """检查入场信号 - DipMaster核心逻辑"""
        
        # 跳过空值
        if pd.isna(row['rsi']) or pd.isna(row[f'ma{self.ma_period}']):
            return False
        
        # 1. RSI范围检查 (30-50，避免极端超卖)
        if not (self.rsi_range[0] <= row['rsi'] <= self.rsi_range[1]):
            return False
        
        # 2. 逢跌买入检查 (价格低于开盘价)
        if not row['is_dip']:
            return False
        
        # 3. MA位置检查 (价格低于MA20)
        if row['close'] >= row[f'ma{self.ma_period}']:
            return False
        
        # 4. 成交量确认 (可选)
        if row['volume_ratio'] < 0.8:  # 成交量不能过低
            return False
        
        return True
    
    def check_exit_signal(self, entry_time: datetime, current_time: datetime, 
                         entry_price: float, current_price: float, 
                         current_rsi: float) -> Tuple[bool, str]:
        """检查出场信号"""
        
        holding_minutes = (current_time - entry_time).total_seconds() / 60
        pnl_percent = ((current_price - entry_price) / entry_price) * self.leverage
        
        # 1. 最小持仓时间检查
        if holding_minutes < self.min_holding_minutes:
            return False, "min_holding"
        
        # 2. 止损检查
        if self.stop_loss and pnl_percent <= -self.stop_loss:
            return True, "stop_loss"
        
        # 3. 盈利目标检查
        if pnl_percent >= self.profit_target:
            return True, "profit_target"
        
        # 4. 最大持仓时间检查
        if holding_minutes >= self.max_holding_minutes:
            return True, "max_holding"
        
        # 5. 边界时间检查
        current_minute = current_time.minute
        
        # 检查是否接近边界
        for boundary in self.exit_boundaries:
            if abs(current_minute - boundary) <= 1 or (boundary == 60 and current_minute <= 1):
                # 基于持仓时间和盈利情况决定出场概率
                time_factor = min(holding_minutes / self.target_avg_minutes, 2.0)
                profit_factor = 1.2 if pnl_percent > 0 else 0.8
                
                exit_prob = self.boundary_exit_probability * time_factor * profit_factor
                
                # 简化为确定性规则
                if (holding_minutes >= self.target_avg_minutes or 
                    pnl_percent > 0.002 or 
                    np.random.random() < exit_prob):
                    return True, f"boundary_{boundary}"
        
        return False, "holding"
    
    def calculate_position_size(self, price: float) -> float:
        """计算仓位大小"""
        return self.base_position_size / price
    
    def calculate_commission_slippage(self, price: float, quantity: float) -> Tuple[float, float]:
        """计算手续费和滑点"""
        value = price * quantity
        commission = value * self.commission_rate
        slippage = value * (self.slippage_bps / 10000)
        return commission, slippage
    
    def run_single_symbol_backtest(self, symbol: str, 
                                  start_date: str = None, 
                                  end_date: str = None) -> BacktestMetrics:
        """单币种回测"""
        
        logger.info(f"🔄 开始回测 {symbol}")
        
        # 加载数据
        df = self.load_market_data(symbol)
        if df.empty:
            return None
        
        # 时间范围过滤
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # 初始化变量
        current_position = None
        capital = self.initial_capital
        trades = []
        equity_curve = [(df.index[0], capital)]
        
        # 遍历数据
        for i in range(len(df)):
            current_row = df.iloc[i]
            current_time = df.index[i]
            
            # 检查出场信号
            if current_position:
                should_exit, exit_reason = self.check_exit_signal(
                    current_position['entry_time'],
                    current_time,
                    current_position['entry_price'],
                    current_row['close'],
                    current_row['rsi']
                )
                
                if should_exit:
                    # 平仓
                    exit_price = current_row['close']
                    quantity = current_position['quantity']
                    
                    # 计算盈亏
                    pnl_usd = (exit_price - current_position['entry_price']) * quantity * self.leverage
                    pnl_percent = ((exit_price - current_position['entry_price']) / current_position['entry_price']) * 100 * self.leverage
                    
                    # 计算费用
                    commission, slippage = self.calculate_commission_slippage(exit_price, quantity)
                    net_pnl = pnl_usd - commission - slippage
                    
                    # 更新资金
                    capital += net_pnl
                    
                    # 记录交易
                    holding_minutes = (current_time - current_position['entry_time']).total_seconds() / 60
                    
                    trade = TradeRecord(
                        symbol=symbol,
                        entry_time=current_position['entry_time'],
                        exit_time=current_time,
                        entry_price=current_position['entry_price'],
                        exit_price=exit_price,
                        quantity=quantity,
                        pnl_usd=net_pnl,
                        pnl_percent=pnl_percent,
                        holding_minutes=holding_minutes,
                        entry_rsi=current_position['entry_rsi'],
                        exit_rsi=current_row['rsi'],
                        exit_reason=exit_reason,
                        commission=commission,
                        slippage=slippage,
                        result=TradeResult.WIN if net_pnl > 0 else (TradeResult.LOSS if net_pnl < 0 else TradeResult.BREAKEVEN)
                    )
                    
                    trades.append(trade)
                    equity_curve.append((current_time, capital))
                    current_position = None
                    
                    if len(trades) % 100 == 0:
                        logger.info(f"📈 {symbol} 已完成 {len(trades)} 笔交易, 当前资金: ${capital:.2f}")
            
            # 检查入场信号
            if not current_position and self.check_entry_signal(current_row):
                # 开仓
                entry_price = current_row['close']
                quantity = self.calculate_position_size(entry_price)
                
                current_position = {
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'quantity': quantity,
                    'entry_rsi': current_row['rsi']
                }
        
        # 如果还有未平仓，强制平仓
        if current_position:
            final_row = df.iloc[-1]
            exit_price = final_row['close']
            quantity = current_position['quantity']
            
            pnl_usd = (exit_price - current_position['entry_price']) * quantity * self.leverage
            pnl_percent = ((exit_price - current_position['entry_price']) / current_position['entry_price']) * 100 * self.leverage
            
            commission, slippage = self.calculate_commission_slippage(exit_price, quantity)
            net_pnl = pnl_usd - commission - slippage
            capital += net_pnl
            
            holding_minutes = (df.index[-1] - current_position['entry_time']).total_seconds() / 60
            
            trade = TradeRecord(
                symbol=symbol,
                entry_time=current_position['entry_time'],
                exit_time=df.index[-1],
                entry_price=current_position['entry_price'],
                exit_price=exit_price,
                quantity=quantity,
                pnl_usd=net_pnl,
                pnl_percent=pnl_percent,
                holding_minutes=holding_minutes,
                entry_rsi=current_position['entry_rsi'],
                exit_rsi=final_row['rsi'],
                exit_reason='forced_exit',
                commission=commission,
                slippage=slippage,
                result=TradeResult.WIN if net_pnl > 0 else (TradeResult.LOSS if net_pnl < 0 else TradeResult.BREAKEVEN)
            )
            
            trades.append(trade)
        
        # 计算回测指标
        metrics = self.calculate_backtest_metrics(trades, capital, df.index[0], df.index[-1])
        
        logger.info(f"✅ {symbol} 回测完成: {len(trades)}笔交易, 胜率{metrics.win_rate:.1f}%, 总收益{metrics.total_return:.1f}%")
        
        return metrics, trades, equity_curve
    
    def calculate_backtest_metrics(self, trades: List[TradeRecord], 
                                  final_capital: float,
                                  start_date: datetime, 
                                  end_date: datetime) -> BacktestMetrics:
        """计算回测指标"""
        
        if not trades:
            return BacktestMetrics(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_return=0, annual_return=0,
                sharpe_ratio=0, max_drawdown=0, profit_factor=0,
                avg_win=0, avg_loss=0, avg_holding_minutes=0,
                largest_win=0, largest_loss=0,
                consecutive_wins=0, consecutive_losses=0
            )
        
        # 基础统计
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.result == TradeResult.WIN])
        losing_trades = len([t for t in trades if t.result == TradeResult.LOSS])
        
        win_rate = (winning_trades / total_trades) * 100
        
        # 收益计算
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        # 年化收益率
        days = (end_date - start_date).days
        years = max(days / 365.25, 0.1)
        annual_return = ((final_capital / self.initial_capital) ** (1/years) - 1) * 100
        
        # PnL统计
        pnls = [t.pnl_usd for t in trades]
        wins = [t.pnl_usd for t in trades if t.result == TradeResult.WIN]
        losses = [t.pnl_usd for t in trades if t.result == TradeResult.LOSS]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        # 盈亏比
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # 夏普比率 (简化计算)
        returns = pd.Series(pnls)
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        sharpe_ratio = sharpe_ratio * np.sqrt(252)  # 年化
        
        # 最大回撤 (简化计算)
        equity_values = [self.initial_capital]
        running_capital = self.initial_capital
        for trade in trades:
            running_capital += trade.pnl_usd
            equity_values.append(running_capital)
        
        equity_series = pd.Series(equity_values)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = abs(drawdown.min()) * 100
        
        # 连续胜负次数
        consecutive_wins = consecutive_losses = 0
        current_wins = current_losses = 0
        
        for trade in trades:
            if trade.result == TradeResult.WIN:
                current_wins += 1
                current_losses = 0
                consecutive_wins = max(consecutive_wins, current_wins)
            elif trade.result == TradeResult.LOSS:
                current_losses += 1
                current_wins = 0
                consecutive_losses = max(consecutive_losses, current_losses)
        
        # 平均持仓时间
        avg_holding_minutes = np.mean([t.holding_minutes for t in trades])
        
        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_holding_minutes=avg_holding_minutes,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses
        )
    
    def parameter_optimization(self, symbol: str) -> Dict:
        """参数优化"""
        
        logger.info(f"🔧 开始参数优化 {symbol}")
        
        # 参数组合
        rsi_ranges = [(25, 45), (30, 50), (35, 55), (40, 60)]
        ma_periods = [15, 20, 25, 30]
        profit_targets = [0.005, 0.008, 0.012, 0.015]
        
        best_params = None
        best_sharpe = -999
        results = []
        
        total_combinations = len(rsi_ranges) * len(ma_periods) * len(profit_targets)
        current_combo = 0
        
        for rsi_range, ma_period, profit_target in itertools.product(rsi_ranges, ma_periods, profit_targets):
            current_combo += 1
            
            # 设置参数
            original_config = {
                'rsi_range': self.rsi_range,
                'ma_period': self.ma_period,
                'profit_target': self.profit_target
            }
            
            self.rsi_range = rsi_range
            self.ma_period = ma_period
            self.profit_target = profit_target
            
            try:
                # 运行回测
                metrics, trades, _ = self.run_single_symbol_backtest(symbol)
                
                # 记录结果
                result = {
                    'rsi_range': rsi_range,
                    'ma_period': ma_period,
                    'profit_target': profit_target,
                    'win_rate': metrics.win_rate,
                    'total_return': metrics.total_return,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'total_trades': metrics.total_trades
                }
                results.append(result)
                
                # 更新最佳参数
                if metrics.sharpe_ratio > best_sharpe:
                    best_sharpe = metrics.sharpe_ratio
                    best_params = result.copy()
                
                logger.info(f"🔧 优化进度 {current_combo}/{total_combinations}: "
                           f"RSI{rsi_range}, MA{ma_period}, PT{profit_target:.3f} "
                           f"-> 胜率{metrics.win_rate:.1f}%, 夏普{metrics.sharpe_ratio:.2f}")
                
            except Exception as e:
                logger.error(f"❌ 参数组合失败: {e}")
            
            # 恢复原始配置
            self.rsi_range = original_config['rsi_range']
            self.ma_period = original_config['ma_period']
            self.profit_target = original_config['profit_target']
        
        logger.info(f"✅ 参数优化完成，最佳夏普比率: {best_sharpe:.2f}")
        
        return {
            'best_params': best_params,
            'all_results': results
        }

def main():
    """主函数 - 执行Phase 1策略验证"""
    
    print("🚀 DipMaster Ultra Deep Backtest - Phase 1: 策略验证")
    print("=" * 80)
    
    # 配置
    config = {
        'initial_capital': 10000,
        'commission_rate': 0.0004,
        'slippage_bps': 2.0,
        'rsi_range': (30, 50),
        'ma_period': 20,
        'min_holding_minutes': 15,
        'max_holding_minutes': 180,
        'profit_target': 0.008,
        'leverage': 10
    }
    
    # 创建回测引擎
    backtest = UltraDeepBacktest(config)
    
    # Phase 1: 单币种验证 (ICPUSDT)
    symbol = "ICPUSDT"
    
    print(f"\n📊 Phase 1: {symbol} 策略验证")
    print("-" * 50)
    
    try:
        # 运行回测
        metrics, trades, equity_curve = backtest.run_single_symbol_backtest(symbol)
        
        # 显示结果
        print(f"\n✅ 回测完成 - {symbol}")
        print(f"📈 总交易数: {metrics.total_trades}")
        print(f"🎯 胜率: {metrics.win_rate:.2f}%")
        print(f"💰 总收益: {metrics.total_return:.2f}%")
        print(f"📊 年化收益: {metrics.annual_return:.2f}%")
        print(f"⭐ 夏普比率: {metrics.sharpe_ratio:.2f}")
        print(f"📉 最大回撤: {metrics.max_drawdown:.2f}%")
        print(f"⏱️ 平均持仓: {metrics.avg_holding_minutes:.1f}分钟")
        print(f"💵 平均盈利: ${metrics.avg_win:.2f}")
        print(f"💸 平均亏损: ${metrics.avg_loss:.2f}")
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存交易记录
        trades_data = [asdict(trade) for trade in trades]
        with open(f"dipmaster_ultra_backtest_{symbol}_{timestamp}.json", 'w') as f:
            json.dump({
                'config': config,
                'metrics': asdict(metrics),
                'trades': trades_data,
                'equity_curve': [(t.isoformat(), v) for t, v in equity_curve]
            }, f, indent=2, default=str)
        
        print(f"\n💾 结果已保存到: dipmaster_ultra_backtest_{symbol}_{timestamp}.json")
        
        # Phase 1.5: 参数优化
        print(f"\n🔧 Phase 1.5: {symbol} 参数优化")
        print("-" * 50)
        
        optimization_results = backtest.parameter_optimization(symbol)
        
        print(f"\n🎯 最佳参数组合:")
        best = optimization_results['best_params']
        print(f"RSI范围: {best['rsi_range']}")
        print(f"MA周期: {best['ma_period']}")
        print(f"盈利目标: {best['profit_target']:.3f}")
        print(f"胜率: {best['win_rate']:.2f}%")
        print(f"夏普比率: {best['sharpe_ratio']:.2f}")
        
        # 保存优化结果
        with open(f"dipmaster_optimization_{symbol}_{timestamp}.json", 'w') as f:
            json.dump(optimization_results, f, indent=2)
        
        print(f"\n💾 优化结果已保存到: dipmaster_optimization_{symbol}_{timestamp}.json")
        
    except Exception as e:
        logger.error(f"❌ 回测失败: {e}")
        raise

if __name__ == "__main__":
    main()