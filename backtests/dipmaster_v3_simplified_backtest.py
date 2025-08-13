#!/usr/bin/env python3
"""
DipMaster V3 简化深度回测
针对ICPUSDT进行2年期回测，重点验证DIP策略和大额亏损分析
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedDipBacktest:
    """简化DipMaster V3回测器"""
    
    def __init__(self):
        self.initial_capital = 10000
        self.current_capital = 10000
        self.commission_rate = 0.0004
        self.slippage_bps = 2.0
        
        # DipMaster策略参数
        self.rsi_range = (30, 50)
        self.max_positions = 3
        self.base_position_size = 1000
        self.max_holding_minutes = 180
        
        # 风险管理参数
        self.emergency_stop = 0.005  # 0.5%紧急止损
        self.profit_levels = [(0.008, 0.25), (0.015, 0.35), (0.025, 0.4)]
        
        # 交易记录
        self.trades = []
        self.current_positions = {}
        self.equity_curve = []
        
    def load_data(self) -> pd.DataFrame:
        """加载ICPUSDT数据"""
        data_file = "data/market_data/ICPUSDT_5m_2years.csv"
        logger.info(f"📊 加载数据: {data_file}")
        
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # 计算技术指标
        df = self.calculate_indicators(df)
        
        logger.info(f"✅ 数据加载完成: {len(df)}条, 时间范围: {df.index[0]} 到 {df.index[-1]}")
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # MA20
        df['ma20'] = df['close'].rolling(20).mean()
        
        # 成交量MA
        df['volume_ma'] = df['volume'].rolling(20).mean()
        
        # ATR
        df['atr'] = self.calculate_atr(df)
        
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
        # 必须有足够的历史数据
        if pd.isna(row['rsi']) or pd.isna(row['ma20']):
            return False
        
        # 1. RSI在30-50区间
        if not (self.rsi_range[0] <= row['rsi'] <= self.rsi_range[1]):
            return False
        
        # 2. 价格在MA20下方（87%概率的DipMaster特征）
        if row['close'] >= row['ma20']:
            return False
        
        # 3. 是否为下跌K线（逢跌买入）
        if row['close'] >= row['open']:
            return False
        
        # 4. 成交量放大确认
        if not pd.isna(row['volume_ma']):
            if row['volume'] < row['volume_ma'] * 1.2:
                return False
        
        return True
    
    def create_position(self, symbol: str, price: float, timestamp: datetime) -> Dict:
        """创建持仓"""
        position_size_usd = self.base_position_size
        quantity = position_size_usd / price
        
        position = {
            'symbol': symbol,
            'entry_time': timestamp,
            'entry_price': price,
            'quantity': quantity,
            'position_size_usd': position_size_usd,
            'stop_loss': price * (1 - self.emergency_stop),
            'profit_levels_hit': [],
            'remaining_quantity': quantity
        }
        
        return position
    
    def check_exit_signals(self, position: Dict, current_price: float, current_time: datetime) -> List[Dict]:
        """检查出场信号"""
        exit_signals = []
        
        # 计算当前盈亏
        pnl_percent = (current_price - position['entry_price']) / position['entry_price']
        holding_minutes = (current_time - position['entry_time']).total_seconds() / 60
        
        # 1. 紧急止损
        if current_price <= position['stop_loss']:
            exit_signals.append({
                'action': 'SELL_ALL',
                'reason': 'emergency_stop',
                'price': current_price,
                'quantity_ratio': 1.0
            })
            return exit_signals
        
        # 2. 分层止盈
        for i, (profit_threshold, exit_ratio) in enumerate(self.profit_levels):
            if i not in position['profit_levels_hit'] and pnl_percent >= profit_threshold:
                exit_signals.append({
                    'action': 'SELL_PARTIAL',
                    'reason': f'profit_level_{i+1}',
                    'price': current_price,
                    'quantity_ratio': exit_ratio
                })
                position['profit_levels_hit'].append(i)
        
        # 3. 15分钟边界出场（DipMaster核心特征）
        if holding_minutes >= 15 and current_time.minute in [15, 30, 45, 0]:
            if pnl_percent > 0:
                exit_signals.append({
                    'action': 'SELL_ALL',
                    'reason': 'boundary_profit',
                    'price': current_price,
                    'quantity_ratio': 1.0
                })
            elif pnl_percent > -0.005:  # 小幅亏损也可以在边界出场
                exit_signals.append({
                    'action': 'SELL_ALL',
                    'reason': 'boundary_neutral',
                    'price': current_price,
                    'quantity_ratio': 1.0
                })
        
        # 4. 最大持仓时间
        if holding_minutes >= self.max_holding_minutes:
            exit_signals.append({
                'action': 'SELL_ALL',
                'reason': 'max_time',
                'price': current_price,
                'quantity_ratio': 1.0
            })
        
        return exit_signals
    
    def execute_trade(self, signal: Dict, position: Dict) -> Dict:
        """执行交易"""
        quantity_traded = position['quantity'] * signal['quantity_ratio']
        position_value = quantity_traded * signal['price']
        
        # 计算成本
        commission = position_value * self.commission_rate
        slippage = position_value * (self.slippage_bps / 10000)
        total_costs = commission + slippage
        
        # 计算盈亏
        entry_value = quantity_traded * position['entry_price']
        pnl_usd = position_value - entry_value - total_costs
        pnl_percent = (signal['price'] - position['entry_price']) / position['entry_price'] * 100
        
        # 更新资金
        self.current_capital += (position_value - total_costs)
        
        # 创建交易记录
        trade = {
            'symbol': position['symbol'],
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),  # 使用当前时间作为简化
            'entry_price': position['entry_price'],
            'exit_price': signal['price'],
            'quantity': quantity_traded,
            'pnl_usd': pnl_usd,
            'pnl_percent': pnl_percent,
            'holding_minutes': (datetime.now() - position['entry_time']).total_seconds() / 60,
            'exit_reason': signal['reason'],
            'commission': commission,
            'slippage': slippage
        }
        
        # 更新持仓
        position['remaining_quantity'] -= quantity_traded
        
        return trade
    
    def run_backtest(self) -> Dict:
        """运行回测"""
        logger.info("🚀 开始简化DipMaster V3回测...")
        
        # 加载数据
        df = self.load_data()
        
        # 初始化
        self.current_capital = self.initial_capital
        self.trades = []
        self.current_positions = {}
        
        total_signals = 0
        dip_entries = 0
        boundary_exits = 0
        
        # 逐行回测
        logger.info("⏳ 开始逐行回测...")
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            if i % 10000 == 0:
                progress = i / len(df) * 100
                logger.info(f"回测进度: {progress:.1f}% ({timestamp})")
            
            # 检查出场信号
            positions_to_remove = []
            for symbol, position in self.current_positions.items():
                exit_signals = self.check_exit_signals(position, row['close'], timestamp)
                
                for signal in exit_signals:
                    trade = self.execute_trade(signal, position)
                    trade['exit_time'] = timestamp  # 使用实际时间
                    trade['holding_minutes'] = (timestamp - position['entry_time']).total_seconds() / 60
                    self.trades.append(trade)
                    
                    if 'boundary' in signal['reason']:
                        boundary_exits += 1
                    
                    # 如果全部平仓，删除持仓
                    if signal['quantity_ratio'] >= 1.0 or position['remaining_quantity'] <= 0:
                        positions_to_remove.append(symbol)
                        break
            
            # 删除已平仓的持仓
            for symbol in positions_to_remove:
                if symbol in self.current_positions:
                    del self.current_positions[symbol]
            
            # 检查入场信号
            if len(self.current_positions) < self.max_positions:
                if self.check_entry_signal(row):
                    total_signals += 1
                    
                    # 检查是否为逢跌买入
                    if row['close'] < row['open']:
                        dip_entries += 1
                    
                    # 创建新持仓
                    symbol = 'ICPUSDT'
                    position = self.create_position(symbol, row['close'], timestamp)
                    self.current_positions[symbol] = position
                    
                    # 扣除入场成本
                    entry_cost = position['position_size_usd'] * (self.commission_rate + self.slippage_bps / 10000)
                    self.current_capital -= entry_cost
            
            # 记录权益曲线
            if i % 100 == 0:  # 每100条记录一次
                unrealized_pnl = 0
                for position in self.current_positions.values():
                    unrealized_pnl += (row['close'] - position['entry_price']) * position['remaining_quantity']
                
                total_equity = self.current_capital + unrealized_pnl
                self.equity_curve.append((timestamp, total_equity))
        
        # 强制平仓剩余持仓
        final_price = df['close'].iloc[-1]
        final_time = df.index[-1]
        
        for symbol, position in self.current_positions.items():
            signal = {
                'action': 'SELL_ALL',
                'reason': 'backtest_end',
                'price': final_price,
                'quantity_ratio': 1.0
            }
            trade = self.execute_trade(signal, position)
            trade['exit_time'] = final_time
            trade['holding_minutes'] = (final_time - position['entry_time']).total_seconds() / 60
            self.trades.append(trade)
        
        # 计算结果
        results = self.calculate_results(total_signals, dip_entries, boundary_exits)
        
        logger.info("✅ 回测完成!")
        return results
    
    def calculate_results(self, total_signals: int, dip_entries: int, boundary_exits: int) -> Dict:
        """计算回测结果"""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        # 基础统计
        wins = [t for t in self.trades if t['pnl_usd'] > 0]
        losses = [t for t in self.trades if t['pnl_usd'] < 0]
        
        total_trades = len(self.trades)
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        
        # 盈亏统计
        total_pnl = sum(t['pnl_usd'] for t in self.trades)
        total_return = (self.current_capital / self.initial_capital - 1) * 100
        
        gross_profit = sum(t['pnl_usd'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl_usd'] for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        avg_win = np.mean([t['pnl_usd'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_usd'] for t in losses]) if losses else 0
        
        # 持仓时间统计
        holding_times = [t['holding_minutes'] for t in self.trades]
        avg_holding = np.mean(holding_times) if holding_times else 0
        
        # DipMaster特征分析
        dip_rate = dip_entries / total_signals * 100 if total_signals > 0 else 0
        boundary_rate = boundary_exits / total_trades * 100 if total_trades > 0 else 0
        
        # 大额亏损分析
        large_losses = [t for t in losses if abs(t['pnl_usd']) >= 100]
        max_loss = min([t['pnl_usd'] for t in losses]) if losses else 0
        
        # 连续亏损分析
        consecutive_losses = 0
        max_consecutive_losses = 0
        current_streak = 0
        
        for trade in self.trades:
            if trade['pnl_usd'] < 0:
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0
        
        # 权益曲线分析
        equity_values = [eq[1] for eq in self.equity_curve] if self.equity_curve else [self.initial_capital, self.current_capital]
        peak_equity = np.maximum.accumulate(equity_values)
        drawdowns = (peak_equity - equity_values) / peak_equity * 100
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # 风险指标
        daily_returns = np.diff(equity_values) / equity_values[:-1] if len(equity_values) > 1 else []
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0
        
        results = {
            'performance_metrics': {
                'total_trades': total_trades,
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'win_rate': win_rate,
                'total_return': total_return,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'avg_holding_minutes': avg_holding
            },
            'dipmaster_characteristics': {
                'total_signals': total_signals,
                'dip_entries': dip_entries,
                'dip_entry_rate': dip_rate,
                'boundary_exits': boundary_exits,
                'boundary_exit_rate': boundary_rate,
                'avg_holding_minutes': avg_holding
            },
            'risk_analysis': {
                'large_losses_count': len(large_losses),
                'largest_single_loss': max_loss,
                'max_consecutive_losses': max_consecutive_losses,
                'total_losing_amount': gross_loss,
                'loss_rate': len(losses) / total_trades * 100
            },
            'trade_details': self.trades,
            'final_capital': self.current_capital,
            'timestamp': datetime.now().isoformat()
        }
        
        return results

def main():
    """主函数"""
    print("🎯 DipMaster V3 简化深度回测")
    print("=" * 50)
    
    backtest = SimplifiedDipBacktest()
    results = backtest.run_backtest()
    
    if 'error' in results:
        print(f"❌ 回测失败: {results['error']}")
        return 1
    
    # 显示结果
    perf = results['performance_metrics']
    dip = results['dipmaster_characteristics']
    risk = results['risk_analysis']
    
    print(f"\n📊 回测结果摘要:")
    print(f"总交易数: {perf['total_trades']}")
    print(f"胜率: {perf['win_rate']:.1f}%")
    print(f"总收益: {perf['total_return']:.1f}%")
    print(f"盈亏比: {perf['profit_factor']:.2f}")
    print(f"最大回撤: {perf['max_drawdown']:.1f}%")
    print(f"夏普率: {perf['sharpe_ratio']:.2f}")
    
    print(f"\n🎯 DipMaster策略特征:")
    print(f"总信号数: {dip['total_signals']}")
    print(f"逢跌买入数: {dip['dip_entries']}")
    print(f"逢跌买入率: {dip['dip_entry_rate']:.1f}%")
    print(f"边界出场数: {dip['boundary_exits']}")
    print(f"边界出场率: {dip['boundary_exit_rate']:.1f}%")
    print(f"平均持仓: {dip['avg_holding_minutes']:.1f}分钟")
    
    print(f"\n⚠️ 风险分析:")
    print(f"亏损交易数: {risk['loss_rate']:.1f}%")
    print(f"大额亏损数: {risk['large_losses_count']}")
    print(f"最大单笔亏损: ${risk['largest_single_loss']:.2f}")
    print(f"最大连续亏损: {risk['max_consecutive_losses']}笔")
    
    # 评估DipMaster复刻效果
    print(f"\n🔍 DipMaster AI复刻评估:")
    dip_target = 87.9  # 原版逢跌买入率
    boundary_target = 100  # 原版边界出场率
    
    dip_score = "✅ 优秀" if dip['dip_entry_rate'] >= 80 else "⚠️ 需要改进"
    boundary_score = "✅ 优秀" if dip['boundary_exit_rate'] >= 80 else "⚠️ 需要改进"
    
    print(f"逢跌买入复刻: {dip_score} ({dip['dip_entry_rate']:.1f}% vs 目标87.9%)")
    print(f"边界出场复刻: {boundary_score} ({dip['boundary_exit_rate']:.1f}% vs 目标100%)")
    
    # 风险评估
    print(f"\n🛡️ 大额亏损风险评估:")
    if risk['max_consecutive_losses'] <= 3:
        print("✅ 连续亏损控制良好")
    elif risk['max_consecutive_losses'] <= 5:
        print("⚠️ 连续亏损需要关注")
    else:
        print("❌ 连续亏损风险较高")
    
    if perf['max_drawdown'] <= 3:
        print("✅ 回撤控制优秀")
    elif perf['max_drawdown'] <= 5:
        print("⚠️ 回撤控制良好")
    else:
        print("❌ 回撤风险较高")
    
    # 保存结果
    results_file = f"dipmaster_v3_simplified_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📄 详细结果已保存: {results_file}")
    print("🎉 回测完成!")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)