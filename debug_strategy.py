#!/usr/bin/env python3
"""
调试简化策略 - 检查为什么没有完成交易
Debug Simple Strategy - Check why no trades are completed

Author: Debug Team
Date: 2025-08-15
"""

import pandas as pd
import numpy as np
from datetime import datetime
import ta
from pathlib import Path

def debug_strategy_signals(data_path: str, symbol: str):
    """调试策略信号生成"""
    print(f"🔍 调试 {symbol} 策略信号...")
    
    # 加载数据
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # 计算RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['price_change'] = df['close'].pct_change()
    
    # 统计RSI分布
    rsi_stats = {
        'min': df['rsi'].min(),
        'max': df['rsi'].max(),
        'mean': df['rsi'].mean(),
        'below_30': (df['rsi'] < 30).sum(),
        'above_70': (df['rsi'] > 70).sum(),
        'total_points': len(df)
    }
    
    print(f"📊 {symbol} RSI统计:")
    print(f"  最小值: {rsi_stats['min']:.2f}")
    print(f"  最大值: {rsi_stats['max']:.2f}")
    print(f"  平均值: {rsi_stats['mean']:.2f}")
    print(f"  RSI<30: {rsi_stats['below_30']} ({rsi_stats['below_30']/rsi_stats['total_points']*100:.2f}%)")
    print(f"  RSI>70: {rsi_stats['above_70']} ({rsi_stats['above_70']/rsi_stats['total_points']*100:.2f}%)")
    
    # 模拟交易信号
    positions = {}
    buy_signals = 0
    sell_signals = 0
    completed_trades = 0
    
    for i in range(14, len(df)):  # 跳过RSI计算周期
        current_row = df.iloc[i]
        timestamp = current_row['timestamp']
        price = current_row['close']
        rsi = current_row['rsi']
        price_change = current_row['price_change']
        
        if pd.isna(rsi) or pd.isna(price_change):
            continue
        
        # 检查买入条件
        if symbol not in positions:
            if rsi < 30 and price_change < 0:
                positions[symbol] = {
                    'entry_time': timestamp,
                    'entry_price': price,
                    'entry_rsi': rsi
                }
                buy_signals += 1
                # 只记录前几个买入信号
                if buy_signals <= 3:
                    print(f"  🟢 买入信号 #{buy_signals}: {timestamp}, 价格{price:.4f}, RSI{rsi:.2f}")
        
        # 检查卖出条件
        else:
            position = positions[symbol]
            holding_minutes = (timestamp - position['entry_time']).total_seconds() / 60
            
            sell_reason = None
            if rsi > 70:
                sell_reason = f"RSI超买({rsi:.1f}>70)"
            elif holding_minutes >= 60:
                sell_reason = f"持仓超时({holding_minutes:.0f}分钟>60)"
            
            if sell_reason:
                sell_signals += 1
                completed_trades += 1
                
                pnl = (price - position['entry_price']) / position['entry_price'] * 100
                print(f"  🔴 卖出信号 #{sell_signals}: {timestamp}, 价格{price:.4f}, RSI{rsi:.2f}")
                print(f"    原因: {sell_reason}")
                print(f"    持仓时间: {holding_minutes:.1f}分钟")
                print(f"    收益率: {pnl:.2f}%")
                print(f"    ---")
                
                del positions[symbol]
                
                if completed_trades >= 3:  # 只显示前3笔交易
                    break
    
    print(f"🎯 {symbol} 信号汇总:")
    print(f"  买入信号: {buy_signals}")
    print(f"  卖出信号: {sell_signals}")
    print(f"  完成交易: {completed_trades}")
    print(f"  未平仓: {len(positions)}")
    print()
    
    return {
        'symbol': symbol,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'completed_trades': completed_trades,
        'rsi_stats': rsi_stats
    }

def main():
    """主调试函数"""
    print("🚀 DipMaster策略调试工具")
    print("="*50)
    
    # 调试几个主要币种
    symbols = ['ICPUSDT', 'ADAUSDT', 'BNBUSDT']
    
    results = []
    for symbol in symbols:
        data_path = f"data/market_data/{symbol}_5m_2years.csv"
        if Path(data_path).exists():
            result = debug_strategy_signals(data_path, symbol)
            results.append(result)
        else:
            print(f"❌ 数据文件不存在: {data_path}")
    
    print("📈 总体调试汇总:")
    print("="*50)
    for result in results:
        print(f"{result['symbol']:10} | 买入:{result['buy_signals']:4} | 卖出:{result['sell_signals']:4} | 完成:{result['completed_trades']:4} | RSI>70:{result['rsi_stats']['above_70']:4}")

if __name__ == "__main__":
    main()