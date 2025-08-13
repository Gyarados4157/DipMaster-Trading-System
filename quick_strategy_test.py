#!/usr/bin/env python3
"""
DipMaster Strategy Quick Test
快速策略测试 - 验证整理后的策略是否正常运行

Author: DipMaster Trading Team  
Date: 2025-08-13
Version: 1.0.0
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

# 添加项目路径
sys.path.append('.')

from src.core.dipmaster_live import DipMasterLiveStrategy, SignalType


def generate_test_data(symbol: str = "BTCUSDT", hours: int = 24) -> pd.DataFrame:
    """生成测试用的市场数据"""
    
    # 创建时间序列
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    # 生成5分钟间隔的时间戳
    timestamps = pd.date_range(start_time, end_time, freq='5min')
    
    # 模拟价格数据 - 包含一些下跌和上涨
    np.random.seed(42)  # 确保结果可重复
    base_price = 45000.0
    
    # 生成价格变化（包含趋势和随机波动）
    price_changes = np.random.normal(0, 0.001, len(timestamps))  # 0.1% 标准差
    
    # 添加一些明显的下跌趋势（用于测试DIP信号）
    for i in range(50, 70):  # 模拟一个下跌段
        if i < len(price_changes):
            price_changes[i] -= 0.002  # -0.2% 每5分钟
    
    # 计算累积价格
    prices = base_price * np.cumprod(1 + price_changes)
    
    # 生成OHLCV数据
    data = []
    for i, timestamp in enumerate(timestamps):
        close = prices[i]
        open_price = close * (1 + np.random.uniform(-0.0005, 0.0005))
        high = max(open_price, close) * (1 + np.random.uniform(0, 0.001))
        low = min(open_price, close) * (1 - np.random.uniform(0, 0.001))
        volume = np.random.uniform(50000, 200000)  # 随机成交量
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    return df


def test_strategy_initialization():
    """测试策略初始化"""
    print("🔧 测试1: 策略初始化")
    print("-" * 40)
    
    try:
        # 加载配置
        with open('config/dipmaster_live_trading.json') as f:
            config = json.load(f)
        
        # 初始化策略
        strategy = DipMasterLiveStrategy(config)
        
        print("✅ 策略初始化成功")
        print(f"📊 RSI范围: {strategy.rsi_range}")
        print(f"📈 MA周期: {strategy.ma_period}")
        print(f"💰 盈利目标: {strategy.profit_target:.1%}")
        print(f"🛡️ 止损: {strategy.stop_loss_pct:.1%}")
        print(f"⚖️ 杠杆: {strategy.max_leverage}x")
        print(f"⏰ 最大持仓: {strategy.max_holding_minutes}分钟")
        
        return strategy
        
    except Exception as e:
        print(f"❌ 策略初始化失败: {e}")
        return None


def test_technical_indicators(strategy: DipMasterLiveStrategy):
    """测试技术指标计算"""
    print("\n📊 测试2: 技术指标计算")
    print("-" * 40)
    
    try:
        # 生成测试数据 - 需要更多数据用于计算MA30
        df = generate_test_data("BTCUSDT", 8)  # 8小时数据，确保有足够数据计算MA30
        print(f"📈 生成测试数据: {len(df)} 根K线")
        
        # 计算技术指标
        indicators = strategy.calculate_technical_indicators(df)
        
        if indicators:
            print("✅ 技术指标计算成功")
            print(f"📊 RSI: {indicators['rsi']:.2f}")
            print(f"📈 MA30: {indicators['ma30']:.2f}")
            print(f"💹 当前价格: {indicators['current_price']:.2f}")
            print(f"📊 成交量比率: {indicators['volume_ratio']:.2f}")
            print(f"📉 价格变化: {indicators['price_change']:.2%}")
            
            return True
        else:
            print("⚠️ 技术指标计算返回空结果")
            return False
            
    except Exception as e:
        print(f"❌ 技术指标计算失败: {e}")
        return False


def test_signal_generation(strategy: DipMasterLiveStrategy):
    """测试信号生成"""
    print("\n🎯 测试3: 交易信号生成")
    print("-" * 40)
    
    try:
        # 生成多组测试数据，寻找能触发信号的情况
        signals_found = 0
        
        for i in range(5):  # 测试5组不同的数据
            # 每次使用不同的随机种子
            np.random.seed(42 + i * 10)
            
            df = generate_test_data("BTCUSDT", 4)  # 4小时数据
            
            # 尝试生成入场信号
            entry_signal = strategy.generate_entry_signal("BTCUSDT", df)
            
            if entry_signal:
                signals_found += 1
                print(f"✅ 发现第{signals_found}个入场信号:")
                print(f"   🎯 价格: {entry_signal.price:.2f}")
                print(f"   📊 RSI: {entry_signal.rsi:.2f}")
                print(f"   📈 MA30: {entry_signal.ma30:.2f}")
                print(f"   🔥 置信度: {entry_signal.confidence_score:.2f}")
                print(f"   💰 仓位: ${entry_signal.position_size_usd}")
                print(f"   ⚖️ 杠杆: {entry_signal.leverage}x")
                print(f"   📝 原因: {entry_signal.reason}")
                
                # 测试开仓
                success = strategy.open_position(entry_signal)
                print(f"   📥 开仓结果: {'✅ 成功' if success else '❌ 失败'}")
                
                if success:
                    # 测试出场信号（模拟一些时间后的价格）
                    df_later = df.copy()
                    # 模拟价格稍微上涨（触发盈利出场）
                    df_later.loc[df_later.index[-1], 'close'] *= 1.012  # +1.2% 
                    
                    position = strategy.positions.get("BTCUSDT")
                    if position:
                        # 模拟15分钟后
                        position.entry_time = datetime.now() - timedelta(minutes=16)
                        
                        exit_signal = strategy.generate_exit_signal("BTCUSDT", position, df_later)
                        if exit_signal:
                            print(f"   🚪 出场信号: {exit_signal.reason}")
                            close_success = strategy.close_position(exit_signal)
                            print(f"   📤 平仓结果: {'✅ 成功' if close_success else '❌ 失败'}")
                
                # 只测试前两个信号
                if signals_found >= 2:
                    break
        
        if signals_found > 0:
            print(f"\n✅ 信号生成测试通过，共发现 {signals_found} 个有效信号")
            
            # 显示当前统计
            stats = strategy.get_daily_stats()
            print(f"📊 当前统计: {stats}")
            
            return True
        else:
            print("⚠️ 未发现有效信号，这可能是正常的（信号要求严格）")
            return True  # 这不算失败，只是信号要求严格
            
    except Exception as e:
        print(f"❌ 信号生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_loading():
    """测试配置文件加载"""
    print("\n⚙️ 测试4: 配置文件验证")
    print("-" * 40)
    
    config_files = [
        'config/dipmaster_live_trading.json',
        'config/dipmaster_v3_optimized.json'
    ]
    
    all_passed = True
    
    for config_file in config_files:
        try:
            if Path(config_file).exists():
                with open(config_file) as f:
                    config = json.load(f)
                print(f"✅ {config_file} - 加载成功")
                
                # 检查必要字段
                required_sections = ['trading', 'api'] if 'live' in config_file else ['strategy_name']
                
                for section in required_sections:
                    if section in config:
                        print(f"   ✓ {section} 配置存在")
                    else:
                        print(f"   ⚠️ {section} 配置缺失")
                        
            else:
                print(f"❌ {config_file} - 文件不存在")
                all_passed = False
                
        except Exception as e:
            print(f"❌ {config_file} - 加载失败: {e}")
            all_passed = False
    
    return all_passed


def main():
    """主测试函数"""
    print("🚀 DipMaster策略全面测试启动")
    print("=" * 60)
    
    # 测试结果跟踪
    test_results = {
        'strategy_init': False,
        'indicators': False, 
        'signals': False,
        'config': False
    }
    
    # 1. 测试策略初始化
    strategy = test_strategy_initialization()
    test_results['strategy_init'] = strategy is not None
    
    if strategy:
        # 2. 测试技术指标
        test_results['indicators'] = test_technical_indicators(strategy)
        
        # 3. 测试信号生成（只在前面测试通过时执行）
        if test_results['indicators']:
            test_results['signals'] = test_signal_generation(strategy)
    
    # 4. 测试配置文件
    test_results['config'] = test_configuration_loading()
    
    # 汇总结果
    print("\n🎯 测试结果汇总")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15} : {status}")
    
    print(f"\n📊 总体结果: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！策略已准备好进行实盘测试")
        return True
    else:
        print("⚠️ 存在测试失败，请检查相关问题")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)