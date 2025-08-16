#!/usr/bin/env python3
"""
测试蒙特卡洛修复 - 验证统计验证器bug修复
Test Monte Carlo Fix - Verify statistical validator bug fix

Author: Debug Team
Date: 2025-08-15
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# 直接导入StatisticalValidator
from src.validation.statistical_validator import StatisticalValidator

def create_sample_trades():
    """创建示例交易数据"""
    np.random.seed(42)  # 为了结果可重现
    
    # 创建一个有正收益的策略 (模拟一个可能有效的策略)
    n_trades = 100
    base_return = 0.002  # 基础正收益 0.2%
    noise_std = 0.015    # 噪声标准差 1.5%
    
    # 生成有轻微正偏的收益
    returns = np.random.normal(base_return, noise_std, n_trades)
    
    # 转换为PnL
    position_size = 1000
    pnl_values = returns * position_size
    
    # 创建交易DataFrame
    trades_df = pd.DataFrame({
        'pnl': pnl_values,
        'timestamp': pd.date_range('2023-01-01', periods=n_trades, freq='1H')
    })
    
    return trades_df

def test_monte_carlo_fix():
    """测试蒙特卡洛修复"""
    print("🧪 测试蒙特卡洛修复...")
    print("="*50)
    
    # 创建示例数据
    trades_df = create_sample_trades()
    
    # 计算原始指标
    original_pnl = trades_df['pnl'].sum()
    original_winrate = (trades_df['pnl'] > 0).mean()
    
    print(f"📊 原始策略表现:")
    print(f"  总PnL: {original_pnl:.2f}")
    print(f"  胜率: {original_winrate:.1%}")
    print(f"  交易数: {len(trades_df)}")
    print()
    
    # 初始化验证器
    validator = StatisticalValidator()
    
    # 运行小规模蒙特卡洛测试 (只用1000次模拟来快速测试)
    print("🎲 运行蒙特卡洛测试...")
    
    mc_pnl = validator.monte_carlo_randomization_test(
        trades_df, 'pnl', n_simulations=1000
    )
    
    mc_winrate = validator.monte_carlo_randomization_test(
        trades_df, 'win_rate', n_simulations=1000
    )
    
    print("\n📈 蒙特卡洛结果:")
    print("-" * 30)
    
    print(f"PnL测试:")
    print(f"  原始值: {mc_pnl.original_metric:.2f}")
    print(f"  随机均值: {mc_pnl.random_mean:.2f}")
    print(f"  随机标准差: {mc_pnl.random_std:.2f}")
    print(f"  P值: {mc_pnl.p_value:.4f}")
    print(f"  显著性: {'✅ 是' if mc_pnl.is_significant else '❌ 否'}")
    print(f"  百分位数: {mc_pnl.percentile_rank:.1f}%")
    print()
    
    print(f"胜率测试:")
    print(f"  原始值: {mc_winrate.original_metric:.1%}")
    print(f"  随机均值: {mc_winrate.random_mean:.1%}")
    print(f"  随机标准差: {mc_winrate.random_std:.4f}")
    print(f"  P值: {mc_winrate.p_value:.4f}")
    print(f"  显著性: {'✅ 是' if mc_winrate.is_significant else '❌ 否'}")
    print(f"  百分位数: {mc_winrate.percentile_rank:.1f}%")
    print()
    
    # 验证修复是否成功
    print("🔍 验证修复效果:")
    print("-" * 30)
    
    success_indicators = []
    
    # 1. P值不应该是1.0
    if mc_pnl.p_value < 1.0:
        success_indicators.append("✅ PnL P值不再是1.0")
    else:
        success_indicators.append("❌ PnL P值仍是1.0")
    
    if mc_winrate.p_value < 1.0:
        success_indicators.append("✅ 胜率P值不再是1.0")
    else:
        success_indicators.append("❌ 胜率P值仍是1.0")
    
    # 2. 随机均值应该接近0 (因为我们生成的是均值为0的随机交易)
    if abs(mc_pnl.random_mean) < 50:  # 应该接近0
        success_indicators.append("✅ 随机PnL均值接近0 (合理)")
    else:
        success_indicators.append(f"❌ 随机PnL均值异常: {mc_pnl.random_mean:.2f}")
    
    # 3. 随机胜率应该接近50%
    if 0.45 <= mc_winrate.random_mean <= 0.55:
        success_indicators.append("✅ 随机胜率接近50% (合理)")
    else:
        success_indicators.append(f"❌ 随机胜率异常: {mc_winrate.random_mean:.1%}")
    
    for indicator in success_indicators:
        print(f"  {indicator}")
    
    print()
    
    # 总结
    if all("✅" in indicator for indicator in success_indicators):
        print("🎉 蒙特卡洛修复成功!")
        print("   现在可以正确区分策略与随机交易的差异")
    else:
        print("⚠️  修复可能不完整，需要进一步调整")
    
    return mc_pnl, mc_winrate

if __name__ == "__main__":
    test_monte_carlo_fix()